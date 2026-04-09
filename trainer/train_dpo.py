import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, \
    init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    将logits转换为对数概率
    :param logits: logits是一个张量，形状为(batch_size, seq_len, vocab_size)
    :param labels: labels是一个张量，形状为(batch_size, seq_len)
    :return: 返回一个形状为(batch_size, seq_len)的对数概率张量log_probs_per_token
    """
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    # 使用log_softmax函数，对logits在维度2上进行操作，得到对数概率log_probs
    log_probs = F.log_softmax(logits, dim=2)
    # 根据labels从log_probs中收集对应位置的值，并去除最后一维
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算DPO（Direct Preference Optimization）损失
    :param ref_log_probs: 参考对数概率，形状为(batch_size, seq_len)
    :param policy_log_probs: 策略对数概率，形状为(batch_size, seq_len)
    :param mask: 掩码张量，用于屏蔽某些位置
    :param beta: 一个超参数，用于调整损失计算中的权重
    :return: 返回平均的DPO损失值
    """
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    # 1. 对每个序列求和（得到整句对数概率）
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开
    # 2. 拆分 chosen（好答案）、rejected（差答案）
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 3. 参考模型的好坏答案差
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs

    # 4. 核心 DPO 偏好差
    logits = pi_logratios - ref_logratios

    # 5. 最终损失
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """
    训练循环
    :param epoch: epoch表示当前是第几个epoch
    :param loader: loader是数据加载器，用于按批次加载训练数据
    :param iters: iters表示一个epoch中总的迭代步数
    :param ref_model: 参考模型，用于生成参考对数概率
    :param lm_config: 语言模型相关的配置
    :param start_step: start_step表示从哪个步骤开始，默认从0开始
    :param wandb: wandb是一个用于实验跟踪和可视化的工具，这里如果传入就可以记录相关指标
    :param beta: 用于计算DPO损失的超参数，默认值为0.1
    :return: 无返回值，该函数主要用于训练模型并记录相关指标
    """
    start_time = time.time()  # 记录训练开始时间
    last_step = start_step  # 记录上一步的步数

    for step, batch in enumerate(loader, start=start_step + 1):  # 从数据加载器中按批次取出数据
        last_step = step
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)
        # 根据当前步数和总步数计算学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新优化器中每个参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:  # 使用自动混合精度上下文
            with torch.no_grad():  # 不计算梯度
                ref_outputs = ref_model(x)  # 通过参考模型得到输出
                ref_logits = ref_outputs.logits  # 参考模型的原始输出分数 是参考模型最后一层线性层的输出
            ref_log_probs = logits_to_log_probs(ref_logits, y)  # 把 logits 转成对数概率

            outputs = model(x)  # 通过当前模型得到输出
            logits = outputs.logits  # 获取当前模型输出的logits
            policy_log_probs = logits_to_log_probs(logits, y)  # 将当前模型的logits转换为对数概率
            # 用 ref_log_probs 和 policy_log_probs 一起计算 DPO 损失
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)  # 计算DPO损失
            # 总损失 = DPO损失 + 辅助损失(aux_loss)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps  # 根据梯度累积步数调整损失

        scaler.scale(loss).backward()  # 使用梯度缩放器缩放损失并反向传播计算梯度

        if step % args.accumulation_steps == 0:  # 当达到梯度累积步数时
            scaler.unscale_(optimizer)  # 取消梯度缩放
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 裁剪梯度范数
            scaler.step(optimizer)  # 使用优化器更新模型参数
            scaler.update()  # 更新梯度缩放器
            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        if step % args.log_interval == 0 or step == iters:  # 当达到日志记录间隔步数或者是最后一步时
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            # 计算当前的辅助损失
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']  # 获取当前的学习率
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60  # 计算预计剩余时间（分钟）

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, '
                f'dpo_loss: {current_dpo_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, '
                f'learning_rate: {current_lr:.8f}, '
                f'epoch_time: {eta_min:.3f}min'
            )

            if wandb: wandb.log({
                "loss"         : current_loss,
                "dpo_loss"     : current_dpo_loss,
                "aux_loss"     : current_aux_loss,
                "learning_rate": current_lr,
                "epoch_time"   : eta_min
            })
        # 当达到保存间隔步数或者是最后一步，并且是主进程时
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 获取原始模型（处理可能的包装）
            state_dict = raw_model.state_dict()  # 获取模型状态字典
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss

    # 如果最后一步步数大于起始步数且不是梯度累积步数的整数倍
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)  # 取消梯度缩放
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 裁剪梯度范数
        scaler.step(optimizer)  # 使用优化器更新模型参数
        scaler.update()  # 更新梯度缩放器
        optimizer.zero_grad(set_to_none=True)  # 清空梯度


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe))
    ckp_data = (lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
                if args.from_resume == 1 else None)

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型和参考模型 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 初始化参考模型（ref_model冻结）
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()

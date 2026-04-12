import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import warnings
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, \
    init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

import torch
import time
from torch.cuda.amp import autocast as autocast_ctx
from torch.nn import functional as F


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """
    计算蒸馏损失
    :param student_logits: 学生模型的logits输出
    :param teacher_logits: 教师模型的logits输出
    :param temperature: 温度参数，默认为1.0，用于控制softmax分布的平滑程度
    :param reduction: 损失归约方式，默认为'batchmean'，表示按批次平均
    :return: 返回计算得到的蒸馏损失
    """
    with torch.no_grad():
        # 将教师模型的logits经过softmax处理并除以温度参数，再detach防止反向传播
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 将学生模型的logits经过log_softmax处理并除以温度参数
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # 计算KL散度损失
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    # 返回经过温度参数平方缩放后的KL散度损失
    """
    1. 调整蒸馏损失的尺度
        温度参数的作用：温度 temperature 在知识蒸馏中用于软化教师模型和学生模型输出的概率分布。
            较高的温度会使概率分布更加平滑，突出那些原本概率较小的类别信息。
            这有助于学生模型学习到教师模型预测中隐含的更多知识，尤其是对于那些非明显正确答案的类别之间的相对关系。
        尺度调整：当温度升高时，softmax 输出的概率分布更加平缓，使得不同类别之间的差异变小。这意味着直接计算的 KL 散度值也会相应变小。
            为了使蒸馏损失在整个训练过程中有合适的尺度，不至于因为温度的作用而变得过小，从而失去对学生模型训练的有效指导，
            所以乘以 (temperature ** 2) 来放大这个损失值。这样调整后，蒸馏损失在总损失（结合交叉熵损失等）中能够保持合适的比重，
            合理地引导学生模型向教师模型学习。
    2. 与理论推导保持一致
        在知识蒸馏的理论推导中，从信息论和优化目标的角度来看，乘以 (temperature ** 2) 是一种经过推导得出的合理操作。
        这种形式能够在最小化学生模型和教师模型分布差异的同时，更好地平衡不同温度设置下的优化效果，使得整个知识蒸馏过程在理论上更加严谨和自洽。
    """
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0,
                temperature=1.0):
    """
    训练一个epoch的函数
    :param epoch: 当前所处的epoch数
    :param loader: 数据加载器，用于按批次加载训练数据
    :param iters: 一个epoch中的总迭代步数
    :param teacher_model: 教师模型，如果为None则不进行知识蒸馏
    :param lm_config_student: 学生模型的配置信息
    :param start_step: 从哪个步骤开始，默认从0开始
    :param wandb: 用于实验跟踪和可视化的工具，如果传入则可以记录相关指标
    :param alpha: 控制CE损失和蒸馏损失权重的参数，默认为0.0
    :param temperature: 蒸馏损失中的温度参数，默认为1.0
    :return: 无返回值
    """
    start_time = time.time()
    last_step = start_step

    # 如果教师模型存在，则设置为评估模式且不需要计算梯度
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        last_step = step
        # 将输入数据和标签移动到指定设备上
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 创建损失掩码，用于过滤掉标签中值为-100的部分
        loss_mask = (labels[..., 1:] != -100).float()
        # 根据当前的epoch和step计算学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 学生模型前向传播
        with autocast_ctx:
            res = model(input_ids)
            # 取出学生模型logits中除最后一个时间步的部分，并调整形状
            # .contiguous()将不连续的数据存储调整为连续的数据储存
            student_logits = res.logits[..., :-1, :].contiguous()

        # 教师模型前向传播（只在评估模式且不计算梯度）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                # 获取学生模型logits的词汇表大小
                vocab_size_student = student_logits.size(-1)
                # 调整教师模型logits的词汇表大小与学生模型一致
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 1) 计算基于真实标签的交叉熵损失（CE Loss）
        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1)
        """
        在多分类场景下，交叉熵损失衡量的是模型预测的概率分布与真实标签分布之间的差异。
            对于一个具有 C 个类别的分类任务，模型会为每个样本输出一个长度为 C 的向量，表示该样本属于每个类别的概率（经过 `softmax` 等操作后）。
            真实标签通常采用独热编码（One - Hot Encoding）表示，即在对应正确类别的位置为 1，其他位置为 0。
        student_logits 是学生模型输出的未经 softmax 激活的对数概率（logits），通过 view(-1, student_logits.size(-1)) 
            将其重塑为二维张量，第一维 -1 表示自动计算该维度的大小以匹配总元素数量，第二维是类别维度。
        shift_labels 是真实标签，同样通过 view(-1) 展平为一维张量。
        ignore_index=-100 表示在计算损失时忽略标签值为 -100 的样本，这在处理序列数据中某些不需要计算损失的位置（如填充位置）很有用。
        reduction='none' 表示对每个样本单独计算交叉熵损失，而不进行任何形式的归约（如求和、平均等），返回一个与样本数量相同长度的损失向量。
            后续代码再根据需要对这些单独的损失值进行处理（如根据掩码计算平均值等）。
        """
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        # 如果学生模型使用了MoE（混合专家模型），则CE损失加上辅助损失
        ce_loss = ce_loss_raw + res.aux_loss if lm_config_student.use_moe else ce_loss_raw

        # 2) 计算蒸馏损失（Distillation Loss）
        if teacher_model is not None:
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 计算总损失，由CE损失和蒸馏损失按权重相加得到
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # 使用梯度缩放器对损失进行反向传播
        scaler.scale(loss).backward()

        # 每积累一定步数进行一次优化器更新
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 每间隔一定步数或到达总步数时记录日志和指标
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, '
                f'ce: {current_ce_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, '
                f'distill: {distill_loss.item():.4f}, '
                f'learning_rate: {current_lr:.8f},'
                f' epoch_time: {eta_min:.3f}min'
            )

            if wandb:
                wandb.log({
                    "loss"        : current_loss,
                    "ce_loss"     : current_ce_loss,
                    "aux_loss"    : current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time"  : eta_min
                })

        # 每间隔一定步数或到达总步数时保存模型
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 删除中间变量，释放内存
        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss

    # 如果最后一步没有达到积累步数，也进行一次优化器更新
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # 模拟用moe模型蒸馏dense模型，也可以用更大teacher_hidden_size模型蒸馏更小student_hidden_size的
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--student_hidden_size', default=768, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--student_use_moe', default=0, type=int, choices=[0, 1], help="学生模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1], help="教师模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers,
                                       use_moe=bool(args.student_use_moe))
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers,
                                       use_moe=bool(args.teacher_use_moe))
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

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
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义学生和教师模型 ==========
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student, start_step, wandb,
                        args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha,
                        args.temperature)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()

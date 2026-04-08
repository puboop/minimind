import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, \
    init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练循环
    :param epoch: epoch表示当前是第几个epoch
    :param loader: loader是数据加载器，用于按批次加载训练数据
    :param iters: iters表示一个epoch中总的迭代步数
    :param start_step: start_step表示从哪个步骤开始，默认从0开始
    :param wandb: wandb是一个用于实验跟踪和可视化的工具，这里如果传入就可以记录相关指标
    :return:
    """
    # 记录训练开始时间
    start_time = time.time()
    # 记录上一步的步数，初始化为start_step
    last_step = start_step
    # 遍历数据加载器中的数据，从start_step + 1开始计数
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将输入的id数据移动到指定的设备（比如GPU）上
        input_ids = input_ids.to(args.device)
        # 将标签数据移动到指定的设备（比如GPU）上
        labels = labels.to(args.device)
        # 更新last_step为当前步骤
        last_step = step
        # 随着训练步数的增加，学习率会按照上述从初始值逐渐减小到初始值 0.1 倍的规律变化，这种学习率调整方式通常被称为余弦退火学习率调整策略，
        # 它可以在训练后期让学习率逐渐变小，有助于模型更稳定地收敛，避免训练后期因学习率过大而错过最优解。
        # 根据当前的总步数（epoch * iters + step）、总的训练步数（args.epochs * iters）和初始学习率（args.learning_rate）计算当前学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 遍历优化器中的参数组，更新每个参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用自动混合精度上下文，在这个上下文内的计算会自动选择合适的精度（比如FP16）以加速计算
        with autocast_ctx:
            # 将输入数据传入模型，得到模型的输出结果
            res = model(input_ids, labels=labels)
            # 计算损失，损失由主损失和辅助损失相加得到
            loss = res.loss + res.aux_loss
            # 由于使用了梯度累积，这里将损失除以累积步数，得到平均损失
            loss = loss / args.accumulation_steps

        # 使用自动混合精度的缩放器对损失进行缩放后反向传播计算梯度
        scaler.scale(loss).backward()

        # 当达到梯度累积步数时
        if step % args.accumulation_steps == 0:
            # 对优化器进行反缩放，因为之前缩放了损失
            """
            自动混合精度（AMP）背景：
                在深度学习训练中，为了利用 GPU 的加速能力并减少内存使用，自动混合精度（AMP）被广泛使用。AMP 通过自动在不同精度（如 FP16 和 FP32）之间切换来加速计算。
                在这个过程中，scaler（缩放器）起到了关键作用。由于 FP16 表示的数值范围较小，容易出现梯度下溢（即梯度值太小，无法用 FP16 准确表示）的问题。
                为了避免这种情况，scaler会在反向传播前对损失进行放大（scale），这样即使梯度很小，放大后也能在 FP16 中准确表示。
            scaler.unscale_(optimizer)的作用：
                在反向传播完成后，梯度已经被放大了。而优化器（如optimizer）在更新参数时，需要使用真实的梯度值。所以在使用优化器更新参数之前，需要将放大后的梯度恢复到原来的大小，
                这就是scaler.unscale_(optimizer)的作用，它对优化器中的梯度进行反缩放，将梯度还原到正常大小。
            为什么要这么做：
                如果不进行反缩放，优化器会使用放大后的梯度来更新参数，这将导致参数更新的步长过大，严重影响模型的训练效果，可能导致模型无法收敛甚至发散。
                通过反缩放，优化器能够基于正确的梯度值来更新模型参数，从而保证模型训练的稳定性和准确性。例如，在使用 Adam 优化器时，它根据梯度计算参数更新量，
                如果梯度是被放大的错误值，那么计算出的参数更新量也会是错误的，最终使得模型训练偏离正确方向。
                而scaler.unscale_(optimizer)确保了优化器使用的梯度是正确的，让模型能够按照预期进行训练。
            """
            scaler.unscale_(optimizer)
            # 对模型的参数梯度进行裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 使用缩放器更新优化器的参数
            scaler.step(optimizer)
            # 更新缩放器的状态
            scaler.update()

            # 清空优化器的梯度，这里设置为None以节省内存
            optimizer.zero_grad(set_to_none=True)

        # 当达到日志记录间隔步数或者当前是最后一步时
        if step % args.log_interval == 0 or step == iters:
            # 计算从训练开始到现在花费的时间
            spend_time = time.time() - start_time
            # 计算当前的实际损失（考虑梯度累积）
            current_loss = loss.item() * args.accumulation_steps
            # 获取辅助损失的值，如果辅助损失不存在则设为0.0
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # 计算主损失（总损失减去辅助损失）
            current_logits_loss = current_loss - current_aux_loss
            # 获取当前的学习率
            current_lr = optimizer.param_groups[-1]['lr']
            # 估计剩余时间，以分钟为单位
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            # 使用日志记录器记录当前训练信息，包括epoch、步骤、损失、主损失、辅助损失、学习率和剩余时间
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, '
                f'logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, '
                f'lr: {current_lr:.8f}, '
                f'epoch_time: {eta_min:.1f}min'
            )
            # 如果传入了wandb，则将相关指标记录到wandb中
            if wandb: wandb.log({"loss"    : current_loss, "logits_loss": current_logits_loss,
                                 "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 当达到保存间隔步数或者当前是最后一步，并且当前进程是主进程时
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 将模型设置为评估模式
            model.eval()
            # 根据模型是否使用混合专家（moe）添加后缀
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 构建模型权重保存的路径
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 获取模型的原始模型（如果模型是分布式数据并行模型，则获取内部的原始模型）
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 获取模型的状态字典
            state_dict = raw_model.state_dict()
            # 将模型状态字典中的参数转换为半精度（FP16）并移动到CPU上，然后保存
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 调用lm_checkpoint函数进行模型检查点保存等相关操作
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            # 将模型重新设置为训练模式
            model.train()
            # 删除状态字典以释放内存
            del state_dict

        # 删除输入数据、标签、模型输出结果和损失，释放内存
        del input_ids, labels, res, loss

    # 如果最后一步大于开始步且最后一步不是累积步数的整数倍
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        # 对优化器进行反缩放
        scaler.unscale_(optimizer)
        # 对模型的参数梯度进行裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # 使用缩放器更新优化器的参数
        scaler.step(optimizer)
        # 更新缩放器的状态
        scaler.update()
        # 清空优化器的梯度
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）"
                                                                     "旋转位置编码的列数，也就是当前训练的最大token数")
    parser.add_argument('--use_moe', default=1, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境，获取当前进程的本地GPU编号（local_rank）
    local_rank = init_distributed_mode()
    # 如果分布式环境已初始化，将设备指定为当前进程对应的GPU
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置全局随机种子，保证实验可复现；分布式下每个进程种子不同，避免数据重复
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 创建模型保存目录，已存在则不报错
    os.makedirs(args.save_dir, exist_ok=True)
    # 初始化模型配置类，设置隐藏层大小、层数、是否使用混合专家模型（MoE）
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe))
    # 如果需要从断点恢复，加载检查点数据；否则为None
    ckp_data = (lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
                if args.from_resume == 1 else None)

    # ========== 3. 设置混合精度训练 ==========
    # 判断设备类型：GPU或CPU
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 设置混合精度数据类型：bfloat16或float16
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 定义自动混合精度上下文，CPU不使用混合精度，GPU启用对应精度
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置可视化训练工具wandb（此处实际使用swanlab） ==========
    wandb = None
    # 仅主进程初始化wandb（swanlab），避免多进程重复初始化
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 导入swanlab并命名为wandb，兼容wandb接口

        # 从断点恢复时，获取之前的wandb运行ID
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # 有ID则恢复运行，无ID则新建
        resume = 'must' if wandb_id else None
        # 定义wandb运行名称，包含训练关键参数
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        # 初始化wandb项目，关联名称、ID、恢复配置
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    # 初始化模型和分词器，分词器词汇量为6400，加载预训练权重（可选），指定运行设备
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 初始化预训练数据集，传入数据路径、分词器、最大序列长度
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 分布式训练时，创建分布式采样器，保证不同进程数据不重复
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 初始化梯度缩放器，仅float16精度时启用，防止梯度下溢
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 初始化AdamW优化器，传入模型参数和学习率
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从检查点恢复训练状态 ==========
    # 定义起始轮次和起始步数，默认从0开始
    start_epoch, start_step = 0, 0
    # 如果有检查点数据，恢复模型、优化器、梯度缩放器参数和训练步数/轮次
    if ckp_data:
        model.load_state_dict(ckp_data['model'])  # 恢复模型权重
        optimizer.load_state_dict(ckp_data['optimizer'])  # 恢复优化器状态（动量、学习率等）
        scaler.load_state_dict(ckp_data['scaler'])  # 恢复梯度缩放器状态
        start_epoch = ckp_data['epoch']  # 恢复中断时的轮次
        start_step = ckp_data.get('step', 0)  # 恢复中断时的步数

    # ========== 7. 模型编译和分布式包装 ==========
    # 如果开启编译，使用torch.compile加速模型推理（PyTorch 2.0+特性）
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')  # 打印日志提示已启用编译
    # 如果分布式环境已初始化，配置分布式数据并行（DDP）
    if dist.is_initialized():
        # 忽略旋转位置编码的频率参数，不进行同步（避免分布式报错）
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 将模型包装为DDP模型，指定当前进程的GPU设备
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练循环 ==========
    # 从恢复的轮次开始，到总轮次结束
    for epoch in range(start_epoch, args.epochs):
        # 分布式训练时，设置采样器轮次，保证每轮数据打乱顺序不同
        train_sampler and train_sampler.set_epoch(epoch)
        # 每轮重置随机种子，打乱数据集索引
        setup_seed(42 + epoch);
        indices = torch.randperm(len(train_ds)).tolist()
        # 如果是恢复的第一轮，跳过中断前的步数；否则从0开始
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 创建可跳过指定步数的批次采样器
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 初始化数据加载器，设置多线程加载、内存 pinned 加速
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 如果需要跳过步数，打印日志并执行训练
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        # 正常训练，无步数跳过
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布式进程 ==========
    # 如果分布式环境已初始化，销毁进程组，释放资源
    if dist.is_initialized(): dist.destroy_process_group()

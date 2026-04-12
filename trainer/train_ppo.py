import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, \
    SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    """
    文本重复惩罚函数，计算n-gram重复率，用于奖励扣分
    Args:
        text: 输入文本
        n: n-gram长度
        cap: 最大惩罚上限
    Return:
        惩罚分数
    """
    # 分词：匹配单词和符号
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    # 构建n-gram
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    # 计算重复度并返回惩罚值
    return min(cap, (len(grams) - len(set(grams)))) * cap * 2 / len(grams) if grams else 0.0


# 自定义的Critic模型，继承自MiniMindLM，用于评估生成状态的价值
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 替换lm_head为输出单一价值的线性层，输出每个token的价值
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model):
    """
    计算最终奖励：规则奖励 + 奖励模型打分
    规则：长度、思考格式、重复度惩罚
    """
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # 解析对话模板，提取角色和内容
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response

            # 规则1：回答长度奖励
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

            # 规则2：思考格式奖励（</think>）
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()

            # 规则3：重复内容惩罚
            rewards[i] -= rep_penalty(answer)

            # 奖励模型打分
            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        # 合并奖励模型分数
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model,
                    start_step=0, wandb=None, use_sglang=False):
    """
    PPO单轮训练核心函数
    epoch: 当前训练轮数
    loader: 数据加载器
    iters: 总迭代步数
    rollout_engine: 生成采样引擎
    ref_model: 参考模型（SFT模型，固定不动）
    actor_scheduler: Actor学习率调度器
    critic_scheduler: Critic学习率调度器
    reward_model: 奖励模型
    """
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0  # 梯度累积步数

    # 遍历每个batch数据
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        # 对prompt进行编码，左填充，适配生成
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                        padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        prompt_length = enc.input_ids.shape[1]

        # ==================== Rollout 阶段：Actor生成回答 ====================
        # 批量生成一批回答
        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        gen_out = rollout_result.output_ids
        responses_text = rollout_result.completions

        # 计算该批生成的奖励
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]

        # Debug模式：打印样本、生成结果、奖励
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-' * 100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_length}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('=' * 100)

        # ==================== Mask 与序列长度处理 ====================
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        seq_len, resp_start = gen_out.size(1) - 1, prompt_length - 1
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= resp_start
        final_mask = (resp_mask & (~labels.eq(tokenizer.pad_token_id))).float()  # [B, P+R-1]
        B = len(prompts)

        # 回答部分的标签与mask处理
        resp_labels = labels[:, resp_start:]  # [B, R]
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        resp_pad_mask = ~resp_labels.eq(tokenizer.pad_token_id)
        resp_lengths = resp_pad_mask.sum(dim=1)
        eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        has_eos = eos_mask.any(dim=1)
        eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        resp_value_mask = resp_policy_mask.clone()

        # ==================== 旧策略/旧价值/参考模型推理（无梯度） ====================
        with torch.no_grad():
            # 获取Critic模型（处理DDP封装）
            critic_for_rollout = critic_model.module if isinstance(critic_model,
                                                                   DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            old_resp_values = values_seq[:, resp_start:-1] * resp_value_mask

            # 获取Actor模型，计算旧策略对数概率
            actor_for_rollout = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            with autocast_ctx:
                logits = actor_for_rollout(input_ids=gen_out, attention_mask=full_mask).logits

            old_resp_logp = (
                F.log_softmax(logits[:, :-1], dim=-1)
                .gather(2, labels.unsqueeze(-1)).squeeze(-1)[:, resp_start:]
            )

            # 参考模型（SFT）对数概率，用于KL惩罚
            ref_logp_all = (
                F.log_softmax(
                    ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1],
                    dim=-1
                )
                .gather(2, labels.unsqueeze(-1))
                .squeeze(-1)
            )
            ref_resp_logp = ref_logp_all[:, resp_start:]

            # 将最终奖励加到最后一个有效token上
            token_rewards = torch.zeros_like(old_resp_logp)
            last_idx = resp_lengths - 1  # [B]
            token_rewards[torch.arange(B, device=args.device), last_idx] += rewards

            # ==================== GAE 优势函数计算 ====================
            gen_len = old_resp_values.size(1)
            lastgaelam = torch.zeros(B, device=args.device)
            advs_rev = []
            for t in reversed(range(gen_len)):
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R]
            returns = advantages + old_resp_values  # [B, R]

            # 优势函数归一化（稳定训练）
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        # ==================== PPO 多轮小批量更新 ====================
        mb_size = max(1, min(args.mini_batch_size, B))
        stop_ppo = False  # KL早停标志
        # 日志统计变量
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        kl_sum = 0.0
        kl_ref_sum = 0.0
        clipfrac_sum = 0.0
        aux_loss_sum = 0.0
        log_count = 0

        # 解开DDP包装
        actor_unwrapped = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
        critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model

        # PPO多次迭代更新同一批数据
        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo:
                break
            # 随机打乱样本
            b_inds = torch.randperm(B, device=args.device)
            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]

                # Critic前向，计算当前价值
                mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq[:, resp_start:-1]

                # Actor前向
                with autocast_ctx:
                    res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)

                # 计算当前策略对数概率
                mb_logp_all = (
                    F.log_softmax(res.logits[:, :-1], dim=-1)
                    .gather(2, labels[inds].unsqueeze(-1))
                    .squeeze(-1)
                )
                mb_resp_logp = mb_logp_all[:, resp_start:]

                # 近似KL散度，用于早停
                log_ratio = mb_resp_logp - old_resp_logp[inds]
                approx_kl = (
                        (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() /
                        resp_policy_mask[inds].sum().clamp(min=1)
                )

                # 多卡同步KL，防止死锁
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)

                # KL超过阈值，提前停止PPO更新
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True

                # 裁剪率统计
                ratio = torch.exp(log_ratio)
                clipfrac = (
                        (((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds])
                        .sum() / resp_policy_mask[inds].sum().clamp(min=1)
                )
                # 参考模型KL惩罚（防止分布偏移过大）
                kl_ref_penalty = (
                        ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) -
                          (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                         * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                )
                # PPO策略损失
                policy_loss = (
                        (torch.max(-advantages[inds] * ratio, -advantages[inds]
                                   * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                         * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                        + args.kl_coef * kl_ref_penalty
                )
                # 价值损失（裁剪版）
                value_loss = (
                        0.5 * (torch.max(
                    (mb_resp_values - returns[inds]) ** 2,
                    (torch.clamp(mb_resp_values, old_resp_values[inds]
                                 - args.cliprange_value, old_resp_values[inds]
                                 + args.cliprange_value) - returns[inds]) ** 2
                ) * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)
                )

                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 早停时loss置0，保持DDP通信闭环
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps

                # 反向传播
                loss.backward()

                # 日志累加
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1

                grad_accum_step += 1

                # 梯度累积达到步数后更新参数
                if grad_accum_step % args.accumulation_steps == 0:
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        # 处理剩余的梯度累积
        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # 主进程更新Rollout引擎的模型权重（SGLang需要）
        if is_main_process() and step % args.save_interval == 0:
            rollout_engine.update_policy(actor_model)

        # ==================== 日志打印与Wandb记录 ====================
        if is_main_process():
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "reward"     : reward_val,
                    "kl_ref"     : kl_ref_val,
                    "approx_kl"  : approx_kl_val,
                    "clipfrac"   : clipfrac_val,
                    "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr"   : actor_lr,
                    "critic_lr"  : critic_lr,
                })

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"Reward: {reward_val:.4f}, "
                f"KL_ref: {kl_ref_val:.4f},"
                f" Approx KL: {approx_kl_val:.4f}, "
                f"ClipFrac: {clipfrac_val:.4f}, "
                f"Critic Loss: {critic_loss_val:.4f}, "
                f"Avg Response Len: {avg_len_val:.2f}, "
                f"Actor LR: {actor_lr:.8f}, "
                f"Critic LR: {critic_lr:.8f}"
            )

        # ==================== 模型保存 ====================
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)

            # 保存完整训练状态（支持断点续训）
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer,
                epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                scheduler=actor_scheduler, critic_model=critic_model,
                critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler
            )
            actor_model.train()
            del actor_state

        # 释放显存
        del enc, gen_out, responses_text, rewards, full_mask, values_seq, advantages
        del logits, labels, final_mask, resp_labels, resp_idx, resp_pad_mask, eos_mask, has_eos, eos_pos, resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_logp_all, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values


if __name__ == "__main__":
    '''
                   ┌─────────────┐
                   │  提示词数据集  │
                   └───────┬─────┘
                           │
                           ▼
            ┌───────────────────────────────┐
            │        Rollout 引擎           │  ← 你用的 torch / sglang
            │  (让 Actor 生成回答)           │
            └──────────────┬────────────────┘
                           │
                           ▼
    ┌───────────────────────────────────────────────────┐
    │               奖励计算 calculate_rewards()          │
    │  = 奖励模型打分 + 长度奖励 + 重复惩罚 + 思考格式奖励     │
    └───────────────────────┬───────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │        GAE 优势计算            │
            │   (算出每一步 token 的优势值)   │
            └───────────────┬───────────────┘
                           │
                           ▼
            ┌───────────────────────────────────────┐
            │             PPO 核心训练              │
            │  ┌──────────┐          ┌──────────┐  │
            │  │  Actor   │◄────────►│  Critic  │  │
            │  │(生成模型) │          │(价值模型) │  │
            │  └──────────┘          └──────────┘  │
            │           多轮小批量更新               │
            └─────────────────────┬─────────────────┘
                                  │
                                  ▼
                        ┌───────────────────┐
                        │    保存模型权重    │
                        └───────────────────┘
    ① Actor 模型（策略网络）
        就是你的 MiniMindForCausalLM
        作用：根据提示词生成回答
        训练目标：让生成的回答获得更高奖励
        会被 PPO 反复更新、反复优化
    ② Critic 模型（价值网络）
        你自定义的 CriticModel，继承自基座模型
        把最后一层输出改成 输出 1 个价值分数
        作用：预测当前生成的 “未来总奖励”
        用来计算 优势函数（GAE），指导 Actor 更新
    ③ Reward Model（奖励模型）
        你加载的 LMForRewardModel
        作用：给模型生成的回答打分
        你的代码还叠加了 规则奖励：
        长度合适 +0.5 / 不合适 -0.5
        有思考格式 ... +1.0
        重复文本惩罚 rep_penalty()
        奖励模型输出的语义分数
    ④ Rollout Engine（采样生成引擎）
        就是你之前问的：
        TorchRolloutEngine：调试用
        SGLangRolloutEngine：训练用（超快）
        作用：让 Actor 批量生成回答，供 PPO 训练
    '''
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="sglang", choices=["torch", "sglang"],
                        help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8997", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
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
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Actor模型
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)  # 最后一层输出改成 输出 1 个价值分数
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = RLAIFDataset(
        args.data_path, tokenizer,
        max_length=(args.max_seq_len + args.max_gen_len),
        thinking_ratio=args.thinking_ratio
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps,
                                         eta_min=args.critic_learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(actor_model)
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    if is_main_process(): rollout_engine.update_policy(actor_model)

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
            ppo_train_epoch(
                epoch, loader, len(loader) + skip, rollout_engine, ref_model, actor_scheduler,
                critic_scheduler, reward_model, start_step, wandb,
                use_sglang=(args.rollout_engine == "sglang")
            )
        else:
            ppo_train_epoch(
                epoch, loader, len(loader), rollout_engine, ref_model, actor_scheduler, critic_scheduler,
                reward_model, 0, wandb, use_sglang=(args.rollout_engine == "sglang")
            )

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()

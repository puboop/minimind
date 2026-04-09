"""
训练工具函数集合
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel
from model.model_minimind import MiniMindForCausalLM


def get_model_params(model, config):
    """
    计算并记录模型的参数数量相关信息
    :param model: 模型对象，从中获取参数数量
    :param config: 配置对象，从中获取与专家相关的配置参数
    :return: 无返回值，主要是记录日志信息
    """
    # 计算模型所有参数的总数，除以1e6是为了将单位转换为百万
    total = sum(p.numel() for p in model.parameters()) / 1e6
    # 从配置对象中获取名为'n_routed_experts'的属性值，如果不存在则获取'num_experts'的属性值，默认值为0
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    # 从配置对象中获取名为'num_experts_per_tok'的属性值，默认值为0
    n_active = getattr(config, 'num_experts_per_tok', 0)
    # 从配置对象中获取名为'n_shared_experts'的属性值，默认值为0
    n_shared = getattr(config, 'n_shared_experts', 0)
    # 计算模型中名字包含'mlp.experts.0.'的参数数量，除以1e6转换为百万
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    # 计算模型中名字包含'mlp.shared_experts.0.'的参数数量，除以1e6转换为百万
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    # 计算基础参数数量，用总参数数减去路由专家参数数量和共享专家参数数量
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    # 计算活跃参数数量，基础参数加上活跃专家参数数量和共享专家参数数量
    active = base + (expert * n_active) + (shared_expert * n_shared)
    # 如果活跃参数数量小于总参数数量，记录总参数和活跃参数数量信息
    if active < total:
        Logger(f'Model Params: {total:.2f}M - A{active:.2f}M')
    # 否则，只记录总参数数量信息
    else:
        Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    根据当前步骤、总步骤数和初始学习率，计算当前的学习率
    :param current_step: 当前所处的步骤数
    :param total_steps: 总的步骤数
    :param lr: 初始学习率
    :return: 返回根据公式计算得到的当前学习率
    """
    # 利用给定的公式计算学习率，公式中使用了余弦退火策略相关的计算方式，
    # 0.1和0.45是调整系数，通过余弦函数结合当前步骤和总步骤的比例来动态调整学习率
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    # 获取环境变量RANK的值，如果未设置则默认为-1
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 如果RANK为-1，说明不是分布式数据并行（DDP）模式，直接返回0

    # 初始化分布式进程组，使用nccl后端，nccl是英伟达提供的用于GPU之间通信的库
    dist.init_process_group(backend="nccl")
    # 获取本地进程的rank，也就是在当前节点内的进程编号
    local_rank = int(os.environ["LOCAL_RANK"])
    # 设置当前进程使用的GPU设备为local_rank对应的设备
    torch.cuda.set_device(local_rank)
    return local_rank  # 返回本地进程的rank


def setup_seed(seed: int):
    """
    设置各种随机数生成器的种子，以确保实验的可重复性
    :param seed: 种子值，是一个整数
    :return: 无返回值
    """
    # 设置Python内置的random模块的种子，这样后续使用random模块生成的随机数序列将是固定的
    random.seed(seed)
    # 设置numpy库的随机数种子，使得numpy中生成随机数的操作具有可重复性
    np.random.seed(seed)
    # 设置PyTorch CPU随机数种子，保证在CPU上的张量操作生成的随机数可重复
    torch.manual_seed(seed)
    # 设置当前GPU的随机数种子，确保在当前GPU上的操作生成的随机数可重复
    torch.cuda.manual_seed(seed)
    # 设置所有GPU的随机数种子，确保在所有GPU上的操作生成的随机数可重复
    torch.cuda.manual_seed_all(seed)
    # 将cuDNN设置为确定性模式，即每次计算的结果是固定的
    torch.backends.cudnn.deterministic = True
    # 关闭cuDNN的自动寻找最佳算法模式，以确保计算的可重复性，
    # 开启这个模式虽然可能会提升性能，但会导致每次计算结果不同
    torch.backends.cudnn.benchmark = False


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None,
                  epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    这个函数用于处理模型的检查点（保存和加载）
    :param lm_config: lm_config应该是一个包含模型相关配置信息的对象
    :param weight: weight是检查点的权重名称，默认是'full_sft'
    :param model: model是要保存的模型对象，如果为None则进入加载模式
    :param optimizer: optimizer是优化器对象，用于保存优化器的状态
    :param epoch: epoch表示当前是第几个epoch
    :param step: step表示当前步骤数
    :param wandb: wandb是一个用于实验跟踪和可视化的工具，这里如果传入就可以记录相关指标
    :param save_dir: save_dir是保存检查点的目录路径，默认是'../checkpoints'
    :param kwargs: 其他可能需要保存的对象或数据
    :return: 如果是加载模式且检查点存在，返回加载的检查点数据，否则返回None
    """
    # 创建保存目录，如果目录已经存在则不会报错
    os.makedirs(save_dir, exist_ok=True)
    # 根据是否使用moe（混合专家模型）来确定路径后缀
    moe_path = '_moe' if lm_config.use_moe else ''
    # 完整检查点文件路径
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    # 可恢复检查点文件路径
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:  # 保存模式
        # 如果模型是DistributedDataParallel类型，获取其内部的原始模型
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        # 获取原始模型（处理可能的包装）
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        # 获取模型的状态字典
        state_dict = raw_model.state_dict()
        # 将状态字典中的参数转换为半精度（half）并移动到CPU上
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        # 临时保存路径
        ckp_tmp = ckp_path + '.tmp'
        # 保存临时检查点
        torch.save(state_dict, ckp_tmp)
        # 将临时检查点移动到正式路径
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            # 尝试获取wandb的运行ID
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 可恢复数据字典，包含模型、优化器等状态
        resume_data = {
            'model'    : state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch'    : epoch,
            'step'     : step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id' : wandb_id
        }
        # 遍历kwargs中的其他数据并添加到可恢复数据字典中
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    # 如果是DistributedDataParallel类型，获取其内部的原始对象
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    # 获取原始对象（处理可能的包装）
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        # 临时可恢复检查点路径
        resume_tmp = resume_path + '.tmp'
        # 保存临时可恢复检查点
        torch.save(resume_data, resume_tmp)
        # 将临时可恢复检查点移动到正式路径
        os.replace(resume_tmp, resume_path)
        # 删除不再需要的状态字典和可恢复数据字典
        del state_dict, resume_data
        # 清空CUDA缓存
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            # 加载可恢复检查点数据
            ckp_data = torch.load(resume_path, map_location='cpu')
            # 获取保存时的GPU数量
            saved_ws = ckp_data.get('world_size', 1)
            # 获取当前的GPU数量
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                # 如果GPU数量变化，调整step
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    # 模型类型标识，Hugging Face 框架使用
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)  # 初始化父类 PretrainedConfig

        # ==================== 基础模型结构 ====================
        self.hidden_size = hidden_size  # 模型隐藏层维度（核心维度）
        self.num_hidden_layers = num_hidden_layers  # Transformer Block 总层数
        self.use_moe = use_moe  # 是否使用 MOE 混合专家模式

        # ==================== 正则化与训练参数 ====================
        self.dropout = kwargs.get("dropout", 0.0)  # Dropout 比例，防止过拟合

        # ==================== 词表与特殊 Token ====================
        self.vocab_size = kwargs.get("vocab_size", 6400)  # 词表大小
        self.bos_token_id = kwargs.get("bos_token_id", 1)  # 句子起始 token ID
        self.eos_token_id = kwargs.get("eos_token_id", 2)  # 句子结束 token ID

        # ==================== 注意力机制配置 ====================
        self.flash_attn = kwargs.get("flash_attn", True)  # 是否启用 FlashAttention 加速
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)  # 注意力头数（Q 头）
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)  # KV 头数（GQA 分组注意力）
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)  # 每个头的维度

        # ==================== 前馈网络 & 激活函数 ====================
        self.hidden_act = kwargs.get("hidden_act", 'silu')  # 激活函数（默认 SiLU/Swish）
        # 中间层维度（自动计算为 64 的整数倍，提升计算效率）
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)

        # ==================== 位置编码 & RoPE ====================
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # 最大支持序列长度 用于生成旋转位置编码的行数
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)  # RMSNorm 极小值
        self.rope_theta = kwargs.get("rope_theta", 1e6)  # RoPE 基础频率

        # YaRN 长度外推（超长上下文扩展，推理时启用）
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast"                       : 32,
            "beta_slow"                       : 1,
            "factor"                          : 16,  # 长度扩展倍数
            "original_max_position_embeddings": 2048,  # 训练时原始长度
            "attention_factor"                : 1.0,
            "type"                            : "yarn"
        } if self.inference_rope_scaling else None

        # ==================== MOE 混合专家专用配置（use_moe=False 时无效） ====================
        self.num_experts = kwargs.get("num_experts", 4)  # 总专家数
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)  # 每个 token 选择几个专家
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)  # 专家中间维度
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)  # 是否归一化 Top-K 权重
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 路由辅助损失系数 平均损失函数


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    """
    RMSNorm 归一化（LLaMA 等大模型标配，比 LayerNorm 更简洁、高效）
    核心：只做缩放归一化，不做均值减法 + 可学习缩放参数
    优势：计算更快、稳定性更好、参数量更小
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # 极小值，防止分母为 0
        self.eps = eps
        # 可学习的缩放参数（初始化为全 1），只对维度进行加权，无偏置
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        # 1. 对输入 x 逐元素平方
        # 2. 在最后一个维度（特征维度）求均值
        # 3. 加 eps 防止除 0，取倒数平方根
        # 4. 与原输入相乘 → 完成 RMS 归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 前向传播：
        # 1. 先做 RMS 归一化（转 float 防止精度丢失）
        # 2. 乘以可学习权重 self.weight
        # 3. 转回原输入精度（保证混合精度训练正常）
        return (self.weight * self.norm(x.float())).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """
    预计算 RoPE 旋转位置编码的 cos / sin 表
    作用：提前算好所有位置的旋转角度，推理时直接查表，速度极快
    支持：原生 RoPE + YaRN 长度外推（让模型支持远超训练长度的上下文）
    dim: 每个注意力头的维度 head_dim 96
        比如 LLaMA-7B：hidden_size=4096，num_heads=32 → dim=4096/32=128
        必须是偶数
            RoPE 的核心是两两维度配对旋转（(0,1), (2,3), ..., (dim-2, dim-1)）
            如果 dim 是奇数，最后一个维度无法配对，代码会报错或舍弃最后一维
            所以大模型的 hidden_size 和 num_heads 设计时，都会保证 dim 是偶数
        维度大小影响位置感知能力
            dim 越大 → 每个头的表征空间越充足 → 能捕捉更精细的位置关系
            dim 越小 → 计算越快，但位置编码的区分度会下降
            行业默认：dim 通常在 64~256 之间（比如 64、128、256），兼顾效果和效率
        和代码的直接关联
            代码中 torch.arange(0, dim, 2) 会生成步长为 2 的维度索引，正好配对
            最终 freqs_cos/sin 的最后一维长度等于 dim，和输入向量维度完全对齐
    end: 最大序列长度（预计算到多长） 32768
        必须 ≥ 模型实际推理的最大序列长度
            推理时会用 start_pos:start_pos+seq_length 切片，如果 start_pos+seq_length > end，会直接数组越界报错
            比如预计算 end=2048，就无法处理 3000 长度的序列
        不是越大越好：平衡内存占用
            end 越大 → 预计算的 freqs_cos/sin 张量越大 → 占用更多内存
            比如 dim=128，end=8192 → 单个张量大小 = 8192×128×4Byte（float32）≈ 4MB
            end=32768 → 张量大小 = 32768×128×4Byte ≈ 16MB
            建议：end 设为 模型实际需要的最大序列长度的 1.2~1.5 倍 即可，不用盲目拉满
        和长度外推的关系
            原生 RoPE 本身有一定长度外推能力，但如果 end 远小于实际序列长度，外推效果会下降
            搭配 YaRN/NTK 等长度外推方法时，end 可以设为外推后的目标长度（比如训练时 end=2048，外推到 8192 就设 end=8192）
        训练和推理的一致性
            训练时 end 设为训练序列长度上限（比如 2048）
            推理时如果要加长上下文，end 要同步增大，且最好搭配长度外推策略，否则效果会断崖式下跌
    rope_base: RoPE 基础频率
    rope_scaling: YaRN 长度外推（超长上下文扩展）
    """
    # 1. 计算基础频率：theta = base^(-2i/dim)
    # 生成频率序列，控制不同维度的旋转速度
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0  # 注意力缩放因子（YaRN用）

    # 2. 如果启用 YaRN 长度扩展（超长上下文）
    if rope_scaling is not None:
        # 外推的最大长度2048，从rope_scaling字典获取原始最大位置嵌入长度，若不存在则设为2048
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        # 从rope_scaling字典获取缩放因子，若不存在则设为16
        factor = rope_scaling.get("factor", 16)
        # 计算高频截断位置对应的维度索引
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        # 计算低频截断位置对应的维度索引
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        # 注意力缩放因子
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        # 仅当当前最大长度（end）与原始训练长度（orig_max）的比值大于1.0时，才进行缩放操作
        # 意味着当前处理的序列长度超过了原始训练设定的长度，才需要进行YaRN相关调整
        if end / orig_max > 1.0:
            # inv_dim函数用于计算截断位置对应的维度索引
            # 它基于给定的参数b（beta_fast或beta_slow）、嵌入维度dim和rope_base计算维度索引
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # 计算低频截断位置，取计算结果与0中的较大值
            # 低频截断位置用于确定哪些维度的频率缩放较小（几乎不缩放），以保持局部信息
            low = max(math.floor(inv_dim(beta_fast)), 0)
            # 计算高频截断位置，取计算结果与dim // 2 - 1中的较小值
            # 高频截断位置用于确定哪些维度的频率缩放较大，以处理长距离依赖信息
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

            # 生成线性斜坡（ramp）：低维不缩放，高维缩放
            # torch.arange(dim // 2, device=freqs.device).float()生成从0到dim // 2 - 1的一维张量
            # 这个张量减去low后，再除以(high - low)，通过torch.clamp将结果限制在0到1之间
            # 生成的ramp张量用于在不同维度上平滑地过渡缩放因子
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0, 1
            )

            # YaRN核心公式：频率缩放，既延长上下文又保持精度
            # 通过(1 - ramp + ramp / factor)得到一个缩放因子张量
            # 这个缩放因子张量会根据ramp值，在低维部分接近1（不缩放），在高维部分接近1/factor（缩放）
            # 将freqs张量的每个元素乘以这个缩放因子张量，实现对频率的缩放调整
            freqs = freqs * (1 - ramp + ramp / factor)
    # 3. 生成位置序列 t = [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)

    # 4. 外积：得到 [end, dim//2] 的位置×频率矩阵  [32768,48]
    freqs = torch.outer(t, freqs).float()

    # 5. 拼接成完整维度（前后两半相同），生成 cos 和 sin 表
    # freqs中的每个值都是一个输入角度值 [32768, 96]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    # 返回预计算好的 cos / sin 张量，供模型直接使用 [32768, 96]
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    对 Q 和 K 应用 RoPE 旋转位置编码
    核心：通过旋转矩阵给 Query 和 Key 注入位置信息，让注意力计算感知序列顺序
    原理：对每个向量的二维子空间进行旋转，不同位置对应不同旋转角度
    :param q: 多头注意力值Q [32, 340, 8, 96]
    :param k: 多头注意力值K [32, 340, 4, 96]
    :param cos: 余弦值
    :param sin: 正弦值
    :param unsqueeze_dim: 增加的维度位置
    """

    # 定义旋转函数：将向量从中间切分，后半部分取反，再拼接
    # 例: [a, b, c, d] → [-c, -d, a, b]，用于实现旋转计算
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2:],  # 后半部分取负值
             x[..., : x.shape[-1] // 2]),  # 前半部分不变
            dim=-1
        )

    # RoPE 核心公式：Q_rot = Q * cos + rotate_half(Q) * sin
    # unsqueeze 扩展维度，适配多头形状 [seq_len, head_dim] → [seq_len, 1, head_dim]
    # 整体乘以余弦值+调转前半部分与负后半部分 [32, 340, 8, 96]
    q_embed = (
        # q [32,340,8,96] cos.unsqueeze [340,1,96]
            (q * cos.unsqueeze(unsqueeze_dim)) +
            (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    ).to(q.dtype)  # 保持精度与原 q 一致

    # 对 Key 执行完全相同的旋转操作 [32, 340, 4, 96]
    k_embed = (
            (k * cos.unsqueeze(unsqueeze_dim)) +
            (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    ).to(k.dtype)  # 保持精度与原 k 一致

    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA 分组查询注意力专用：将 K/V 头重复 n_rep 次，匹配 Q 头数量
    作用：让 KV 头数 = Q 头数，保证注意力矩阵计算维度匹配
    例如：Q=32头, KV=8头 → n_rep=4 → KV重复4次 → 32头

    参数：
        x: 输入的 K 或 V 张量 shape: [bs, seq_len, num_kv_heads, head_dim] [32, 340, 4, 96]
        n_rep: 每个 KV 头需要重复的次数
    返回：
        重复后的张量 shape: [bs, seq_len, num_kv_heads * n_rep, head_dim]
    """
    # 拆解输入维度：batch大小、序列长度、KV头数量、每个头的维度
    bs, slen, num_key_value_heads, head_dim = x.shape

    # 如果不需要重复，直接返回
    if n_rep == 1:
        return x

    # 核心步骤： [32, 340, 8, 96]
    # 1. 增加一个维度  [bs, slen, num_kv_heads, 1, head_dim]
    # 2. expand 扩展到 n_rep 次  [bs, slen, num_kv_heads, n_rep, head_dim]
    # 3. reshape 把 KV头 和 重复维度合并 → 得到和 Q 一样的头数
    return (
        x[:, :, :, None, :]  # 增加维度
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)  # 扩展重复
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)  # 维度融合
    )


class Attention(nn.Module):
    """
    多头注意力机制（类LLaMA/GPT，支持GQA、RoPE、KV Cache、FlashAttention）
    核心功能：计算序列中每个token与其他token的关联权重，提取上下文信息
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()

        # ========== 1. 注意力头配置（支持 GQA 分组查询注意力）==========
        # KV 头数量（不指定则等于 Q 头数，即普通 MHA）
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        # K 投影：hidden_size → KV头总数 × 头维度
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # V 投影：hidden_size → KV头总数 × 头维度
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出投影：注意力输出拼接 → 回到 hidden_size
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # ========== 3. 归一化（增强训练稳定性）==========
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # ========== 4. Dropout 防止过拟合 ==========
        self.attn_dropout = nn.Dropout(config.dropout)  # 注意力权重 dropout
        self.resid_dropout = nn.Dropout(config.dropout)  # 输出残差 dropout
        self.dropout = config.dropout

        # ========== 5. 是否启用 FlashAttention（加速）==========
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape  # 批次大小、序列长度、隐藏维度

        # ========== 1. QKV 线性投影 ==========
        xq = self.q_proj(x)  # (bsz, seq_len, num_heads * head_dim)
        xk = self.k_proj(x)  # (bsz, seq_len, num_kv_heads * head_dim)
        xv = self.v_proj(x)  # (bsz, seq_len, num_kv_heads * head_dim)

        # ========== 2. 维度重塑：拆分成多头 ==========
        # 形状从：[32,340,768]->[32,340,8,96]
        # 含义为：将多批次的数据转为给多个注意力头进行计算
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # Q: (bsz, seq_len, n_heads, head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # K: (bsz, seq_len, n_kv_heads, head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # V: (bsz, seq_len, n_kv_heads, head_dim)

        # ========== 3. 对 Q、K 分别做归一化 ==========
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # ========== 4. 应用 RoPE 旋转位置编码 ==========
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # ========== 5. KV Cache 增量推理（生成式模型必备）==========
        if past_key_value is not None:
            # 把历史 KV 和当前 KV 拼接，实现流式生成
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # 如果需要缓存，保存当前 KV
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        if (self.flash and (seq_len > 1) and
                (not self.is_causal or past_key_value is None) and
                (attention_mask is None or torch.all(attention_mask == 1))):
            output = F.scaled_dot_product_attention(xq, xk, xv,
                                                    dropout_p=self.dropout if self.training else 0.0,
                                                    is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len),
                    float("-inf"),
                    device=scores.device
                ).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    Transformer前馈网络 (FFN) - LLaMA风格 (SwiGLU)
    结构：升维 → 双路门控激活 → 降维
    作用：对注意力输出的特征进行非线性变换，增强模型表达能力
    """

    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()

        # 中间层维度（通常是 hidden_size 的 4 倍左右）
        intermediate_size = intermediate_size or config.intermediate_size

        # 门控投影：hidden_size → intermediate_size（生成门控信号）
        # 门控机制 = 让神经网络自己学会 “开 / 关 / 调节” 信息流动的开关。
        # 哪些信息重要 → 让它通过
        # 哪些信息没用 → 把它挡住
        # 哪些信息需要加强 → 给它放大
        # 哪些信息需要减弱 → 给它压低
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)

        # 上采样投影：hidden_size → intermediate_size（特征升维）
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)

        # 下采样投影：intermediate_size → hidden_size（特征降维，回到原维度）
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

        # 激活函数（如 SiLU / Swish）
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        输入 → 分成两路
           路1：Linear(gate) → SiLU/Swish 激活
           路2：Linear(up)   → 不激活
        两路相乘 → Linear(down) 降维 → 输出
        :param x:
        :return:
        """
        # 前向传播逻辑（SwiGLU 核心公式）：
        # 1. gate_proj(x) → 门控分支
        # 2. act_fn(门控分支) → 激活过滤
        # 3. up_proj(x) → 特征分支
        # 4. 两个分支逐元素相乘（门控机制）
        # 5. down_proj → 降维输出
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 1. 路由器（门控）：把词特征 → 分给 N 个专家的分数
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # 2. 专家列表：创建 N 个独立的 FeedForward 专家
        self.experts = nn.ModuleList(
            [FeedForward(config, intermediate_size=config.moe_intermediate_size)
             for _ in range(config.num_experts)]
        )

        # 激活函数（这里主要给专家用，路由已经用softmax）
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # x 形状：批量大小 x 句子长度 x 特征维度
        batch_size, seq_len, hidden_dim = x.shape
        # 展平：将所有批次的token整合为一个批次，方便逐个分配专家  矩阵【总词数 × 特征维度】
        x_flat = x.view(-1, hidden_dim)
        # 3. 给每个词计算【专家信任分数】，并归一化成概率   每个词对所有专家的打分，分数越高越适合这个专家
        # 整合为：【总词数 × 专家数】 矩阵
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # 4. 挑选分数最高的 K 个专家（比如每个词选 2 个专家）
        """
        假设：总词数 = 3 个词专家数 = 5 个专家每个词选 2 个专家（k=2）
            词1：[0.1, 0.5, 0.05, 0.3, 0.05]  # 词1对5个专家的打分
            词2：[0.4, 0.1, 0.05, 0.05, 0.4]
            词3：[0.05, 0.05, 0.6, 0.2, 0.1]
        运行 topk(scores, k=2, dim=-1) 结果：
            词 1 选：专家 2 (0.5)、专家 4 (0.3)
            词 2 选：专家 1 (0.4)、专家 5 (0.4)
            词 3 选：专家 3 (0.6)、专家 4 (0.2)
        """
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        # 可选：把选中专家的概率重新归一化，让总和=1
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        # 5. 初始化输出空张量
        y = torch.zeros_like(x_flat)
        # 6. 遍历所有专家，只处理【分配给自己的词】
        for i, expert in enumerate(self.experts):
            # 查看哪些词 分配 给了第 i 个专家 形状：[总词数N, K]
            mask = (topk_idx == i)
            if mask.any():  # 判断有1的情况
                # 拿到分配给该专家的所有词的下标 变为一个1维的向量
                token_idx = mask.any(dim=-1).nonzero().flatten()  # 展平为1维，并将其中的0去除
                # 取出这些 token 对应这个专家的权重，并扩维方便相乘
                # 形状变成 [有效token数，1]
                weight = topk_weight[mask].view(-1, 1)
                # 专家处理 → 加权 → 加到输出里
                y.index_add_(
                    0,  # 沿第 0 维（token 维度）累加
                    token_idx,  # 要把值加到哪几个 token 位置
                    # x_flat[token_idx]: 取出对应token的原始权重 形状为：[token_idx, 768]
                    # expert(x_flat[token_idx]): 将这些数放在前馈神经网络中计算
                    # expert(x_flat[token_idx]) * weight): 将其与专家的打分权重进行相乘计算
                    (expert(x_flat[token_idx]) * weight).to(y.dtype)  # 专家计算好的加权结果
                )
            elif self.training:
                # 遍历专家所有参数，强行让它们进入计算图
                # 虽然乘 0 不影响数值，但梯度链路保留了
                # 即使专家没分到任何 token，梯度依然会正常计算、正常更新
                # 防止专家 “饿死、不更新、废掉”
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        # 7. 训练时：辅助损失（防止某些专家永远不干活）
        if self.training and self.config.router_aux_loss_coef > 0:
            # 统计每个专家被选中的频率
            # F.one_hot(索引张量, 总类别数) # 把每个数字索引 → 变成只有对应位置是 1，其他都是 0 的向量
            # 形状为：[topk_idx行数，topk_idx列数，self.config.num_experts] = 【总词数, 每个词选K个专家, 总专家数】
            # .mean(0)沿着第0维求平均值
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # 辅助损失：让所有专家被使用概率尽量平均
            # scores.mean(0) [E]：所有 token 对每个专家的平均打分
            # load * scores.mean(0) [E]：把「被选频率」和「门控分数」相乘
            # .sum()：得到不平衡程度：专家忙闲差距越大，loss 越大
            # × 专家数 × 系数：缩放 loss 大小，方便训练
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        # 把形状恢复成原来的 批量x句子x维度
        return y.view(batch_size, seq_len, hidden_dim)


class MiniMindBlock(nn.Module):
    """
    Transformer 基础块（类 LLaMA 结构）
    结构：Pre-Norm -> Self-Attention -> 残差 -> Pre-Norm -> MLP -> 残差
    所有 LLM 都是 N 个这样的 block 堆叠而成
    """

    def __init__(self, layer_id: int, config: MiniMindConfig):
        # 初始化 nn.Module 父类，获得参数管理、设备迁移等能力
        super().__init__()

        # 核心：多头自注意力层（负责捕捉词与词之间的关系）
        self.self_attn = Attention(config)

        # Attention 之前的归一化（Pre-Norm 结构，LLaMA 标配）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP 之前的归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        """
        FeedForward（普通前馈网络）：所有token都走同一个网络
            输入 → 升维（gate+up）→ 激活 → 降维 → 输出
        MOEFeedForward（混合专家）：每个token只走 K 个小网络（专家）
            输入 → 路由门（选专家）→ 只进入选中的专家 → 加权输出
        """
        # 前馈网络 / 混合专家网络
        # 根据配置选择：普通 FeedForward 或 MOE 混合专家 MOEFeedForward
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # ==================== 1. 自注意力子模块 ====================
        # 保存残差连接（Residual）
        residual = hidden_states

        # 先归一化（Pre-Norm）→ 再进入注意力层
        hidden_states = self.input_layernorm(hidden_states)

        # 进入自注意力，输出：新特征 + KV 缓存（用于生成加速）
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            position_embeddings,  # RoPE 旋转位置编码
            past_key_value,  # 历史 KV 缓存（增量生成）
            use_cache,  # 是否开启 KV Cache
            attention_mask  # 注意力掩码
        )

        # 注意力输出 + 残差
        hidden_states += residual

        # ==================== 2. MLP / MOE 前馈子模块 ====================
        # 保存残差
        residual = hidden_states

        # 先归一化 → 再进入 MLP
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP 输出 + 残差
        hidden_states = hidden_states + self.mlp(hidden_states)

        # 返回最终特征 + KV 缓存
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    nn.Module: PyTorch 模型的祖宗类；所有神经网络层、模型、模块都必须继承它；它提供了训练 / 推理、参数管理、设备移动、保存加载的全套能力
        1.自动管理可训练参数（weight/bias）
        2..to(device) 一键把模型搬到 GPU/CPU
        3..train() / .eval() 切换训练 / 推理模式
        4..state_dict() / .load_state_dict() 保存 / 加载权重
        5.支持子模块嵌套（Layer 里套 Layer）
    """

    def __init__(self, config: MiniMindConfig):
        # 初始化父类 nn.Module，必须调用，激活 PyTorch 模型核心能力
        super().__init__()
        # 保存模型配置文件（包含词表大小、隐藏层维度、层数、dropout 等所有超参数）
        self.config = config
        # 从配置中提取关键超参数：词表大小、Transformer 隐藏层数量
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        # 词嵌入层：将输入的 token ID 转换为对应维度的向量表示 [vocab_size, hidden_size]
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # Dropout 层：防止模型过拟合，随机失活一部分神经元
        self.dropout = nn.Dropout(config.dropout)

        # 构建多层 Transformer Block，使用 nn.ModuleList 保证参数被正确管理
        # nn.ModuleList 只是一个装层的盒子（不执行前向传播，只帮你管理参数）
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])

        # 最终层归一化：LLaMA 系列使用 RMSNorm，稳定训练、加速收敛
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 旋转位置编码（LLM 必备，为序列添加位置信息）
        # 模型初始化时提前用公式算好的余弦 / 正弦位置编码，固定不变，不是训练出来的
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        # 注册为模型缓冲区：随模型迁移设备，但不参与梯度更新
        """self.register_buffer
        把一个张量绑定到模型上（可以用 self.xxx 调用）
        区别于模型参数：不会被优化器更新、不会参与反向传播
        会自动跟随模型移动设备（CPU → GPU / GPU → CPU 同步）
        是存放固定常量、预计算值的官方最佳方案
        
        persistent=False：不保存到模型权重文件（.bin/.pth）
        persistent=True（默认）：会跟着权重一起保存
        因为 freqs_cos /freqs_sin 可以随时用公式重新计算，不需要存在权重里：
        不存 → 模型文件更小
        不存 → 加载速度更快
        不存 → 不同长度序列都能适配，更灵活
        如果写成 True，权重文件会变大，且推理时不好动态修改序列长度，大模型一律用 False。
        """
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        # 获取输入批次大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 兼容处理：如果传入的 KV 缓存格式不标准，重置为 None，从头计算
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        # 初始化 KV 缓存：没有历史缓存时，创建与层数相同的 None 列表
        past_key_values = past_key_values or [None] * len(self.layers)
        # 计算历史序列起始位置：有缓存则从缓存长度开始，无缓存则从 0 开始（用于 KV Cache 增量推理）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 词嵌入 + Dropout：输入 token → 向量 → 防止过拟合
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 截取当前序列对应的 RoPE 位置编码
        position_embeddings = (
            # seq_length为当前输入token的个数 [340,96]
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []  # 保存每一层新生成的 KV 缓存，用于后续推理加速
        # 逐层通过 Transformer Block 进行计算
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,  # 当前层输入特征
                position_embeddings,  # 位置编码
                past_key_value=past_key_value,  # 历史 KV 缓存（增量生成用）
                use_cache=use_cache,  # 是否开启 KV Cache 加速
                attention_mask=attention_mask  # 注意力掩码（防止看到 padding/未来 token）
            )
            presents.append(present)  # 收集新的 KV 缓存

        # 最后一次归一化，稳定输出特征
        hidden_states = self.norm(hidden_states)

        # 计算 MOE（混合专家）模块的辅助损失：普通模型无此项，仅用于平衡专家负载
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()
        )

        # 返回：模型最终输出特征、最新 KV 缓存、MOE 辅助损失
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    PreTrainedModel: 预训练模型的抽象基类，是 Transformers 库中所有模型的父类，提供了模型加载、保存、配置、初始化的统一能力。
        1.统一加载预训练权重
        2.统一保存/导出模型
        3.管理模型配置
        4.统一前向传播接口
    GenerationMixin: 生成式模型的混合类（Mixin），专门为文本生成任务提供解码生成能力，本身不独立使用，而是作为功能扩展混入模型中。
        1.主流生成策略：贪心搜索、束搜索、Top-K、Top-P、温度采样
        2.流式生成/批量生成
        3.生成参数控制：最大长度、停止词、重复惩罚
        4.标准生成接口：generate() 方法
    """
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0,
                labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache,
                                                              **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values,
                                         hidden_states=hidden_states)

    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50,
                 eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True,
                 repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache,
                                   **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)],
                                       -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(
                logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1),
                                                                  next_token.new_full((next_token.shape[0], 1),
                                                                                      eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids

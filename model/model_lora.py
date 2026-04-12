import torch
from torch import nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        # 寻找每个【多头自注意力层】中q_proj层、o_proj层
        """
        在进行LoRA垂直领域训练时，替换多头注意力块的查询者 `q` 与输出结果 `o` 层主要基于以下原因：
        ### 1. 高效参数微调
         - **降低参数量**：多头注意力机制在大型语言模型中计算量和参数量都很大。通过在 `q` 和 `o` 层应用LoRA，仅引入少量低秩矩阵（`A` 和 `B`）来微调模型。
            在垂直领域训练中，数据特点可能与预训练时有所不同，这种低秩调整方式能够以较小的参数量适应新数据分布，避免对整个模型进行大规模参数更新，大大减少了训练成本和计算资源需求。
         - **快速收敛**：相比全模型微调，LoRA在 `q` 和 `o` 层的微调方式使得模型能够更快地在垂直领域数据上收敛。
            因为只关注与输入查询和输出相关的关键层，训练过程更具针对性，能更快捕捉到垂直领域数据的特征。
        
        ### 2. 注意力机制的关键作用
         - **查询（`q`）的重要性**：查询 `q` 在多头注意力机制中负责定义要关注的内容。在垂直领域，数据可能具有独特的语义和特征，
            调整 `q` 层的LoRA可以让模型在注意力计算时更关注与该领域相关的信息。例如，在医疗领域的文本处理中，通过LoRA调整 `q` 层，
            模型能更聚焦于医学术语、症状描述等关键信息，而忽略无关的通用信息。
         - **输出（`o`）的作用**：输出 `o` 层将注意力计算后的结果进行整合和转换。在垂直领域，需要模型输出与该领域适配的特征表示。
            通过LoRA对 `o` 层进行调整，能够使模型生成更符合垂直领域需求的输出，比如在金融领域生成更准确的风险评估特征表示。
        
        ### 3. 保持模型通用性与领域特异性的平衡
         - **通用性保持**：对 `q` 和 `o` 层进行LoRA微调，不会破坏模型在预训练阶段学到的通用知识。因为大部分模型参数保持不变，
            仅在特定的关键层引入领域相关的微调，模型在处理其他领域任务时仍能利用预训练的通用性。
         - **特异性增强**：针对垂直领域数据，`q` 和 `o` 层的LoRA微调能够有效增强模型对该领域数据的理解和处理能力。
            这种平衡使得模型既可以利用预训练的优势，又能在垂直领域表现出色。 
        """
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            # 替换多头注意块的每个多头注意力层的前向传播函数
            module.forward = forward_with_lora


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)

import json
import os
import random

from datasets import Features, Value

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content: str, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


# 导入必要的库 (代码隐含依赖)
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    大模型预训练专用数据集类
    功能：加载JSON格式文本数据，完成文本分词、序列截断/填充、标签构建
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        """
        数据集初始化函数
        Args:
            data_path: 训练数据的JSON文件路径
            tokenizer: 文本分词器 (如BertTokenizer/GPT2Tokenizer)
            max_length: 模型输入的最大序列长度，默认512
        """
        # 调用父类Dataset的初始化方法
        super().__init__()
        # 保存分词器到实例变量，供后续分词使用
        self.tokenizer = tokenizer
        # 保存最大序列长度
        self.max_length = max_length
        # 加载JSON格式数据集，split='train'表示加载为训练集
        # ====================== 核心：load_dataset 加载 .jsonl 数据 ======================
        # 1. 第一个参数 'json'：告诉 Hugging Face 库要加载【JSON格式】数据
        # 2. data_files=data_path：指定要加载的文件路径（你这里是 pretrain_t2t.jsonl 这类 .jsonl 文件）
        # 3. split='train'：将加载的所有数据统一划分为【训练集】
        # 4. 自动识别 .jsonl：jsonl = 每一行是一个独立JSON对象，load_dataset 会自动按行解析
        # 5. 加载结果：返回一个 Dataset 对象，可直接通过索引/长度遍历使用
        self.samples = load_dataset('json', data_files=data_path, split='train')
        # ==================================================================================

    def __len__(self):
        """
        重写Dataset的长度方法
        返回：数据集的总样本数量
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        重写Dataset的索引取值方法，根据索引获取单条样本并预处理
        Args:
            index: 样本索引值
        Returns:
            input_ids: 模型输入的token id序列 (已填充至max_length)
            labels: 预训练标签序列 (与input_ids一致，padding位设为-100)
        """
        # 根据索引获取单条原始样本数据
        sample = self.samples[index]
        # 对样本文本进行分词：
        # 1. 转为字符串避免非文本数据报错
        # 2. add_special_tokens=False：暂不自动添加特殊token
        # 3. max_length=self.max_length-2：预留2个位置给首尾特殊token
        # 4. truncation=True：超长文本自动截断
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        ).input_ids

        # 手动添加起始符(BOS)和结束符(EOS)
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 对短文本进行填充：用pad_token_id填充至max_length
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        # 转换为PyTorch长整型张量
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 构建预训练标签：标签与输入完全一致（语言模型自监督学习）
        labels = input_ids.clone()
        # 关键处理：将填充位(pad)的标签设为-100，PyTorch会自动忽略该位置的损失计算
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # 返回模型训练所需的输入和标签
        return input_ids, labels


class SFTDataset(Dataset):
    """
    全量微调模型回答结构化数据输出与理解
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        # 将传入的tokenizer保存为类的属性，tokenizer通常用于将文本转换为模型可以处理的token等操作
        self.tokenizer = tokenizer
        # 将传入的最大长度保存为类的属性，这个最大长度可能用于限制输入文本的长度
        self.max_length = max_length
        # Features 就像是一个用来描述数据集特征结构的模板。在你之前的代码里，用它来定义了数据集中 conversations
        # 这个部分的详细样子，每个子部分是什么类型都能靠它规定。
        # Value 是用来指定数据具体类型的。比如在定义 conversations 的特征时，用 Value('string') 来说明像 role 、content
        # 这些字段的数据类型是字符串。简单理解，Value 就是用来给数据定个 “型”，告诉程序这部分数据是什么类型的。
        features = Features({'conversations': [{
            'role'             : Value('string'),
            'content'          : Value('string'),
            'reasoning_content': Value('string'),
            'tools'            : Value('string'),
            'tool_calls'       : Value('string')
        }]})
        # 从指定路径的jsonl文件中加载数据集，只加载'train'分割的数据，并应用上面定义的特征
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # 获取tokenizer中表示开始的token对应的id，这里通过对特定字符串进行编码得到，add_special_tokens=False表示不添加额外的特殊标记
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        # 获取tokenizer中表示结束的token对应的id，同样通过对特定字符串进行编码得到，add_special_tokens=False表示不添加额外的特殊标记
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        # 返回数据集中样本的数量，也就是self.samples的长度
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        提取出工具列表，进行工具回调生成
        :param conversations: 包含多个批次的数据
        :return:
        """
        # 初始化一个空列表，用于存储处理后的消息
        messages = []
        # 初始化tools为None，tools可能用于一些特定的处理
        tools = None
        # 遍历对话中的每一条消息
        for message in conversations:
            # 将消息转换为字典形式
            message = dict(message)
            # 如果消息的角色是"system"并且包含"tools"字段
            if message.get("role") == "system" and message.get("tools"):
                # 如果tools是字符串形式，将其解析为json对象，否则直接使用
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            # 如果消息包含"tool_calls"字段并且是字符串形式
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                # 将其解析为json对象
                message["tool_calls"] = json.loads(message["tool_calls"])
            # 将处理后的消息添加到messages列表中
            messages.append(message)
        # 使用tokenizer的apply_chat_template方法生成聊天提示，tokenize=False表示不进行token化，
        # add_generation_prompt=False表示不添加生成提示，传入tools
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """
        提取self.bos_id【'<|im_start|>assistant\n'】与self.eos_id【'<|im_end|>\n'】之间的内容作为label
        :param input_ids:
        :return:
        """
        # 初始化一个长度与input_ids相同的列表，所有元素为-100，这个列表用于存储标签
        labels = [-100] * len(input_ids)
        # 初始化索引i为0
        i = 0
        # 遍历input_ids
        while i < len(input_ids):
            # 如果当前位置开始的一段id与开始标记的id相同
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 计算开始位置
                start = i + len(self.bos_id)
                # 初始化结束位置为开始位置
                end = start
                ##### 找到结束标记的位置
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 对于开始和结束标记之间的部分，将标签设置为对应的input_ids
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                # 更新索引i，跳过当前这段内容
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                # 如果当前位置不是开始标记，直接移动到下一个位置
                i += 1
        # 返回生成的标签
        return labels

    def __getitem__(self, index):
        # 获取指定索引的样本
        sample = self.samples[index]
        # 对样本中的对话进行预处理
        conversations = pre_processing_chat(sample['conversations'])
        # 根据对话生成聊天提示
        prompt = self.create_chat_prompt(conversations)
        # 对生成的聊天提示进行后处理 完全就是字符串形式
        prompt = post_processing_chat(prompt)
        # 将聊天提示转换为token id，并截取到最大长度
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 使用填充token id将input_ids填充到最大长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 根据input_ids生成标签
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        # 将input_ids和labels转换为torch张量并返回
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen'   : x_chosen,
            'y_chosen'   : y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected' : x_rejected,
            'y_rejected' : y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """
    PPO：基于RLAIF机制来训练模型
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }


class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass

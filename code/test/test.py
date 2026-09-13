# -*- coding: utf-8 -*-
"""
model_grpo格式
整理自 Jupyter Notebook
"""

import os
import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== 全局配置 ====================
# 模型路径（GRPO 格式微调后的 checkpoint）
MODEL_PATH = "/run/ai_report/lijiusong/output/model/GRPO_format/checkpoint-1500"
# 数据集路径
DATA_PATH = "../../data_source/formatted_RL_table_samples1225.json"


# ==================== 函数定义 ====================
def load_pretrain_samples(load_path):
    """
    从本地JSON文件读取并还原预训练样本列表（pretrain_samples）

    Args:
        load_path: 保存文件的路径（支持字符串或Path对象）

    Returns:
        pretrain_samples: 还原后的预训练样本列表，结构与保存时一致（[{"text": 样本文本1}, ...]）

    Raises:
        FileNotFoundError: 若指定路径的文件不存在，抛出文件未找到异常
        json.JSONDecodeError: 若文件内容不是合法JSON格式，抛出解析异常
    """
    # 统一路径格式为Path对象
    load_path = Path(load_path) if isinstance(load_path, str) else load_path

    # 检查文件是否存在，不存在则抛异常（提前暴露错误，避免后续逻辑崩溃）
    if not load_path.exists():
        raise FileNotFoundError(f"预训练样本文件不存在: {load_path}")

    # 读取并解析JSON文件
    with open(load_path, "r", encoding="utf-8") as f:
        pretrain_samples = json.load(f)

    # 验证还原后的数据结构（可选：避免文件被篡改导致后续逻辑出错）
    if not isinstance(pretrain_samples, list) or (
        pretrain_samples and not isinstance(pretrain_samples[0], dict)
    ):
        raise ValueError(
            f"文件{load_path}内容格式错误，需为'列表套字典'结构（[{{'text': ...}}, ...]）"
        )

    # 打印加载信息（明确样本数量，便于验证）
    print(f"已从{load_path}加载预训练样本，共{len(pretrain_samples)}个样本")
    return pretrain_samples


def Inspect_format(response):
    """
    检查模型回答是否包含规定的 XML 标签，并按标签内容打分。
    每个有效标签（内容非空）得 2.5 分，共 4 个标签，满分 10 分。
    """
    required_tags = [
        ("<reasoning>", "</reasoning>"),
        ("<error_position>", "</error_position>"),
        ("<specific_error_reason>", "</specific_error_reason>"),
        ("<corrected_text>", "</corrected_text>"),
    ]

    score = 0.0
    # 检查所有标签是否存在且格式正确
    all_tags_present = True
    tag_scores = []

    for start_tag, end_tag in required_tags:
        try:  # 增加异常捕获
            if start_tag in response and end_tag in response:
                start_idx = response.index(start_tag) + len(start_tag)
                end_idx = response.index(end_tag, start_idx)
                content = response[start_idx:end_idx].strip()
                if content:
                    tag_scores.append(2.5)  # 每个有效标签2.5分，总计10分
                else:
                    tag_scores.append(0.25)  # 标签存在但内容为空
            else:
                tag_scores.append(0.0)
        except ValueError:  # 标签顺序错误/嵌套错误
            tag_scores.append(0.0)

    # 计算总分
    score = sum(tag_scores)
    return score


# ==================== 模型加载与推理 ====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载保存的模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=True
)


def get_qwen_7B_answer(prompt):
    """
    使用 Qwen2.5-Coder-7B-Instruct 模型对输入 prompt 进行纠错回答。

    Args:
        prompt: 用户输入的待纠错文本

    Returns:
        response: 模型生成的 XML 格式回答
    """
    SYSTEM_PROMPT = """You are a professional expert in text error diagnosis and correction, with specialized expertise in the field of AECQ automotive-grade chip testing.

### STRICT FORMAT REQUIREMENT (MUST FOLLOW EXACTLY) ###
You MUST respond in the following XML-like format with NO extra content, NO explanations, NO additional text:

<reasoning>
[Your reasoning process here]
</reasoning>
<error_position>
[Smallest possible error-containing segment (write "none" if no error)]
</error_position>
<specific_error_reason>
[Brief error reason (write "none" if no error)]
</specific_error_reason>
<corrected_text>
[Complete revised text]
</corrected_text>

### EXAMPLE ###
Input text: "The ambient operating temperature range for Grade 3 in AEC-Q100 Rev-H is -40°C to +95°C."
Your response should be:
<reasoning>
The given temperature range is incorrect according to the AEC-Q100 Rev-H specification. The standard specifies that Grade 3 has an ambient operating temperature range of -40°C to +85°C, not +95°C.
</reasoning>
<error_position>
The ambient operating temperature range for Grade 3 in AEC-Q100 Rev-H is -40°C to +95°C.
</error_position>
<specific_error_reason>
The upper temperature limit for Grade 3 is specified as +85°C, but +95°C is incorrectly stated in the text.
</specific_error_reason>
<corrected_text>
The ambient operating temperature range for Grade 3 in AEC-Q100 Rev-H is -40°C to +85°C.
</corrected_text>

### TASKS ###
1. Judge if there is an error in the text based on what you know.
2. Precisely extract the error-containing sentence.Extract the smallest complete sentence (or clause) containing the error.
3. Explain the specific cause of the error briefly.
4. Generate the complete revised text.

### FORMAT VIOLATION ###
If you do not follow this exact format, your response will be considered completely invalid."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=600)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    return response


# ==================== 主流程 ====================
def main():
    # 加载数据集
    pretrain_samples = load_pretrain_samples(DATA_PATH)

    # 遍历每个样本，调用模型生成回答并打分
    for idx, item in enumerate(pretrain_samples):
        query = item["error_text"]
        response = get_qwen_7B_answer(query)
        score = Inspect_format(response)
        print(f"样本{idx},分数:{score}")
        if score < 10.0:
            print(f"错误回答:\n{response}")


if __name__ == "__main__":
    main()
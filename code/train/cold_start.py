# -*- coding: utf-8 -*-
"""
增强单卡全量续训练21_综合混合ft数据_少长度
整理自 Jupyter Notebook
"""

import os
import json
import traceback
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

# ==================== 全局配置 ====================
# 输出目录（新模型保存路径）
OUTPUT_DIR = "/run/ai_report/lijiusong/output/model/zqv_14B_LS_TI_LL"
# 本地模型目录
local_model_dir = "/run/ai_report/lijiusong/qwen2.5-14b-instruct"
# 缓存目录
CACHE_DIR = "/run/ai_report/lijiusong"


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


def get_samples_from_folder(folder_path):
    """
    从文件夹中读取所有 JSON 文件，并合并样本
    """
    folder_path = Path(folder_path) if isinstance(folder_path, str) else folder_path
    all_samples = []  # 用于累积所有文件的semantic_units

    # 获取文件夹下所有.json文件（只处理json文件）
    json_files = [
        file
        for file in folder_path.iterdir()
        if file.is_file() and file.suffix.lower() == ".json"
    ]
    for json_file in json_files:
        print(f"正在处理文件: {json_file.name}...")
        pretrain_samples = load_pretrain_samples(json_file)
        print(f"获得{len(pretrain_samples)}个样本")
        all_samples.extend(pretrain_samples)
    print(f"总共获得了{len(all_samples)}个样本")
    return all_samples


# ==================== 主流程 ====================
def main():
    # 关键：强制仅暴露第0张GPU，避免多卡检测触发分布式
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    # 关键：禁用NCCL的分布式初始化（可选，进一步避免冲突）
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "WARN"  # 仅在调试时启用，可改为"INFO"查看详情

    try:
        # ------------------- 构造预训练数据集 -------------------
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_dir,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            local_files_only=True,  # 仅使用本地文件，不尝试从远程下载
        )
        tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

        # 加载两个文件夹的样本并合并
        folder_path1 = "./少量样本_ft"
        pretrain_samples = get_samples_from_folder(folder_path1)

        folder_path1 = "./少量样本_少长度"
        pretrain_samples1 = get_samples_from_folder(folder_path1)
        pretrain_samples.extend(pretrain_samples1)

        # 构建 HuggingFace Dataset
        dataset = Dataset.from_list(pretrain_samples)

        # 分词处理
        def tokenize_function(examples):
            # 先进行分词
            outputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048,
                padding="max_length",
            )
            # 将 input_ids 作为 labels（因果语言模型训练需要）
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs

        print("开始数据集预处理...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=8,  # 并行处理进程数
        )
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.001)
        print(
            f"数据集处理完成，训练样本: {len(tokenized_dataset['train'])}，"
            f"验证样本: {len(tokenized_dataset['test'])}"
        )

        # =============== 步骤3: 继续预训练 ===============
        # ----------------全量训练不可取-------------------------------
        # 单卡训练：明确指定使用第0张GPU
        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            cache_dir=CACHE_DIR,
            dtype=torch.bfloat16,  # 使用BF16精度节省显存
            device_map="auto",     # 自动分配到多卡
            trust_remote_code=True,
            local_files_only=True,
        )

        model.config.use_cache = False
        # 启用梯度检查点节省显存（注意：原代码调用的是 disable）
        model.gradient_checkpointing_disable()

        # 训练参数配置 - 单卡训练设置
        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/training_output",
            num_train_epochs=10,                # 根据需求调整
            per_device_train_batch_size=2,      # 单卡的batch_size，根据显存调整
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,      # 梯度累积，总batch_size=2*4=8
            learning_rate=3e-5,                 # 初始学习率可适当提高，配合衰减
            lr_scheduler_type="cosine_with_restarts",  # 余弦退火调度器
            warmup_steps=250,                   # 延长预热步数，稳定前期训练
            weight_decay=0,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=20,
            eval_steps=20,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=150,
            bf16=True,                          # 如显卡支持BF16（如A100、3090等）
            report_to="tensorboard",
            optim="adamw_torch_fused",
            # 单卡训练：移除分布式相关配置
            dataloader_num_workers=2,           # 单卡可适当提高工作进程数
            local_rank=-1,                      # 核心参数
            # 节省显存选项
            gradient_checkpointing=False,
            fp16=False,
            tf32=True,                          # 启用TF32加速（如显卡支持）
            label_smoothing_factor=0,           # 标签平滑正则化
            max_grad_norm=1.0,                  # 梯度裁剪
            # 数据与保存策略（基于损失值的保存策略）
            dataloader_drop_last=True,
            load_best_model_at_end=False,
            metric_for_best_model=None,         # 改为使用损失值作为判断标准
            greater_is_better=False,            # 损失值越小越好
            save_total_limit=2,                 # 最多保存2个模型文件
        )

        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
        )

        # 开始训练
        print("开始继续预训练...")
        trainer.train()
        print("训练完成!")

        # =============== 步骤4: 保存新模型 ===============
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"新模型已保存到: {OUTPUT_DIR}")

    except Exception as e:
        print("训练报错详情：")
        traceback.print_exc()  # 打印完整堆栈
        raise e  # 重新抛出异常，不影响后续清理

    finally:
        # 清理
        torch.cuda.empty_cache()
        print("结束啦")


if __name__ == "__main__":
    main()
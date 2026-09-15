import os
import re
import json
import torch
import numpy as np
from typing import List
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# -------------------------- Environment Configuration --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# -------------------------- Path Configuration --------------------------
# Routed through `override` so an ablation variant can redirect them without editing this file.
# With no ABLATION_* environment set the defaults below apply and a direct run is unchanged.
from ablation_config import override, announce  # noqa: E402

MODEL_PATH = override("3grpo.MODEL_PATH", "../model/output/2Cold_start_output")
OUTPUT_DIR = override("3grpo.OUTPUT_DIR", "../model/output/3GRPO_format_output")
data_path = override("3grpo.DATA_PATH",
                     "../../data/train/RL_data/GRPO_format/RL_format_samples.json")

# -------------------------- Simplified Prompt --------------------------
SYSTEM_PROMPT = """You are a professional expert in text error diagnosis and correction, with specialized expertise in the field of AEC-Q automotive-grade chip testing.
###  TASKS ###
1. Judge if there is an error in the text based on what you know.  
2. Precisely extract the smallest complete error-containing sentence (or clause).
3. Explain the specific cause of the error briefly.  
4. Generate the complete revised text.
###  STRICT FORMAT REQUIREMENT (MUST FOLLOW EXACTLY) ###
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
###  FORMAT VIOLATION ###
If you do not follow this exact format, your response will be considered completely invalid.
###  INPUT ###
Text:  {Input_text}
"""

# -------------------------- Dataset Processing --------------------------
def get_correction_dataset(split="train") -> (Dataset, Dataset):
    data = load_dataset('json', data_files=data_path)['train']
    print(data)
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['error_text']}
        ],
        'error_text':x['error_text'],
        'answer': x['corrected_content'],
        'error_position': x['error_position'],
        'specific_error_reason': x['specific_error_reason'],
        'corrected_text': x['corrected_content'],
        'reasoning': x['reasoning']
    })
    data = data.shuffle(seed=42)
    return data

# Load dataset
train_dataset = get_correction_dataset()

####### Core Reward Function Design (Format-only Validation) #######
class RewardManager:
    def __init__(self):
        self.reward_history = []
        self.baseline = 0.0
        self.alpha = 0.9

    def extract_field(self, text: str, field: str) -> str:
        """Improved field extraction"""
        start_tag = f"<{field}>"
        end_tag = f"</{field}>"
        if start_tag not in text or end_tag not in text:
            return "not find"
        try:
            start_idx = text.index(start_tag) + len(start_tag)
            end_idx = text.index(end_tag, start_idx)
            return text[start_idx:end_idx].strip()
        except:
            return "not find"

    def core_correction_reward(self, prompts, completions, **kwargs):
        """Format-only reward: tag existence + tag order validation"""
        responses = [completion[0]['content'] for completion in completions]
        rewards = []

        for response in responses:
            # Extract all required format fields
            res_reason = self.extract_field(response, 'reasoning')
            res_pos = self.extract_field(response, 'error_position')
            res_spec = self.extract_field(response, 'specific_error_reason')
            res_corr = self.extract_field(response, 'corrected_text')

            total_reward = 0.0

            # 1.8 points for each existing valid tag
            if res_reason != "not find":
                total_reward += 1.8
            if res_pos != "not find":
                total_reward += 1.8
            if res_spec != "not find":
                total_reward += 1.8
            if res_corr != "not find":
                total_reward += 1.8

            # Additional 2.8 points if all tags exist and order is correct
            all_tags_exist = (res_reason != "not find" and
                              res_pos != "not find" and
                              res_spec != "not find" and
                              res_corr != "not find")

            if all_tags_exist:
                idx_reason = response.index("<reasoning>")
                idx_pos = response.index("<error_position>")
                idx_spec = response.index("<specific_error_reason>")
                idx_corr = response.index("<corrected_text>")

                # Correct order: reasoning → error_position → specific_error_reason → corrected_text
                if idx_reason < idx_pos < idx_spec < idx_corr:
                    total_reward += 2.8

            # Clamp reward within [0, 10]
            total_reward = max(0.0, min(total_reward, 10.0))
            rewards.append(total_reward)

        # Update baseline and training log
        avg_reward = np.mean(rewards)
        self.baseline = self.alpha * self.baseline + (1 - self.alpha) * avg_reward
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        self.reward_history.append(max_reward)

        print(f"step-{len(self.reward_history)} -Baseline: {self.baseline:.3f}, Avg: {avg_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
        return rewards

# -------------------------- Training Configuration --------------------------
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    run_name="Qwen-Correction-GRPO-format-v2",
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.95,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type='cosine',
    logging_steps=5,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=4,
    generation_batch_size=4,
    max_prompt_length=896,
    max_completion_length=1280,
    num_train_epochs=10,
    save_steps=100,
    max_grad_norm=1.0,
    save_total_limit=2,
    report_to="none",
)

# -------------------------- Model Loading --------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.generation_config.temperature = 1.3
model.generation_config.top_p = 0.93
model.generation_config.top_k = -1
model.generation_config.do_sample = True
model.generation_config.max_new_tokens = training_args.max_completion_length
model.generation_config.repetition_penalty = 1.1
model.generation_config.num_beams = 1

# -------------------------- Training Preparation --------------------------
reward_manager = RewardManager()
def core_reward_wrapper(prompts, completions, **kwargs):
    return reward_manager.core_correction_reward(prompts, completions, **kwargs)

# -------------------------- Trainer Initialization --------------------------
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=core_reward_wrapper,
    args=training_args,
    train_dataset=train_dataset,
)

def save_list_as_string_to_txt(lst, file_path):
    """
    Convert list directly to string and save to txt file
    :param lst: List of any type (supports strings, numbers, Chinese, nested lists/dicts, etc.)
    :param file_path: Save path (e.g. "./list_str.txt")
    """
    list_str = str(lst)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(list_str)
    print(f"List saved as string to: {file_path}")

# -------------------------- Training Loop --------------------------
try:
    print("Starting GRPO training...")
    trainer.train()
    print("Training completed, saving model...")
finally:
    reward_path = "./reward.txt"
    save_list_as_string_to_txt(reward_manager.reward_history, reward_path)

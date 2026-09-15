import os
import re
import json
import torch
import numpy as np
from typing import List
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from collections import Counter

# -------------------------- Environment Configuration --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# -------------------------- Path Configuration --------------------------
MODEL_PATH = "../model/output/3GRPO_format_output"
OUTPUT_DIR = "../model/output/4GRPO_content_output"
data_path = "../../data/train/RL_data/GRPO_content/RL_content_samples.json"

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

# -------------------------- Dataset Processing (Core Optimization: Label Validation + Shuffling + Validation Split) --------------------------
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

####### Code-based Reward Calculation Section
def prepare_text(text):
    """
    Use regular expressions to insert spaces before and after each Chinese character,
    so that the text can be split into words by spaces in subsequent steps.
    """
    text = re.sub(r'([\u4e00-\u9fff])', r' \1 ', text)
    return text

def get_ngrams(words, n, deduplicate=False):
    """
    Generate n-gram list from text
    Args:
        words: List of split words (e.g. [char1, char2] or [word1, word2])
        n: N value for n-gram (e.g. 1/2/3)
        deduplicate: Whether to deduplicate (to avoid interference from repeated words)
    Returns:
        n-gram list
    """
    if len(words) < n:
        return []
    ngrams = []
    for i in range(len(words) - n + 1):
        sub_words = words[i:i+n]
        ngram_str = "-".join(words[i:i+n])
        ngrams.append(ngram_str)
    if deduplicate:
        ngrams = list(set(ngrams))
    return ngrams

def calculate_f1(pred_grams, gold_grams) -> float:
    """Improved F1 calculation, no deduplication, preserves order information"""
    pred_counter = Counter(pred_grams)
    gold_counter = Counter(gold_grams)
    common = sum((pred_counter & gold_counter).values())
    precision = common / len(pred_grams) if pred_grams else 0
    recall = common / len(gold_grams) if gold_grams else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def calculate_3gram_f1(pred: str, gold: str) -> float:
    pred_text = prepare_text(pred)
    gold_text = prepare_text(gold)
    # Split text directly by spaces
    pred_words = pred_text.split()
    gold_words = gold_text.split()
    if len(pred_words) == 0 or len(gold_words) == 0:
        return 0.0
    if len(pred_words) < 3 and len(gold_words) < 3:
        if pred == gold:
            return 1.0
        else:
            return 0.0
    pred_3grams = get_ngrams(pred_words, 3, deduplicate=False)
    gold_3grams = get_ngrams(gold_words, 3, deduplicate=False)
    score = calculate_f1(pred_3grams, gold_3grams)
    return score

####### LLM as a Judge Reward Calculation Section
def extract_xml_score(text: str) -> str:
    match = re.search(r'([01].\d)\s*$', text.strip())
    if match:
        return match.group(1)
    return "0.0"

def str_to_num(s):
    """Safely convert string to float, return 0.0 on conversion failure"""
    try:
        return float(s)
    except ValueError:
        return 0.0

from openai import OpenAI
client = OpenAI(
        api_key="xx-xxxxxxxxxxxxxxxxxxxxx",
        base_url="https://api.deepseek.com")

def llm_as_a_judge(error_text,AI_pos,AI_rsn,AI_cor,gold_pos,gold_rsn,gold_cor):
    judge_prompt = f"""You are a rigorous yet flexible evaluation expert for error‑correction models in the AEC‑Q automotive chip testing domain. Evaluate model output and give final score per rules below.
## Evaluation Rules
Step‑wise cumulative scoring. Check AI output against gold standard in order: Error Location → Error Reason → Corrected Text. **Subsequent steps are skipped if current step fails**.
1. Step1: Check "Error Location". 0.3 points for matching location, proceed; otherwise 0.0 points, stop.
2. Step2: Check "Error Reason". Add 0.4 points (total 0.7) for equivalent reason, proceed; otherwise keep 0.3 points, stop.
3. Step3: Check "Corrected Text". Add 0.3 points to reach full score for consistent correction; otherwise keep 0.7 points, stop.
## Equivalence Criteria
- **Error Location**: Different wording allowed, must point to identical text span or semantic unit.
- **Error Reason**: Different diction & sentence structure allowed, must retain the core error point.
- **Corrected Text**: Synonym substitution and rephrasing allowed. Revised professional content must be equivalent to gold standard and free of new errors.
## Input Information
- Original erroneous text: {{error_text}}
- AI predicted error location: {{AI_pos}}
- AI predicted error reason: {{AI_rsn}}
- AI corrected text: {{AI_cor}}
- Gold‑standard error location: {{gold_pos}}
- Gold‑standard error reason: {{gold_rsn}}
- Gold‑standard corrected text: {{gold_cor}}
## Output Format
Output strictly following the structure below:
<reasoning>
[Write step‑by‑step reasoning, state comparison basis and points obtained at each step.]
</reasoning>
<score>
[Final score, valid values: 0.0 / 0.3 / 0.7 / 1.0]
</score>
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": judge_prompt},
        ],
        stream=False
    )
    res = response.choices[0].message.content
    score = extract_xml_score(res)
    score = str_to_num(score)
    return score

# Hard Sample Section
def structured_data_to_string(error_text,reasoning,gold_pos,gold_spec,gold_corr):
    return f"{error_text}-*-{reasoning}-*-{gold_pos}-*-{gold_spec}-*-{gold_corr}"

def save_set_to_txt(set_data, file_path, separator="*|||*"):
    """
    Save a set of strings to a txt file, elements separated by the specified delimiter
    :param set_data: String set to save
    :param file_path: Path to the output file
    :param separator: Delimiter (multi-character recommended to avoid conflict with content)
    """
    str_data = separator.join(set_data)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str_data)
    print(f"Hard samples successfully saved to {file_path}")

####### Core Reward Function Design
class RewardManager:
    def __init__(self):
        self.reward_history = []
        self.baseline = 0.0
        self.alpha = 0.9
        self.hard_reward = 9.0
        self.hard_sams_set = set()

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
        """Improved core reward function"""
        responses = [completion[0]['content'] for completion in completions]
        error_positions = kwargs.get('error_position', [])
        corrected_texts = kwargs.get('corrected_text', [])
        error_texts = kwargs.get('error_text', [])
        specific_error_reasons = kwargs.get('specific_error_reason', [])
        reasonings = kwargs.get('reasoning', [])
        rewards = []
        for response, gold_pos, gold_spec, gold_corr, error_text in zip(responses, error_positions, specific_error_reasons, corrected_texts, error_texts):
            res_reason = self.extract_field(response, 'reasoning')
            res_pos = self.extract_field(response, 'error_position')
            res_spec = self.extract_field(response, 'specific_error_reason')
            res_corr = self.extract_field(response, 'corrected_text')
            total_reward = 0.0
            if all([res_reason != "not find", res_pos != "not find", res_spec != "not find", res_corr != "not find" ]):
                format_score = 0.5
                if gold_pos == "" and res_pos.lower() == "none":
                    pos_score = 8.0
                else:
                    pos_f1 = calculate_3gram_f1(res_pos, gold_pos)
                    pos_score = pos_f1 * 8.0
                corr_f1 = calculate_3gram_f1(res_corr, gold_corr)
                corr_score = corr_f1 * 1.5
                code_reward = format_score + pos_score + corr_score
                code_reward = max(0.0, min(code_reward, 10.0))
                judge_reward = llm_as_a_judge(error_text,res_pos,res_spec,res_corr,
                                             gold_pos,gold_spec,gold_corr)
                total_reward = code_reward * 0.1 + judge_reward * 9
            else:
                print("format error,reward:0.0")
                total_reward = 0.0
            rewards.append(total_reward)
        avg_reward = np.mean(rewards)
        self.baseline = self.alpha * self.baseline + (1 - self.alpha) * avg_reward
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        self.reward_history.append(max_reward)
        sam_str = structured_data_to_string(error_texts[0],reasonings[0],
                                           error_positions[0],specific_error_reasons[0],corrected_texts[0])
        if max_reward < self.hard_reward:
            if sam_str not in self.hard_sams_set:
                self.hard_sams_set.add(sam_str)
        else:
            if sam_str in self.hard_sams_set:
                self.hard_sams_set.remove(sam_str)
        print(f"step-{len(self.reward_history)} -Baseline: {self.baseline:.3f}, Avg: {avg_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
        return rewards

    def save2txt(self,content,file_path):
        with open(file_path, mode='a', encoding='utf-8') as f:
            f.write(content)

# -------------------------- Training Configuration --------------------------
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    run_name="Qwen-Correction-GRPO-v2",
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
model.generation_config.temperature = 0.2
model.generation_config.top_p = 0.6
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

# -------------------------- Training Loop (with Early Stopping) --------------------------
try:
    print("Starting GRPO training...")
    trainer.train()
    print("Training completed, saving model...")
finally:
    file_path = "../../data/train/Hard_samples/hard_samples.txt"
    separator = "*|||*\n"       
    save_set_to_txt(reward_manager.hard_sams_set, file_path, separator)
    reward_path = "./reward.txt"
    save_list_as_string_to_txt(reward_manager.reward_history, reward_path)
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# ==================== Global Configuration ====================
# Note: MODEL_PATH and load_pretrain_samples are defined in the preceding notebook cells.
# When running independently, they must be defined here; otherwise a NameError will occur.
MODEL_PATH = "../model/output/1DAP_output"   # Modify according to your actual path
OUTPUT_DIR = "../model/output/2Cold_start_output"
DATA_PATH = "../../data/train/RL_data/Cold_start/cold_start_samples.json"

# ==================== Dependent Functions (from previous notebook cells) ====================
def load_pretrain_samples(load_path):
    """Read sample list from local JSON file."""
    import json
    from pathlib import Path
    load_path = Path(load_path) if isinstance(load_path, str) else load_path
    if not load_path.exists():
        raise FileNotFoundError(f"Sample file not found: {load_path}")
    with open(load_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list) or (samples and not isinstance(samples[0], dict)):
        raise ValueError(f"Invalid content format in file {load_path}. Expected list-of-dicts structure")
    print(f"Loaded {len(samples)} pretraining samples from {load_path}")
    return samples

# ==================== 1. Build Training Dataset ====================
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token          # Set padding token
tokenizer.pad_token_id = tokenizer.eos_token_id
END_TOKEN = tokenizer.eos_token

def format_prompt(example):
    """Format samples into complete text with chat template."""
    SYSTEM_PROMPT = """You are a professional expert in text error diagnosis and correction, with specialized expertise in the field of AEC-Q automotive-grade chip testing.
###  TASKS ###
1. Judge whether there is an error in the text based on your knowledge.
2. Precisely extract the smallest complete sentence (or clause) containing the error.
3. Briefly explain the specific cause of the error.
4. Generate the complete revised text.
###  STRICT FORMAT REQUIREMENT (MUST FOLLOW EXACTLY) ###
You MUST respond in the following XML format with NO extra content, NO explanations, NO additional text:
<reasoning>
[Your reasoning process here]
</reasoning>
<error_position>
[Smallest possible error-containing segment, write "none" if no error exists]
</error_position>
<specific_error_reason>
[Brief reason for the error, write "none" if no error exists]
</specific_error_reason>
<corrected_text>
[Complete revised text]
</corrected_text>
###  FORMAT VIOLATION ###
If you do not strictly follow this format, your response will be regarded as invalid.
###  INPUT ###
Text:  {Input_text}
"""
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["error_text"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}

data_path = DATA_PATH
pretrain_samples = load_pretrain_samples(data_path)
pretrain_samples = pretrain_samples * 2                       # Double dataset size
tem_dataset = Dataset.from_list(pretrain_samples)
dataset = tem_dataset.map(format_prompt, remove_columns=tem_dataset.column_names)

# ==================== Tokenization Function ====================
def tokenize_function(examples):
    """Tokenize inputs and compute loss only for assistant responses (mask preceding parts)."""
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt",
    )
    # Construct labels: only compute loss for assistant response, mask instruction part
    labels = outputs["input_ids"].clone()
    # Critical: obtain full token sequence for "<|im_start|>assistant\n"
    assistant_start_text = "<|im_start|>assistant\n"
    assistant_start_tokens = tokenizer.encode(
        assistant_start_text, add_special_tokens=False
    )
    start_len = len(assistant_start_tokens)

    # Iterate over each sample, locate assistant start position and apply mask
    for i in range(len(labels)):
        input_ids = labels[i].tolist()
        start_idx = None
        # Sliding window matching for the full starting token sequence
        for j in range(len(input_ids) - start_len + 1):
            if input_ids[j:j + start_len] == assistant_start_tokens:
                start_idx = j + start_len   # Start loss calculation from response content
                break
        # Mask content before the starting position
        if start_idx is not None:
            labels[i, :start_idx] = -100
        else:
            # Mask all tokens if start marker is not found (prevent training error)
            labels[i, :] = -100
    outputs["labels"] = labels
    return outputs

print("Starting dataset preprocessing...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],     # Fix: original code uses "text" string; passed as list
    num_proc=8,
)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.02)
print(
    f"Dataset processing completed. Training samples: {len(tokenized_dataset['train'])}, "
    f"Validation samples: {len(tokenized_dataset['test'])}"
)

# ==================== Step3: Continued Fine-tuning ====================
# ---------------- Full-parameter training is not recommended ----------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
# tokenizer.padding_side = "left"   # Not important for training; important for inference
model.config.use_cache = False
# Enable gradient checkpointing to save VRAM
model.gradient_checkpointing_enable()

# ==================== Training Argument Configuration ====================
training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/training_output",
    num_train_epochs=20,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,              # Effective batch = 2*4=8 (old comment value 16 was incorrect)
    learning_rate=3e-5,
    lr_scheduler_type="cosine_with_restarts",
    warmup_steps=10,
    weight_decay=0.03,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=2,
    eval_steps=2,
    save_strategy="steps",
    eval_strategy="steps",
    save_steps=20,
    bf16=True,
    report_to="tensorboard",
    optim="adamw_torch_fused",
    # Single-GPU training: remove distributed related settings
    dataloader_num_workers=2,
    local_rank=-1,
    # VRAM saving options
    gradient_checkpointing=True,
    fp16=False,
    tf32=True,
    label_smoothing_factor=0,
    max_grad_norm=1.0,
    # Data and checkpoint saving strategy
    dataloader_drop_last=True,
    load_best_model_at_end=False,
    metric_for_best_model=None,
    greater_is_better=False,
    save_total_limit=2,
)

# ==================== Initialize Trainer ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# ==================== Start Training ====================
print("Starting cold-start training...")
trainer.train()
print("Training completed!")

# ==================== Step4: Save New Model ====================
# model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)
print(f"The new model has been saved to: {OUTPUT_DIR}")

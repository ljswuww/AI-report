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

# ==================== Global Configuration ====================
# Base model path (input: GRPO stage 2 output)
MODEL_PATH = "../model/output/4GRPO_content_output"
# Output directory (DAP4HS trained model)
OUTPUT_DIR = "../model/output/5DAP4HS_output"
# Cache directory
CACHE_DIR = "../model/base/dap4hs_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Augmented hard sample data path
AUGMENTED_SAMPLES_PATH = "../../data/train/Hard_samples/agumented_hard_samples.json"

# System prompt template (100% consistent with GRPO stage to ensure paradigm alignment)
SYSTEM_PROMPT_TEMPLATE = """You are a professional expert in text error diagnosis and correction, with specialized expertise in the field of AEC-Q automotive-grade chip testing.

### TASKS ###

1. Judge if there is an error in the text based on what you know.  

2. Precisely extract the smallest complete error-containing sentence (or clause).

3. Explain the specific cause of the error briefly.  

4. Generate the complete revised text.  

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

### FORMAT VIOLATION ###

If you do not follow this exact format, your response will be considered completely invalid.

### INPUT ###

Text:  {Input_text}
"""


# ==================== Function Definitions ====================
def build_dap_text_from_sample(sample: dict) -> str:
    """
    Convert a single augmented hard sample into full DAP causal language modeling text.
    Structure: system prompt (with input) + standard formatted response
    """
    # 1. Replace input placeholder in system prompt
    input_text = sample["error_text"]
    system_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{Input_text}", input_text)
    
    # 2. Build standard response with strict tag alignment
    response = f"""<reasoning>
{sample["reasoning"]}
</reasoning>
<error_position>
{sample["error_position"] if sample["error_position"] else "none"}
</error_position>
<specific_error_reason>
{sample["specific_error_reason"] if sample["specific_error_reason"] else "none"}
</specific_error_reason>
<corrected_text>
{sample["corrected_content"]}
</corrected_text>"""
    
    # 3. Concatenate into full sequence for CLM training
    full_text = system_prompt + "\n" + response
    return full_text


def load_pretrain_samples(load_path):
    """
    Read and restore pretraining sample list from a local JSON file.
    Args:
        load_path: Path to the saved file (supports string or Path object)
    Returns:
        pretrain_samples: Restored pretraining sample list [{"text": sample_text_1}, ...]
    """
    load_path = Path(load_path) if isinstance(load_path, str) else load_path
    if not load_path.exists():
        raise FileNotFoundError(f"Pretraining sample file not found: {load_path}")
    
    with open(load_path, "r", encoding="utf-8") as f:
        pretrain_samples = json.load(f)
    
    if not isinstance(pretrain_samples, list) or (
        pretrain_samples and not isinstance(pretrain_samples[0], dict)
    ):
        raise ValueError(
            f"Invalid content format in file {load_path}. Expected list of dicts in format [{{'text': ...}}, ...]"
        )
    
    print(f"Loaded {len(pretrain_samples)} pretraining samples from {load_path}")
    return pretrain_samples


# ==================== Main Pipeline ====================
def main():
    # Environment configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

    try:
        # ------------------- Step 1: Data Conversion & Dataset Build -------------------
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load augmented hard samples and convert to DAP format on the fly
        print("=" * 60)
        print("Loading & converting augmented hard samples...")
        with open(AUGMENTED_SAMPLES_PATH, "r", encoding="utf-8") as f:
            augmented_samples = json.load(f)
        print(f"Loaded {len(augmented_samples)} raw augmented hard samples.")

        # Convert all samples to DAP CLM format
        pretrain_samples = []
        length_list = []
        for sample in augmented_samples:
            full_text = build_dap_text_from_sample(sample)
            pretrain_samples.append({"text": full_text})
            length_list.append(len(full_text))

        # Length statistics for verification
        avg_len = sum(length_list) / len(length_list)
        max_len = max(length_list)
        print(f"Converted to {len(pretrain_samples)} DAP training samples.")
        print(f"Text length - Avg: {avg_len:.0f} chars, Max: {max_len} chars")
        print(f"Estimated max tokens: ~{max_len // 1.8} (char→token ratio ~1.8:1)")
        print("=" * 60)

        # Build HuggingFace Dataset
        dataset = Dataset.from_list(pretrain_samples)

        # Tokenization
        def tokenize_function(examples):
            outputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048,
                padding="max_length",
            )
            # Standard CLM training: labels = input_ids
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs

        print("Starting dataset tokenization...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=8,
        )
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.001)
        print(
            f"Dataset processing completed. Training samples: {len(tokenized_dataset['train'])}, "
            f"Validation samples: {len(tokenized_dataset['test'])}"
        )

        # ------------------- Step 2: Model Loading -------------------
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_disable()

        # ------------------- Step 3: DAP Training Configuration -------------------
        # Parameters tuned for hard-sample targeted DAP: fewer epochs, moderate LR
        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/training_output",
            num_train_epochs=5,                     # Hard sample DAP: 3-5 epochs recommended (avoid overfitting)
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=3e-5,
            lr_scheduler_type="cosine_with_restarts",
            warmup_steps=250,
            weight_decay=0,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=20,
            eval_steps=50,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=150,
            bf16=True,
            report_to="none",
            optim="adamw_torch_fused",
            dataloader_num_workers=2,
            local_rank=-1,
            gradient_checkpointing=False,
            fp16=False,
            tf32=True,
            label_smoothing_factor=0,
            max_grad_norm=1.0,
            dataloader_drop_last=True,
            load_best_model_at_end=False,
            metric_for_best_model=None,
            greater_is_better=False,
            save_total_limit=2,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
        )

        # Start training
        print("=" * 60)
        print("Starting hard-sample targeted DAP training...")
        trainer.train()
        print("Training completed!")

        # ------------------- Step 4: Save Model -------------------
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"DAP4HS model saved to: {OUTPUT_DIR}")

    except Exception as e:
        print("Training error details:")
        traceback.print_exc()
        raise e

    finally:
        torch.cuda.empty_cache()
        print("Process finished")


if __name__ == "__main__":
    main()

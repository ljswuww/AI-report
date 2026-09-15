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
# Paths are routed through `override` so an ablation variant can redirect them without editing this
# file. With no ABLATION_* environment set the defaults below apply and a direct run is unchanged.
from ablation_config import override, announce  # noqa: E402

# Local model directory
local_model_dir = override("1dap.MODEL_PATH", "../model/base/qwen2.5-14b-instruct")
# Output directory (path to save the fine-tuned model)
OUTPUT_DIR = override("1dap.OUTPUT_DIR", "../model/output/1DAP_output")
# Cache directory
CACHE_DIR = "../model/base/dap_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================== Function Definitions ====================
def load_pretrain_samples(load_path):
    """
    Read and restore the pretraining sample list from a local JSON file.
    Args:
        load_path: Path to the saved file (supports string or Path object)
    Returns:
        pretrain_samples: Restored pretraining sample list with the same structure as saved ([{"text": sample_text_1}, ...])
    Raises:
        FileNotFoundError: Raised if the file at the specified path does not exist
        json.JSONDecodeError: Raised if the file content is not valid JSON format
    """
    # Normalize path to Path object
    load_path = Path(load_path) if isinstance(load_path, str) else load_path
    # Check if file exists; raise error early to avoid downstream failures
    if not load_path.exists():
        raise FileNotFoundError(f"Pretraining sample file not found: {load_path}")
    # Read and parse the JSON file
    with open(load_path, "r", encoding="utf-8") as f:
        pretrain_samples = json.load(f)
    # Validate restored data structure (prevents corrupted files from breaking downstream logic)
    if not isinstance(pretrain_samples, list) or (
        pretrain_samples and not isinstance(pretrain_samples[0], dict)
    ):
        raise ValueError(
            f"Invalid content format in file {load_path}. Expected list of dicts in format [{{'text': ...}}, ...]"
        )
    # Print loading info (sample count for verification)
    print(f"Loaded {len(pretrain_samples)} pretraining samples from {load_path}")
    return pretrain_samples

def get_samples_from_folder(folder_path):
    """
    Read all JSON files from a folder and merge the samples.
    """
    folder_path = Path(folder_path) if isinstance(folder_path, str) else folder_path
    all_samples = []  # Used to accumulate semantic units from all files
    # Get all .json files in the folder (process JSON files only)
    json_files = [
        file
        for file in folder_path.iterdir()
        if file.is_file() and file.suffix.lower() == ".json"
    ]
    for json_file in json_files:
        print(f"Processing file: {json_file.name}...")
        pretrain_samples = load_pretrain_samples(json_file)
        print(f"Obtained {len(pretrain_samples)} samples")
        all_samples.extend(pretrain_samples)
    print(f"Total samples collected: {len(all_samples)}")
    return all_samples

# ==================== Main Pipeline ====================
def main():
    # Critical: Restrict visible GPUs to avoid unintended distributed initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    # Critical: Disable NCCL P2P to prevent distributed initialization (optional, avoids conflicts)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "WARN"  # Enable only for debugging; set to "INFO" for detailed logs

    try:
        # ------------------- Build Pretraining Dataset -------------------
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_dir,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            local_files_only=True,  # Use local files only, do not attempt remote download
        )
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token

        # Load samples from two folders and merge
        folder_path1 = "../../data/train/DAP_data/few_shot_ft"
        pretrain_samples = get_samples_from_folder(folder_path1)
        folder_path1 = "../../data/train/DAP_data/few_shot_short_length"
        pretrain_samples1 = get_samples_from_folder(folder_path1)
        pretrain_samples.extend(pretrain_samples1)

        # Build HuggingFace Dataset
        dataset = Dataset.from_list(pretrain_samples)

        # Tokenization
        def tokenize_function(examples):
            # Perform tokenization first
            outputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048,
                padding="max_length",
            )
            # Use input_ids as labels (required for causal language modeling training)
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs

        print("Starting dataset preprocessing...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=8,  # Number of parallel processing workers
        )
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.001)
        print(
            f"Dataset processing completed. Training samples: {len(tokenized_dataset['train'])}, "
            f"Validation samples: {len(tokenized_dataset['test'])}"
        )

        # =============== Step 3: Continued Pretraining ===============
        # ---------------- Full training is not recommended -------------------------------
        # Single-GPU training: explicitly use GPU 0
        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            cache_dir=CACHE_DIR,
            dtype=torch.bfloat16,  # Use BF16 precision to save VRAM
            device_map="auto",     # Auto-distribute across available GPUs
            trust_remote_code=True,
            local_files_only=True,
        )
        model.config.use_cache = False
        # Gradient checkpointing disabled (note: this call disables gradient checkpointing)
        model.gradient_checkpointing_disable()

        # Training arguments - single-GPU configuration
        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/training_output",
            num_train_epochs=10,                # Adjust as needed
            per_device_train_batch_size=2,      # Batch size per GPU, adjust based on VRAM
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,      # Gradient accumulation, effective batch size = 2 * 4 = 8
            learning_rate=3e-5,                 # Initial learning rate, can be increased with scheduler decay
            lr_scheduler_type="cosine_with_restarts",  # Cosine annealing with restarts scheduler
            warmup_steps=250,                   # Extended warmup steps for stable early training
            weight_decay=0,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=20,
            eval_steps=20,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=150,
            bf16=True,                          # Enable if GPU supports BF16 (e.g. A100, RTX 3090)
            report_to="tensorboard",
            optim="adamw_torch_fused",
            # Single-GPU training: remove distributed-related configs
            dataloader_num_workers=2,           # Can be increased for single-GPU setup
            local_rank=-1,                      # Critical parameter to disable distributed mode
            # VRAM saving options
            gradient_checkpointing=False,
            fp16=False,
            tf32=True,                          # Enable TF32 acceleration (if supported by GPU)
            label_smoothing_factor=0,           # Label smoothing regularization
            max_grad_norm=1.0,                  # Gradient clipping
            # Data and saving strategy (loss-based checkpointing)
            dataloader_drop_last=True,
            load_best_model_at_end=False,
            metric_for_best_model=None,         # Use loss as the evaluation metric
            greater_is_better=False,            # Lower loss is better
            save_total_limit=2,                 # Keep maximum 2 checkpoint files
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
        print("Starting continued pretraining...")
        trainer.train()
        print("Training completed!")

        # =============== Step 4: Save the Fine-tuned Model ===============
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Fine-tuned model saved to: {OUTPUT_DIR}")

    except Exception as e:
        print("Training error details:")
        traceback.print_exc()  # Print full stack trace
        raise e  # Re-raise the exception without interfering with cleanup
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        print("Process finished")

if __name__ == "__main__":
    main()

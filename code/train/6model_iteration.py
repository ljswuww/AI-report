# -*- coding: utf-8 -*-
"""
Stage 6 -- Closed-loop iterative optimization (paper Sec. III-F).

After targeted hard-sample optimization (stage 5) the model audits hard cases better but is
prone to catastrophic forgetting: the previously learned knowledge is still latently encoded,
yet the probability of sampling a correct answer drops. Exploratory RL can reactivate it.
The loop therefore has three modules, matching the paper:

  1. Iterative training loop : GRPO on the FULL dataset (original samples + augmented hard
     samples) to strengthen weak skills while consolidating existing knowledge.
  2. Self-iteration mechanism: train -> bottleneck diagnosis -> data augmentation -> retrain,
     driven by a dynamically updated hard-sample pool (samples are cached when their reward is
     unsatisfactory and automatically evicted once the model handles them).
  3. Convergence check       : terminate when the validation mixed reward fluctuates by less
     than 3% for two consecutive rounds AND hard samples fall below 5% of the total pool.

Paper hyper-parameters: max iterations = 3, fluctuation tolerance = 3%, hard-sample fraction = 5%.

Pipeline position:
    stage 4 (4grpo_content.py)  -> 4GRPO_content_output
    stage 5 (5DAP_for_hardsamples.py) -> 5DAP4HS_output
    stage 6 (this script)       -> 6Iteration_output/iter_N + final

Run from the `code/train` directory (all paths below are relative to it).
"""

import os
import re
import sys
import json
import time
import random
import traceback
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from openai import OpenAI

# Sits beside this script, so no sys.path surgery is needed for it
from ablation_config import override, reward_mode, augmented_seed_enabled, announce

# ==================== Global Configuration ====================
# Environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Paths. Each is routed through `override` so an ablation variant can redirect it without editing
# this file; with no environment set the defaults below apply and a direct run is unchanged.
MODEL_PATH = override("6it.MODEL_PATH", "../model/output/5DAP4HS_output")   # stage 5 output
OUTPUT_DIR = override("6it.OUTPUT_DIR", "../model/output/6Iteration_output")
BASE_DATA_PATH = override(
    "6it.BASE_DATA_PATH", "../../data/train/RL_data/GRPO_content/RL_content_samples.json")
POOL_PATH = override("6it.POOL_PATH", "../../data/train/Hard_samples/iteration_pool.json")
AUGMENTED_PATH = override(
    "6it.AUGMENTED_PATH", "../../data/train/Hard_samples/iteration_augmented.json")
# Stage 5 already produced one round of augmented hard samples; seed the accumulated set with them.
# The `no-HSO` ablation disables the seed: that variant removes the module which produced this file,
# so inheriting it would import part of the removed module's effect back into the loop.
AUGMENTED_SEED_PATH = override(
    "6it.AUGMENTED_SEED_PATH", "../../data/train/Hard_samples/augmented_hard_samples.json")
REPORT_PATH = f"{OUTPUT_DIR}/iteration_report.json"

# Closed-loop settings (paper Sec. IV-C / III-F)
MAX_ITERATIONS = 3              # "total iterations = 3"
CONVERGENCE_TOL = 0.03          # validation mixed reward fluctuation < 3%
CONVERGENCE_PATIENCE = 2        # ...for two consecutive rounds
HARD_POOL_RATIO = 0.05          # hard-sample pool < 5% of the total sample pool

# Hard-sample diagnosis: 9.0 on the 0-10 reward scale, the paper's normalised 0.9 threshold.
# Assigned below from the shared definition, so stage 4, stage 6 and static diagnosis cannot drift
# apart on what counts as hard.

# Data split. The paper uses 9:1 train/validation over the CoT dataset; the split is fixed by
# seed so that every iteration is scored on exactly the same validation samples.
VAL_RATIO = 0.1
VAL_SEED = 42

# Per-iteration training. The paper does not fix these for stage 6; kept aligned with the
# content-GRPO stage (4grpo_content.py) and scaled down because the loop retrains MAX_ITERATIONS
# times. Tune freely.
EPOCHS_PER_ITERATION = 3

# Hard-sample augmentation: "rules" (deterministic, offline) or "llm" (LLM-generated variants).
# Both implement the same 1:5 ratio and the same four error types as the paper.
AUGMENT_MODE = "rules"
AUGMENT_VARIANT_NUM = 5

# LLM-as-a-judge. The paper states the GRPO-training judge and the ECA-evaluation judge must use
# DIFFERENT backbones and DIFFERENT prompts so the policy cannot overfit one judge's style.
# The convergence criterion is defined on the *validation mixed reward*, i.e. the training-style
# reward, so "training" is the default here. Set to "eca" to score validation with the
# all-or-nothing ECA judge instead.
JUDGE_MODEL = "deepseek-chat"
JUDGE_STYLE = "training"        # "training" | "eca"
JUDGE_MAX_RETRIES = 3

# Reward aggregate. "hybrid" is Eq. 7; "code_only" drops the judge term and keeps the code
# component on its native 0-10 scale, which is what the `no-HR` ablation runs. Read once at import.
REWARD_MODE = reward_mode()

UTILS_DIR = Path(__file__).resolve().parent / "utils"
sys.path.insert(0, str(UTILS_DIR))

# The reward has one definition, shared with the other stages and with static hard-sample
# diagnosis. Importing it here instead of redeclaring it is what keeps a variant's pool selected
# by the same criterion as the full pipeline's.
from reward_utils import (  # noqa: E402
    compute_mixed_reward, llm_as_a_judge, sample_key, DEFAULT_HARD_REWARD_THRESHOLD,
)

HARD_REWARD_THRESHOLD = DEFAULT_HARD_REWARD_THRESHOLD

# ==================== Prompts ====================
# Must stay byte-identical to 4grpo_content.py / 5DAP_for_hardsamples.py so that the policy sees
# the same input paradigm at every stage.
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

# -------------------------- LLM-as-a-judge client --------------------------
# NOTE: replace with your real key. LLM_augmented_hardsamples.py keeps its own client,
# so if AUGMENT_MODE == "llm" you must set the key there as well.
client = OpenAI(
    api_key="xx-xxxxxxxxxxxxxxxxxxxxx",
    base_url="https://api.deepseek.com",
)


def judge_style_score(error_text, ai_pos, ai_rsn, ai_cor, gold_pos, gold_rsn, gold_cor):
    """
    Bind this stage's client, prompt style and retry policy into the reward's `judge_fn` slot.

    `reward_utils` holds no client of its own, so the endpoint and key stay declared in exactly one
    place per stage. Only reached in hybrid reward mode -- code-only never calls it.
    """
    return llm_as_a_judge(
        client, error_text, ai_pos, ai_rsn, ai_cor, gold_pos, gold_rsn, gold_cor,
        style=JUDGE_STYLE, model=JUDGE_MODEL, max_retries=JUDGE_MAX_RETRIES,
    )




def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        print(f"[data] {path} not found, using default")
        return default if default is not None else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[data] saved {len(obj)} record(s) -> {path}")


class HardSamplePool:
    """
    Dynamically maintained set of hard samples (paper Sec. III-E / III-F).

    A sample enters the pool when the model still cannot reach HARD_REWARD_THRESHOLD on it, and is
    automatically evicted once it does -- so the pool shrinks as the loop converges.
    """

    def __init__(self, samples=None):
        self.samples = {}
        for sample in (samples or []):
            self.samples[sample_key(sample)] = sample
        self.added_keys = set()          # samples added during the current iteration
        self.evicted_keys = set()        # samples evicted during the current iteration

    def __len__(self):
        return len(self.samples)

    def reset_iteration_tracking(self):
        self.added_keys = set()
        self.evicted_keys = set()

    def observe(self, sample: dict, group_max_reward: float):
        """Register one monitored sample with the best reward it achieved inside its GRPO group."""
        key = sample_key(sample)
        if group_max_reward < HARD_REWARD_THRESHOLD:
            if key not in self.samples:
                self.samples[key] = dict(sample)
                self.added_keys.add(key)
        else:
            if key in self.samples:
                del self.samples[key]
                self.evicted_keys.add(key)
                # keep added_keys a *net* count: a sample that is cached and then recovered
                # inside the same iteration must not be reported (or augmented) as new
                self.added_keys.discard(key)

    def newly_added(self):
        return [self.samples[k] for k in self.added_keys if k in self.samples]

    def to_list(self):
        return list(self.samples.values())


# ==================== GRPO reward function ====================
class ClosedLoopRewardManager:
    """
    Mixed reward (Eq. 7) plus real-time hard-sample monitoring.

    Group semantics: TRL hands the reward function one *generation batch*, which holds
    `num_generations` completions of each of `generation_batch_size // num_generations` unique
    prompts, with the dataset columns replicated to match. With the config used here
    (generation_batch_size 4 = num_generations 4) that is exactly ONE unique prompt per call, so
    every entry carries the same gold fields. The pool is therefore updated once per unique
    prompt, using the max reward inside its group: a sample counts as hard only if even the best
    of `num_generations` attempts stays below the threshold. The loop below derives the group size
    at runtime instead of assuming it, so raising `generation_batch_size` later cannot silently
    drop hard samples.
    """

    def __init__(self, pool: HardSamplePool):
        self.pool = pool
        self.reward_history = []
        self.baseline = 0.0
        self.alpha = 0.9

    def __call__(self, prompts, completions, **kwargs):
        responses = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        error_texts = kwargs.get("error_text", [])
        gold_reasonings = kwargs.get("reasoning", [])
        gold_positions = kwargs.get("error_position", [])
        gold_reasons = kwargs.get("specific_error_reason", [])
        gold_corrected = kwargs.get("corrected_content", [])

        if not error_texts:
            raise ValueError("reward function received no `error_text` column from the dataset")

        rewards = []
        for response, error_text, gold_pos, gold_spec, gold_corr in zip(
            responses, error_texts, gold_positions, gold_reasons, gold_corrected
        ):
            total, _, _ = compute_mixed_reward(
                response, error_text, gold_pos, gold_spec, gold_corr,
                judge_fn=judge_style_score, reward_mode=REWARD_MODE,
            )
            rewards.append(total)

        self._update_pool(
            error_texts, gold_reasonings, gold_positions, gold_reasons, gold_corrected, rewards
        )

        avg_reward = float(np.mean(rewards))
        self.baseline = self.alpha * self.baseline + (1 - self.alpha) * avg_reward
        self.reward_history.append(float(np.max(rewards)))
        print(f"step-{len(self.reward_history)} -Baseline: {self.baseline:.3f}, "
              f"Avg: {avg_reward:.2f}, Min: {np.min(rewards):.2f}, Max: {np.max(rewards):.2f}, "
              f"Pool: {len(self.pool)}")
        return rewards

    def _update_pool(self, error_texts, gold_reasonings, gold_positions, gold_reasons,
                     gold_corrected, rewards):
        n = len(error_texts)
        unique_prompts = len(set(error_texts))
        if unique_prompts == 0 or n % unique_prompts != 0:
            raise ValueError(
                f"cannot infer the GRPO group size: {n} completions over {unique_prompts} unique prompts"
            )
        group_size = n // unique_prompts

        for start in range(0, n, group_size):
            group_rewards = rewards[start:start + group_size]
            sample = {
                "error_text": error_texts[start],
                "reasoning": gold_reasonings[start],
                "error_position": gold_positions[start],
                "specific_error_reason": gold_reasons[start],
                "corrected_content": gold_corrected[start],
            }
            self.pool.observe(sample, max(group_rewards))


# ==================== Dataset construction ====================
def build_grpo_dataset(samples):
    """Convert CoT samples into the TRL prompt/column schema used by stage 4."""
    rows = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["error_text"]},
            ],
            "error_text": s["error_text"],
            "reasoning": s.get("reasoning", ""),
            "error_position": s.get("error_position", ""),
            "specific_error_reason": s.get("specific_error_reason", ""),
            "corrected_content": s.get("corrected_content", ""),
        }
        for s in samples
    ]
    return Dataset.from_list(rows).shuffle(seed=VAL_SEED)


def split_train_val(samples, val_ratio=VAL_RATIO, seed=VAL_SEED):
    """
    Fixed 9:1 split, identical across every iteration so the convergence signal is comparable.
    Augmented hard samples are appended to the training side only -- they are derived from
    training samples, so letting them reach the validation set would leak.
    """
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val, train = shuffled[:n_val], shuffled[n_val:]
    print(f"[split] train={len(train)}, val={len(val)} (augmented hard samples -> train only)")
    return train, val


# ==================== Hard-sample augmentation ====================
def augment_hard_samples(samples, mode=AUGMENT_MODE, variant_num=AUGMENT_VARIANT_NUM):
    """
    Generate `variant_num` error-injected variants per hard sample (paper Sec. III-E, 1:5 ratio).

    Both back-ends reuse the four error types from the paper (clause citation error, test
    parameter deviation, failure criterion misuse, cross-chapter inconsistency) and always
    rebuild from `corrected_content`, which stays the ground truth -- so re-augmenting an
    already-augmented sample cannot compound annotation drift.
    """
    if not samples:
        return []

    if mode == "llm":
        from LLM_augmented_hardsamples import augment_single_sample as _augment  # has its own client
        produced = []
        for i, sample in enumerate(samples):
            variants = _augment(sample)
            produced.extend(variants)
            print(f"  [augment:llm] {i + 1}/{len(samples)} -> {len(variants)} variants")
        return produced

    from Rules_augmented_hardsamples import augment_single_sample as _augment
    produced = []
    for i, sample in enumerate(samples):
        variants = _augment(sample, variant_num=variant_num)
        produced.extend(variants)
        print(f"  [augment:rules] {i + 1}/{len(samples)} -> {len(variants)} variants")
    return produced


# ==================== Validation ====================
@torch.no_grad()
def generate_completion(model, tokenizer, error_text, max_new_tokens=1280):
    """Greedy decoding: the convergence criterion must not be dominated by sampling noise."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": error_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def evaluate_validation(model, tokenizer, val_samples):
    """Mean mixed reward on the held-out validation split -- the convergence signal."""
    was_training = model.training
    use_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = True

    scores, code_scores, judge_scores = [], [], []
    try:
        for i, sample in enumerate(val_samples):
            response = generate_completion(model, tokenizer, sample["error_text"])
            total, code, judge = compute_mixed_reward(
                response, sample["error_text"], sample["error_position"],
                sample["specific_error_reason"], sample["corrected_content"],
                judge_fn=judge_style_score, reward_mode=REWARD_MODE,
            )
            scores.append(total)
            code_scores.append(code)
            judge_scores.append(judge)
            print(f"  [val] {i + 1}/{len(val_samples)} reward={total:.3f} "
                  f"(code={code:.2f}, judge={judge:.2f})")
    finally:
        model.config.use_cache = use_cache
        if was_training:
            model.train()

    return {
        "mean_reward": float(np.mean(scores)),
        "mean_code_reward": float(np.mean(code_scores)),
        "mean_judge_score": float(np.mean(judge_scores)),
    }


# ==================== Convergence ====================
def check_convergence(reward_history, pool_size, total_size):
    """
    Terminate when both paper conditions hold:
      (a) validation mixed reward fluctuated < CONVERGENCE_TOL over the last
          CONVERGENCE_PATIENCE consecutive rounds, and
      (b) hard samples are < HARD_POOL_RATIO of the total sample pool.
    """
    pool_ratio = pool_size / total_size if total_size else 1.0
    if pool_ratio >= HARD_POOL_RATIO:
        return False, (f"hard-sample ratio {pool_ratio:.2%} >= {HARD_POOL_RATIO:.0%} "
                       f"({pool_size}/{total_size})")

    if len(reward_history) < CONVERGENCE_PATIENCE + 1:
        return False, (f"only {len(reward_history)} round(s); need >= {CONVERGENCE_PATIENCE + 1} "
                       f"to test {CONVERGENCE_PATIENCE} consecutive fluctuations")

    prev = reward_history[-CONVERGENCE_PATIENCE - 1:-1]
    curr = reward_history[-CONVERGENCE_PATIENCE:]
    fluctuations = [
        abs(c - p) / abs(p) if p else float("inf")
        for p, c in zip(prev, curr)
    ]
    shown = ", ".join(f"{f:.2%}" for f in fluctuations)
    if all(f < CONVERGENCE_TOL for f in fluctuations):
        return True, (f"validation reward fluctuation [{shown}] < {CONVERGENCE_TOL:.0%} for "
                      f"{CONVERGENCE_PATIENCE} consecutive rounds; hard-sample ratio "
                      f"{pool_ratio:.2%} < {HARD_POOL_RATIO:.0%}")
    return False, f"validation reward fluctuation [{shown}] exceeds {CONVERGENCE_TOL:.0%}"


# ==================== GRPO configuration ====================
def build_grpo_config(iteration: int) -> GRPOConfig:
    """Same optimisation setup as the content-GRPO stage, so the loop only changes the data."""
    return GRPOConfig(
        output_dir=f"{OUTPUT_DIR}/iter_{iteration}",
        run_name=f"Qwen-Correction-iterative-loop-iter{iteration}",
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=4,
        generation_batch_size=4,
        max_prompt_length=896,
        max_completion_length=1280,
        num_train_epochs=EPOCHS_PER_ITERATION,
        save_steps=100,
        max_grad_norm=1.0,
        save_total_limit=2,
        report_to="none",
    )


# ==================== Main loop ====================
def main():
    ablation_variant = announce("6model_iteration")
    seed_augmented = augmented_seed_enabled()

    report = {
        "config": {
            "model_path": MODEL_PATH,
            "base_data": BASE_DATA_PATH,
            "max_iterations": MAX_ITERATIONS,
            "convergence_tol": CONVERGENCE_TOL,
            "convergence_patience": CONVERGENCE_PATIENCE,
            "hard_pool_ratio": HARD_POOL_RATIO,
            "hard_reward_threshold": HARD_REWARD_THRESHOLD,
            "epochs_per_iteration": EPOCHS_PER_ITERATION,
            "augment_mode": AUGMENT_MODE,
            "augment_variant_num": AUGMENT_VARIANT_NUM,
            "judge_model": JUDGE_MODEL,
            "judge_style": JUDGE_STYLE,
            # In code-only mode the convergence signal is the code reward, not the mixed reward,
            # because there is no judge term to mix in. Recorded so a report is self-describing.
            "reward_mode": REWARD_MODE,
            "ablation_variant": ablation_variant,
            "augmented_seed_enabled": seed_augmented,
        },
        "iterations": [],
        "converged": False,
        "stop_reason": "",
    }

    pool = HardSamplePool(load_json(POOL_PATH, []))
    augmented = load_json(AUGMENTED_PATH, None)
    if augmented is None:
        # First run: seed the accumulated augmented set with stage 5's output. The `no-HSO`
        # ablation suppresses this, because it removes the module that produced that file.
        augmented = load_json(AUGMENTED_SEED_PATH, []) if seed_augmented else []
        if seed_augmented:
            print(f"[data] seeded accumulated augmented set with {len(augmented)} stage-5 samples")
        else:
            print("[data] augmented-seed disabled for this variant: starting from an empty "
                  "accumulated set")

    base_samples = load_json(BASE_DATA_PATH, [])
    if not base_samples:
        raise FileNotFoundError(f"no base CoT samples loaded from {BASE_DATA_PATH}")
    train_samples, val_samples = split_train_val(base_samples)

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    reward_history = []
    trainer = None

    try:
        for iteration in range(1, MAX_ITERATIONS + 1):
            print("\n" + "=" * 70)
            print(f"ITERATION {iteration}/{MAX_ITERATIONS}")
            print("=" * 70)

            full_train = train_samples + augmented
            total_pool_size = len(full_train)
            print(f"[data] full training set: {len(train_samples)} original + "
                  f"{len(augmented)} augmented hard = {len(full_train)}")
            print(f"[data] hard-sample pool carried in: {len(pool)}")

            # ---- 1. Iterative training loop: GRPO on the full dataset ----
            reward_manager = ClosedLoopRewardManager(pool)
            reward_manager.pool.reset_iteration_tracking()
            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=reward_manager,
                args=build_grpo_config(iteration),
                train_dataset=build_grpo_dataset(full_train),
            )
            print(f"[train] starting GRPO for iteration {iteration}...")
            trainer.train()
            print(f"[train] iteration {iteration} finished")

            # ---- 2. Bottleneck diagnosis: refresh the hard-sample pool ----
            n_added, n_evicted = len(pool.added_keys), len(pool.evicted_keys)
            print(f"[pool] +{n_added} new hard samples, -{n_evicted} recovered, total={len(pool)}")
            save_json(pool.to_list(), POOL_PATH)

            # ---- 3. Validation mixed reward (convergence signal) ----
            metrics = evaluate_validation(trainer.model, tokenizer, val_samples)
            reward_history.append(metrics["mean_reward"])
            print(f"[val] iteration {iteration}: {metrics}")

            # ---- 4. Data augmentation on newly discovered hard samples ----
            newly_hard = pool.newly_added()
            new_variants = augment_hard_samples(newly_hard)
            augmented = augmented + new_variants
            save_json(augmented, AUGMENTED_PATH)
            print(f"[augment] {len(newly_hard)} hard sample(s) -> {len(new_variants)} variants; "
                  f"accumulated augmented set = {len(augmented)}")

            # ---- 5. Persist this iteration's model ----
            iter_dir = f"{OUTPUT_DIR}/iter_{iteration}"
            trainer.save_model(iter_dir)
            tokenizer.save_pretrained(iter_dir)
            print(f"[model] saved -> {iter_dir}")

            # ---- 6. Convergence check ----
            converged, reason = check_convergence(reward_history, len(pool), total_pool_size)
            report["iterations"].append({
                "iteration": iteration,
                "train_set_size": len(full_train),
                "validation": metrics,
                "hard_pool_size": len(pool),
                "hard_pool_ratio": len(pool) / total_pool_size if total_pool_size else None,
                "new_hard_samples": n_added,
                "evicted_hard_samples": n_evicted,
                "augmented_variants_added": len(new_variants),
                "accumulated_augmented_size": len(augmented),
                "converged": converged,
                "convergence_reason": reason,
                "model_dir": iter_dir,
            })
            print(f"[convergence] {'CONVERGED' if converged else 'continue'}: {reason}")

            if converged:
                report["converged"] = True
                report["stop_reason"] = f"converged at iteration {iteration}: {reason}"
                break
        else:
            report["stop_reason"] = (f"reached MAX_ITERATIONS={MAX_ITERATIONS} without meeting both "
                                     f"convergence conditions")
    except Exception:
        print("Iteration error details:")
        traceback.print_exc()
        report["stop_reason"] = "aborted by exception"
        raise
    finally:
        # The loop is long-running: always land the pool, the augmented set and the report,
        # even if training raised or the process was interrupted.
        try:
            save_json(pool.to_list(), POOL_PATH)
            save_json(augmented, AUGMENTED_PATH)
            save_json(report, REPORT_PATH)
            if trainer is not None:
                final_dir = f"{OUTPUT_DIR}/final"
                trainer.save_model(final_dir)
                tokenizer.save_pretrained(final_dir)
                print(f"[model] final model saved -> {final_dir}")
        except Exception:
            traceback.print_exc()
        torch.cuda.empty_cache()
        print("=" * 70)
        print(f"Closed-loop iteration finished. converged={report['converged']}")
        print(f"stop_reason: {report['stop_reason']}")
        print(f"validation reward history: {[round(r, 4) for r in reward_history]}")
        print("=" * 70)
        print("Process finished")


if __name__ == "__main__":
    main()

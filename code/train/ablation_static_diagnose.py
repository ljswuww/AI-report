# -*- coding: utf-8 -*-
"""
Static hard-sample diagnosis -- the substitution the `no-T-GRPO` ablation needs.

In the full pipeline, hard samples are diagnosed from the rewards observed while the content-GRPO
stage trains: a sample enters the pool when even the best of its `num_generations` rollouts stays
below the threshold. Removing both GRPO stages removes that source, so the pipeline would have no
pool to augment for stage 5 and nothing to seed stage 6 with. Leaving both empty would confound the
ablation with a change to HSO's and CIO's starting state, so this script produces the pool by a
different route instead.

What it does: run the variant's current checkpoint greedily over the RL training split, score each
sample with the same reward the GRPO stages would have used, and admit the samples that fall below
the same threshold. It then writes the admitted samples in exactly the dump format stage 4 writes,
so everything downstream -- `txt2json.py`, the 1:5 augmentation, stage 5, stage 6 -- is unchanged.

What differs is only the origin of the scores. A greedy pass over a fixed checkpoint replaces
grouped rollouts of a policy mid-update, so the pool reflects the weaknesses of a cold-start model
rather than those of a partially optimised one. This is deliberate and must be reported alongside
the variant's result; it is not equivalent to what stage 4 would have diagnosed.

Run from `code/train`:

    python ablation_static_diagnose.py \
        --model-path ../model/output/ablation/no-T-GRPO/2Cold_start_output \
        --out        ../../data/train/Hard_samples/ablation/no-T-GRPO/hard_samples.txt

Then the usual path: `txt2json.py` -> `Rules_augmented_hardsamples.py` -> stage 5.
`run_ablation.py` drives all of that for the variant.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

UTILS_DIR = Path(__file__).resolve().parent / "utils"
sys.path.insert(0, str(UTILS_DIR))

from ablation_config import announce, reward_mode  # noqa: E402
from prompts import SYSTEM_PROMPT  # noqa: E402
from reward_utils import (  # noqa: E402
    DEFAULT_HARD_REWARD_THRESHOLD, compute_mixed_reward, sample_key, structured_data_to_string,
)

# Same dump encoding stage 4 uses, and the same one txt2json.py parses
RECORD_SEPARATOR = "*|||*\n"

DEFAULT_DATA_PATH = "../../data/train/RL_data/GRPO_content/RL_content_samples.json"
DEFAULT_OUT_PATH = "../../data/train/Hard_samples/hard_samples.txt"

# Judge settings for the hybrid reward. Same endpoint and backbone as the training stages.
JUDGE_MODEL = "deepseek-chat"
JUDGE_STYLE = "training"
JUDGE_MAX_RETRIES = 3

# Greedy decoding, matching the validation pass in stage 6 so the two produce comparable scores
MAX_NEW_TOKENS = 1280

# Gold fields carried by each sample of the RL split
GOLD_FIELDS = ("error_text", "reasoning", "error_position", "specific_error_reason",
               "corrected_content")


def build_judge(style):
    """
    Build the judge callable for the hybrid reward, or None in code-only mode.

    Reused verbatim from the training stages so a statically diagnosed pool is selected by the same
    reward function the GRPO stages use. Returning None in code-only mode is what keeps the
    `no-HR` variant from reaching the judge through this path.
    """
    if reward_mode() == "code_only":
        print("[judge] code-only reward mode: no judge will be consulted")
        return None

    from openai import OpenAI

    client = OpenAI(api_key="xx-xxxxxxxxxxxxxxxxxxxxx", base_url="https://api.deepseek.com")

    def judge_fn(error_text, ai_pos, ai_rsn, ai_cor, gold_pos, gold_rsn, gold_cor):
        from reward_utils import llm_as_a_judge
        return llm_as_a_judge(
            client, error_text, ai_pos, ai_rsn, ai_cor, gold_pos, gold_rsn, gold_cor,
            style=style, model=JUDGE_MODEL, max_retries=JUDGE_MAX_RETRIES,
        )

    return judge_fn


def load_samples(path, max_samples=None):
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if max_samples:
        samples = samples[:max_samples]
    missing = [i for i, s in enumerate(samples) if s.get("error_text") is None]
    if missing:
        raise ValueError(f"{len(missing)} sample(s) lack an `error_text` field, e.g. index {missing[0]}")
    return samples


@torch.no_grad()
def generate_completion(model, tokenizer, error_text, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy decoding, identical to the validation pass in stage 6."""
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


def diagnose(model, tokenizer, samples, threshold, judge_fn):
    """
    Score every sample with the reward the GRPO stages use, and return the ones that fall short.

    A sample is admitted when its score is below the threshold -- the same rule stage 4 applies to
    the maximum reward inside a rollout group. The group dimension is absent here by construction:
    decoding is greedy, so each sample yields exactly one completion rather than `num_generations`
    of them, and there is no group maximum to take.
    """
    admitted = {}            # key -> dump string, so a duplicate cannot be counted twice
    scores = []

    for idx, sample in enumerate(samples):
        response = generate_completion(model, tokenizer, sample["error_text"])
        total, code, judge = compute_mixed_reward(
            response,
            sample["error_text"],
            sample.get("error_position", ""),
            sample.get("specific_error_reason", ""),
            sample.get("corrected_content", ""),
            judge_fn=judge_fn,
            reward_mode=reward_mode(),
        )
        scores.append(total)

        if total < threshold:
            dump = structured_data_to_string(*[str(sample.get(f, "")) for f in GOLD_FIELDS])
            admitted[sample_key(sample)] = dump

        print(f"[{idx + 1}/{len(samples)}] reward={total:6.3f} (code={code:5.2f}, judge={judge:.2f}) "
              f"hard={total < threshold} pool={len(admitted)}")

    return admitted, scores


def write_dump(dump_strings, out_path):
    """
    Write the pool in the encoding stage 4 writes and `txt2json.py` parses.

    The file is written even when the pool is empty: an absent file and an empty pool mean different
    things downstream, and stage 5 fails on a missing file rather than on an empty one.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(RECORD_SEPARATOR.join(dump_strings))
    return len(dump_strings)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Static hard-sample diagnosis, the substitute used by the `no-T-GRPO` ablation.")
    parser.add_argument("--model-path", default=None,
                        help="checkpoint to diagnose with; defaults to ABLATION_6IT_MODEL_PATH, "
                             "else the stage-6 default")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH,
                        help="RL split to diagnose over (default: %(default)s)")
    parser.add_argument("--out", default=None,
                        help=f"dump path (default: {DEFAULT_OUT_PATH}, or ABLATION_4GRPO_HARD_SAMPLES_TXT)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_HARD_REWARD_THRESHOLD,
                        help="hard-sample threshold on the 0-10 reward scale (default: %(default)s)")
    parser.add_argument("--judge-style", default=JUDGE_STYLE, choices=("training", "eca"),
                        help="judge prompt to score with (default: %(default)s)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="diagnose only the first N samples, for a smoke test")
    parser.add_argument("--device-map", default="auto",
                        help="transformers device_map for the diagnosed checkpoint")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    variant = announce("ablation_static_diagnose")

    model_path = args.model_path or os.environ.get("ABLATION_6IT_MODEL_PATH")
    if not model_path:
        raise SystemExit("no checkpoint to diagnose with: pass --model-path or set "
                         "ABLATION_6IT_MODEL_PATH")
    out_path = args.out or os.environ.get("ABLATION_4GRPO_HARD_SAMPLES_TXT") or DEFAULT_OUT_PATH

    print(f"Static diagnosis: variant={variant or 'full'} model={model_path}")
    print(f"  data      : {args.data}")
    print(f"  out       : {out_path}")
    print(f"  threshold : {args.threshold} (reward mode: {reward_mode()})")

    judge_fn = build_judge(args.judge_style)

    samples = load_samples(args.data, args.max_samples)
    print(f"Loaded {len(samples)} samples to diagnose")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=args.device_map,
        local_files_only=True, trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = True

    try:
        admitted, scores = diagnose(model, tokenizer, samples, args.threshold, judge_fn)
    except Exception:
        print("Static diagnosis failed:")
        traceback.print_exc()
        raise
    finally:
        torch.cuda.empty_cache()

    n_written = write_dump(list(admitted.values()), out_path)
    ratio = n_written / len(samples) if samples else 0.0
    print(f"\nhard samples admitted: {n_written}/{len(samples)} ({ratio:.1%})")
    print(f"mean reward: {sum(scores) / len(scores):.4f}" if scores else "no samples scored")
    print(f"dump written -> {out_path}")

    if not n_written:
        # Not fatal, but it changes what the variant measures: with an empty pool stage 5 trains on
        # nothing and this variant stops being "no T-GRPO" and becomes "no T-GRPO and no HSO".
        print("\nWARNING: the diagnosed pool is empty. Stage 5 would have nothing to train on, so "
              "this variant would silently also remove HSO. Check the checkpoint and the threshold "
              "before using the result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

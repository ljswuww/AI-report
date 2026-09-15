"""
ECA (Error-Correction Accuracy) evaluation for the AEC-Q test report auditing model.
"""

import os
import re
import sys
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

# ==================== Configuration ====================
# Paths are overridable via ABLATION_<KEY> so an ablation variant's checkpoint can be evaluated
# without editing this file; with nothing set the defaults below apply.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from ablation_config import override  # noqa: E402

# Models from all stages can undergo ECA testing.
# Model under evaluation. `final` is the converged artifact of the closed-loop iteration stage;
# point this at `6Iteration_output/iter_N` or at an earlier stage for ablation runs.
MODEL_PATH = override("test.MODEL_PATH", "../model/output/6Iteration_output/final")
DATA_PATH = override("test.DATA_PATH", "../../data/test/test_samples.json")
# Per-variant result directories, so one variant's records cannot overwrite another's
RESULT_DIR = override("test.RESULT_DIR", "./eca_results")
RESULT_JSONL = f"{RESULT_DIR}/eca_per_sample.jsonl"
SUMMARY_PATH = f"{RESULT_DIR}/eca_summary.json"

# Set to e.g. 20 for a quick smoke test
MAX_SAMPLES = None

# Generation. Greedy by default: the reported ECA must be reproducible and free of sampling noise.
MAX_NEW_TOKENS = 1280
DO_SAMPLE = False
TEMPERATURE = 0.7

# Judge backbone for the ECA stage. The paper requires this to DIFFER from the GRPO-training
# judge, which is DeepSeek (`deepseek-chat`) in 4grpo_content.py -- Qwen3.8-Max here.
JUDGE_MODEL = "qwen3.8-max"
JUDGE_MAX_RETRIES = 3

# 33 of the 200 test samples are compliant reports that contain NO error: error_text equals
# corrected_content and the gold location/reason are empty strings. The ECA prompt has no clause
# for that case, so an empty reference confuses the judge into scoring 0.0. When this flag is on,
# empty gold fields are rendered as an explicit "no error" phrase. Either way those samples are
# reported as their own cohort in the summary.
FILL_EMPTY_GOLD = True
NO_ERROR_PHRASE = "none (this text is compliant and contains no error)"

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

# ECA judge prompt -- all-or-nothing, {0.0, 1.0}.
# Mirrors prompts/LLM-as-a-judge prompt used during computing ECA.txt verbatim.
ECA_JUDGE_PROMPT = """You are a rigorous yet flexible evaluation expert for error‑correction models in the AEC‑Q automotive chip testing domain. Evaluate model output and give final score per rules below.

## Evaluation Rules

Step‑wise all‑or‑nothing scoring. Check AI output against gold standard in order:
Step1: Check "Error Location". Proceed to Step 2 for matching location; otherwise score 0.0, stop.
Step2: Check "Error Reason". Proceed to Step 3 for equivalent reason; otherwise score 0.0, stop.
Step3: Check "Corrected Text". Score 1.0 for consistent correction with no new errors; otherwise score 0.0, stop.

## Equivalence Criteria

- **Error Location**: Different wording allowed, must point to identical text span or semantic unit.
- **Error Reason**: Different diction & sentence structure allowed, must retain the core error point.
- **Corrected Text**: Synonym substitution and rephrasing allowed. Revised professional content must be equivalent to gold standard and free of new errors.

## Input Information

- Original erroneous text: {error_text}
- AI predicted error location: {AI_pos}
- AI predicted error reason: {AI_rsn}
- AI corrected text: {AI_cor}
- Gold‑standard error location: {gold_pos}
- Gold‑standard error reason: {gold_rsn}
- Gold‑standard corrected text: {gold_cor}

## Output Format

Output strictly following the structure below:<reasoning>
[Write step‑by‑step reasoning, state comparison basis and judgment result at each step.]</reasoning><score>
[Final score, valid values: 0.0 / 1.0]</score>
"""

# The four structured-CoT elements of the auditing output paradigm
COT_FIELDS = ("reasoning", "error_position", "specific_error_reason", "corrected_text")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ECA judge backend: Alibaba Cloud Model Studio (Bailian / DashScope), OpenAI-compatible mode.
# Credentials come from the environment so no key is ever committed:
#   export DASHSCOPE_API_KEY=sk-xxx
#   export DASHSCOPE_BASE_URL=https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
# The workspace-scoped MaaS endpoint above is what the paper's runs used; the public compatible
# endpoint also works and is the default here.
DEFAULT_JUDGE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_judge_client = None


def get_judge_client():
    """
    Lazily build the judge client.

    Kept out of import time so that the module's pure helpers stay importable (and testable)
    without credentials, while `main()` still fails early and loudly on a missing key instead of
    after burning a model generation.
    """
    global _judge_client
    if _judge_client is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not set. Export it before running the ECA evaluation:\n"
                "  export DASHSCOPE_API_KEY=sk-xxx"
            )
        base_url = os.getenv("DASHSCOPE_BASE_URL", DEFAULT_JUDGE_BASE_URL)
        print(f"ECA judge backend: model={JUDGE_MODEL} base_url={base_url}")
        _judge_client = OpenAI(api_key=api_key, base_url=base_url)
    return _judge_client


# ==================== Field extraction ====================
def extract_field(text: str, field: str) -> str:
    """Pull one <field>...</field> block out of a model completion; "not find" when absent."""
    start_tag = f"<{field}>"
    end_tag = f"</{field}>"
    if start_tag not in text or end_tag not in text:
        return "not find"
    try:
        start_idx = text.index(start_tag) + len(start_tag)
        end_idx = text.index(end_tag, start_idx)
        return text[start_idx:end_idx].strip()
    except ValueError:
        return "not find"


def parse_prediction(response: str) -> dict:
    return {f: extract_field(response, f) for f in COT_FIELDS}


def has_full_format(prediction: dict) -> bool:
    """All four tags present with content -- otherwise the output is not a valid audit reply."""
    return all(v != "not find" for v in prediction.values())


# ==================== ECA judge ====================
VALID_ECA_SCORES = ("0.0", "1.0")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _to_verdict(token: str):
    """
    Map a numeric token onto the binary verdict, or None when it is not a valid one.

    Compared numerically rather than as strings so a judge writing "1" or "1.00" instead of
    "1.0" is not silently scored 0.0 -- swapping the judge backbone makes that a real risk.
    Anything else (e.g. an invented 0.5) is rejected and the caller falls back to 0.0.
    """
    try:
        value = float(token)
    except ValueError:
        return None
    if abs(value) < 1e-9:
        return 0.0
    if abs(value - 1.0) < 1e-9:
        return 1.0
    return None


def extract_eca_score(text: str) -> float:
    """
    Read the all-or-nothing verdict out of a judge reply.

    The prompt asks for a <score>...</score> block, so an end-anchored search for a bare trailing
    number can never match a compliant reply. Search the tag first, fall back to a bare number.
    """
    text = (text or "").strip()

    block = re.search(r"<score\s*>(.*?)</score>", text, re.S)
    if block:
        tokens = _NUMBER_RE.findall(block.group(1))
        # A block holding a single number is unambiguous -- take it, whatever its spelling
        if len(tokens) == 1:
            verdict = _to_verdict(tokens[0])
            return verdict if verdict is not None else 0.0
        # Several numbers: the block is not a clean verdict, so only an explicitly written
        # 0.0 / 1.0 counts. A judge that merely echoes the prompt's value list therefore lands
        # on 0.0, the safe failure mode.
        for token in tokens:
            if token in VALID_ECA_SCORES:
                return float(token)
        return 0.0

    tail = re.search(r"(\d+(?:\.\d+)?)\s*$", text)
    if tail:
        verdict = _to_verdict(tail.group(1))
        if verdict is not None:
            return verdict
    return 0.0


def render_gold(sample: dict):
    """Gold reference fields for the judge, with the no-error cohort filled in explicitly."""
    gold_pos = sample.get("error_position", "").strip()
    gold_rsn = sample.get("specific_error_reason", "").strip()
    if FILL_EMPTY_GOLD:
        gold_pos = gold_pos or NO_ERROR_PHRASE
        gold_rsn = gold_rsn or NO_ERROR_PHRASE
    return gold_pos, gold_rsn


def llm_as_a_judge_eca(error_text, ai_pos, ai_rsn, ai_cor, gold_pos, gold_rsn, gold_cor,
                       model=JUDGE_MODEL, max_retries=JUDGE_MAX_RETRIES):
    """
    Fine-grained semantic verification of the pre-matched CoT elements.

    Performs NO full AEC-Q compliance audit, no error localisation and no clause lookup -- the
    judge only compares what the model produced against the gold annotation. That keeps its task
    load low and avoids misjudgement from asking it to master the domain specs.

    Returns (score, raw_reply).
    """
    prompt = ECA_JUDGE_PROMPT.format(
        error_text=error_text, AI_pos=ai_pos, AI_rsn=ai_rsn, AI_cor=ai_cor,
        gold_pos=gold_pos, gold_rsn=gold_rsn, gold_cor=gold_cor,
    )
    judge_client = get_judge_client()
    for attempt in range(max_retries):
        try:
            response = judge_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
            raw = response.choices[0].message.content
            return extract_eca_score(raw), raw
        except Exception as exc:
            print(f"  [judge] attempt {attempt + 1}/{max_retries} failed: {exc}")
            time.sleep(2 ** attempt)
    print("  [judge] all retries exhausted, scoring 0.0")
    return 0.0, ""


# ==================== Inference ====================
def generate_completion(model, tokenizer, error_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": error_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE if DO_SAMPLE else None,
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


# ==================== Aggregation ====================
def summarize(records):
    """ECA overall, plus the breakdowns that separate 'format broke' from 'content wrong'."""
    n = len(records)
    scores = np.array([r["eca"] for r in records], dtype=float)
    format_ok = np.array([r["format_ok"] for r in records], dtype=bool)
    no_error = np.array([r["is_no_error_sample"] for r in records], dtype=bool)

    def cohort(mask, label):
        count = int(mask.sum())
        if count == 0:
            return {"label": label, "n": 0, "eca": None, "format_ok_rate": None}
        return {
            "label": label,
            "n": count,
            "eca": float(scores[mask].mean()),
            "format_ok_rate": float(format_ok[mask].mean()),
        }

    return {
        "num_samples": n,
        "eca": float(scores.mean()),
        "num_correct": int((scores == 1.0).sum()),
        "num_incorrect": int((scores == 0.0).sum()),
        "format_compliance_rate": float(format_ok.mean()),
        # ECA restricted to well-formed outputs: separates a formatting failure from a genuine
        # auditing failure
        "eca_on_format_compliant": (float(scores[format_ok].mean()) if format_ok.any() else None),
        "num_format_failures": int((~format_ok).sum()),
        "cohorts": {
            "with_error": cohort(~no_error, "gold contains an error"),
            "no_error": cohort(no_error, "compliant report, correct answer is 'none'"),
        },
        "judge_model": JUDGE_MODEL,
        "fill_empty_gold": FILL_EMPTY_GOLD,
        "generation": {"do_sample": DO_SAMPLE, "max_new_tokens": MAX_NEW_TOKENS},
    }


def print_summary(summary):
    print("\n" + "=" * 70)
    print("ECA EVALUATION SUMMARY")
    print("=" * 70)
    print(f"samples                : {summary['num_samples']}")
    print(f"ECA (all-or-nothing)   : {summary['eca']:.4f}  "
          f"({summary['num_correct']}/{summary['num_samples']} correct)")
    print(f"format compliance rate : {summary['format_compliance_rate']:.4f}  "
          f"({summary['num_format_failures']} malformed outputs -> automatic 0)")
    if summary["eca_on_format_compliant"] is not None:
        print(f"ECA | format-compliant : {summary['eca_on_format_compliant']:.4f}")
    for name, c in summary["cohorts"].items():
        if c["n"]:
            print(f"  cohort {name:<11}: n={c['n']:<4} ECA={c['eca']:.4f} "
                  f"format_ok={c['format_ok_rate']:.4f}   [{c['label']}]")
    print(f"judge backbone         : {summary['judge_model']}")
    print("=" * 70)


# ==================== Main ====================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Fail fast on a missing judge credential, before the multi-minute model load
    get_judge_client()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if MAX_SAMPLES:
        samples = samples[:MAX_SAMPLES]
    print(f"Loaded {len(samples)} test samples from {DATA_PATH}")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto",
        local_files_only=True, trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = True

    records = []
    try:
        for idx, sample in enumerate(samples):
            error_text = sample["error_text"]
            gold_pos, gold_rsn = render_gold(sample)
            gold_cor = sample.get("corrected_content", "")
            is_no_error = not sample.get("error_position", "").strip()

            response = generate_completion(model, tokenizer, error_text)
            prediction = parse_prediction(response)
            format_ok = has_full_format(prediction)

            eca, judge_raw = llm_as_a_judge_eca(
                error_text,
                prediction["error_position"],
                prediction["specific_error_reason"],
                prediction["corrected_text"],
                gold_pos, gold_rsn, gold_cor,
            )

            record = {
                "idx": idx,
                "is_no_error_sample": is_no_error,
                "format_ok": format_ok,
                "eca": eca,
                "error_text": error_text,
                "gold_error_position": sample.get("error_position", ""),
                "gold_specific_error_reason": sample.get("specific_error_reason", ""),
                "gold_corrected_content": gold_cor,
                "model_response": response,
                "pred_error_position": prediction["error_position"],
                "pred_reasoning": prediction["reasoning"],
                "pred_specific_error_reason": prediction["specific_error_reason"],
                "pred_corrected_text": prediction["corrected_text"],
                "judge_raw_reply": judge_raw,
            }
            records.append(record)

            # Write through after every sample so a crash or a killed job keeps its progress
            with open(RESULT_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            running = float(np.mean([r["eca"] for r in records]))
            print(f"[{idx + 1}/{len(samples)}] eca={eca:.1f} format_ok={format_ok} "
                  f"no_error_gold={is_no_error} running_ECA={running:.4f}")
    except Exception:
        print("Evaluation error details:")
        traceback.print_exc()
    finally:
        if records:
            summary = summarize(records)
            with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print_summary(summary)
            print(f"\nper-sample records -> {RESULT_JSONL}")
            print(f"summary            -> {SUMMARY_PATH}")
        print("Process finished")


if __name__ == "__main__":
    main()

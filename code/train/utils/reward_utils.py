# -*- coding: utf-8 -*-
"""
Canonical training reward: 3-gram F1 (Eqs. 1-6), the format eligibility gate, the LLM-as-a-judge
step-wise score, and the hybrid aggregate (Eq. 7).

This module exists so the reward has exactly one definition. The static-diagnosis path used by the
`no-T-GRPO` ablation has to score samples with the *same* reward the GRPO stages would have used,
otherwise the ablated pool is selected by a different criterion than the full pipeline's and the
comparison measures the wrong thing. Importing the same functions is what guarantees that.

Two deliberate differences from a training-script local copy:

- `llm_as_a_judge` takes the client as an argument instead of reading a module global. Credentials
  stay where the caller already keeps them (`4grpo_content.py` and `6model_iteration.py` each build
  their own client), so nothing about key handling changes.
- `compute_mixed_reward` takes a `reward_mode`, so the code-only branch the `no-HR` ablation needs
  lives next to the hybrid one and cannot drift away from it.

Note on naming: `extract_xml_score` here returns a `float`. `4grpo_content.py` has a function of the
same name that returns a `str` and is left as it is; the two are not interchangeable.
"""

import re
import time
from collections import Counter

# ==================== 3-gram F1 (Eqs. 1-6) ====================
def prepare_text(text):
    """Insert spaces around every CJK character so the text can be split on whitespace."""
    return re.sub(r"([一-鿿])", r" \1 ", text)


def get_ngrams(words, n, deduplicate=False):
    """Build the n-gram list of a word sequence (Eqs. 1-3)."""
    if len(words) < n:
        return []
    ngrams = ["-".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if deduplicate:
        ngrams = list(set(ngrams))
    return ngrams


def calculate_f1(pred_grams, gold_grams):
    """Harmonic mean of multiset precision and recall (Eqs. 4-6)."""
    pred_counter = Counter(pred_grams)
    gold_counter = Counter(gold_grams)
    common = sum((pred_counter & gold_counter).values())
    precision = common / len(pred_grams) if pred_grams else 0
    recall = common / len(gold_grams) if gold_grams else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_3gram_f1(pred: str, gold: str) -> float:
    """
    Order-3 character n-gram F1 between a prediction and the gold text.

    Multiset, not set: repeated n-grams count repeatedly, so a correction that repeats a phrase it
    should not is penalised. Falls back to exact match when both sides are shorter than 3 tokens,
    where the n-gram construction has nothing to compare.
    """
    pred_words = prepare_text(pred).split()
    gold_words = prepare_text(gold).split()
    if not pred_words or not gold_words:
        return 0.0
    if len(pred_words) < 3 and len(gold_words) < 3:
        return 1.0 if pred == gold else 0.0
    return calculate_f1(
        get_ngrams(pred_words, 3, deduplicate=False),
        get_ngrams(gold_words, 3, deduplicate=False),
    )


# ==================== Response parsing ====================
COT_FIELDS = ("reasoning", "error_position", "specific_error_reason", "corrected_text")


def extract_field(text: str, field: str) -> str:
    """
    Return the content of `<field>...</field>`, or "not find" when absent.

    "not find" is a sentinel rather than an empty string because the two mean different things:
    an empty tag is a present-but-empty answer, a missing tag is a broken output contract. Only the
    latter should zero the reward.
    """
    start_tag, end_tag = f"<{field}>", f"</{field}>"
    if start_tag not in text or end_tag not in text:
        return "not find"
    try:
        start_idx = text.index(start_tag) + len(start_tag)
        end_idx = text.index(end_tag, start_idx)
        return text[start_idx:end_idx].strip()
    except ValueError:
        return "not find"


def parse_response(response: str) -> dict:
    """Split a model completion into the four structured-CoT elements."""
    return {f: extract_field(response, f) for f in COT_FIELDS}


def has_full_format(parsed: dict) -> bool:
    """All four tags present with content -> eligible for content scoring."""
    return all(v != "not find" for v in parsed.values())


# ==================== LLM-as-a-judge ====================
# Training-style judge: step-wise CUMULATIVE scoring, {0.0, 0.3, 0.7, 1.0}.
JUDGE_PROMPT_TRAINING = """You are a rigorous yet flexible evaluation expert for error-correction models in the AEC-Q automotive chip testing domain. Evaluate model output and give final score per rules below.

## Evaluation Rules
Step-wise cumulative scoring. Check AI output against gold standard in order:
Step1: Check "Error Location". 0.3 points for matching location, proceed; otherwise 0.0 points, stop.
Step2: Check "Error Reason". Add 0.4 points (total 0.7) for equivalent reason, proceed; otherwise keep 0.3 points, stop.
Step3: Check "Corrected Text". Add 0.3 points to reach full score for consistent correction; otherwise keep 0.7 points, stop.

## Equivalence Criteria
- **Error Location**: Different wording allowed, must point to identical text span or semantic unit.
- **Error Reason**: Different diction & sentence structure allowed, must retain the core error point.
- **Corrected Text**: Synonym substitution and rephrasing allowed. Revised professional content must be equivalent to gold standard and free of new errors.

## Input Information
- Original erroneous text: {error_text}
- AI predicted error location: {AI_pos}
- AI predicted error reason: {AI_rsn}
- AI corrected text: {AI_cor}
- Gold-standard error location: {gold_pos}
- Gold-standard error reason: {gold_rsn}
- Gold-standard corrected text: {gold_cor}

## Output Format
Output strictly following the structure below:
<reasoning>
[Write step-by-step reasoning, state comparison basis and points obtained at each step.]
</reasoning>
<score>
[Final score, valid values: 0.0 / 0.3 / 0.7 / 1.0]
</score>
"""

# ECA-style judge: step-wise ALL-OR-NOTHING scoring, {0.0, 1.0}. Kept here alongside the training
# prompt so both scorers can be run from one place; the ECA evaluation itself uses qwen3.8-max and
# lives in code/test/test.py.
JUDGE_PROMPT_ECA = """You are a rigorous yet flexible evaluation expert for error-correction models in the AEC-Q automotive chip testing domain. Evaluate model output and give final score per rules below.

## Evaluation Rules

Step-wise all-or-nothing scoring. Check AI output against gold standard in order:
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
[Write step-by-step reasoning, state comparison basis and judgment result at each step.]</reasoning><score>
[Final score, valid values: 0.0 / 1.0]</score>
"""

JUDGE_TEMPLATES = {"training": JUDGE_PROMPT_TRAINING, "eca": JUDGE_PROMPT_ECA}

# Discrete score sets the two judge prompts can emit
VALID_JUDGE_SCORES = ("0.0", "0.3", "0.7", "1.0")
_SCORE_ALT = "|".join(re.escape(v) for v in VALID_JUDGE_SCORES)


def extract_xml_score(text: str) -> float:
    """
    Read the score out of a judge reply, as a float.

    The prompt asks the judge to answer inside a `<score>...</score>` block, so an end-anchored
    search for a bare trailing number can never match a compliant reply -- it only fires when the
    judge omits the closing tag. Search the block first, fall back to a trailing number.

    Anything unparseable returns 0.0, the safe failure mode: a reply that gave no verdict must not
    be rewarded.
    """
    text = (text or "").strip()

    block = re.search(r"<score\s*>(.*?)</score>", text, re.S)
    if block:
        inner = block.group(1).strip()
        # Preferred form: the block holds nothing but the value
        bare = re.fullmatch(f"({_SCORE_ALT})", inner)
        if bare:
            return float(bare.group(1))
        # Judge added prose around the value; take the first value it names. A judge that merely
        # echoes the prompt's value list yields 0.0, which is again the safe failure mode.
        found = re.search(f"({_SCORE_ALT})", inner)
        return float(found.group(1)) if found else 0.0

    # Unclosed <score>, or a judge that answered without tags at all
    tail = re.search(f"({_SCORE_ALT})\\s*$", text)
    return float(tail.group(1)) if tail else 0.0


def llm_as_a_judge(client, error_text, ai_pos, ai_rsn, ai_cor, gold_pos, gold_rsn, gold_cor,
                   style="training", model="deepseek-chat", max_retries=3) -> float:
    """
    Semantic comparison over the pre-matched CoT elements.

    Performs no full AEC-Q audit, no error localisation and no clause lookup -- the judge only
    compares what the model produced against the gold annotation, which keeps its task load low.

    `client` is an `openai.OpenAI` instance. It is passed in rather than constructed here so that
    each caller keeps its own endpoint and key.
    """
    prompt = JUDGE_TEMPLATES[style].format(
        error_text=error_text, AI_pos=ai_pos, AI_rsn=ai_rsn, AI_cor=ai_cor,
        gold_pos=gold_pos, gold_rsn=gold_rsn, gold_cor=gold_cor,
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
            return extract_xml_score(response.choices[0].message.content)
        except Exception as exc:                      # network / rate limit / malformed reply
            print(f"  [judge] attempt {attempt + 1}/{max_retries} failed: {exc}")
            time.sleep(2 ** attempt)
    # Exhausted retries: return the neutral-low fallback rather than crashing a multi-hour loop
    print("  [judge] all retries exhausted, falling back to 0.0")
    return 0.0


# ==================== Reward aggregate (Eq. 7) ====================
REWARD_MODE_HYBRID = "hybrid"
REWARD_MODE_CODE_ONLY = "code_only"

# Weight split between the code-based and semantic components (Eq. 7)
W_CODE, W_JUDGE = 0.1, 9.0

# Format eligibility credit, granted only when all four tags are present
FORMAT_CREDIT = 0.5
POS_WEIGHT, CORR_WEIGHT = 8.0, 1.5
POS_NO_ERROR_CREDIT = 8.0

# Default hard-sample threshold, in reward units on the 0-10 scale.
DEFAULT_HARD_REWARD_THRESHOLD = 9.0


def compute_code_reward(parsed: dict, gold_pos: str, gold_corr: str) -> float:
    """
    The code-based component: format eligibility, error-position agreement and correction
    agreement, clamped to 0-10.

    A compliant report (empty gold position) is credited in full when the model also says "none",
    rather than being scored on 3-gram overlap with an empty reference -- there is nothing to
    overlap with, and the correct answer here is a single fixed word.
    """
    res_pos = parsed["error_position"]
    if gold_pos == "" and res_pos.lower() == "none":
        pos_score = POS_NO_ERROR_CREDIT
    else:
        pos_score = calculate_3gram_f1(res_pos, gold_pos) * POS_WEIGHT
    corr_score = calculate_3gram_f1(parsed["corrected_text"], gold_corr) * CORR_WEIGHT
    return max(0.0, min(FORMAT_CREDIT + pos_score + corr_score, 10.0))


def compute_mixed_reward(response: str, error_text: str, gold_pos: str, gold_spec: str,
                         gold_corr: str, judge_fn=None,
                         reward_mode: str = REWARD_MODE_HYBRID):
    """
    Eq. 7: R_total = w_code * R_code + w_judge * S_judge, with w_code:w_judge = 0.1:0.9.

    Format eligibility is enforced first -- an output that does not carry the full CoT structure
    receives zero and is never content-scored. That gate applies in every reward mode, including
    the code-only one: it belongs to the output contract rather than to the judge term, so removing
    the judge must not also silently remove the mechanism that enforces the four-tag format.

    `reward_mode`:
      "hybrid"    -- the aggregate above. `judge_fn` must be supplied.
      "code_only" -- the code component alone, on its native 0-10 scale, with no judge call. The
                     scale is deliberately left alone so the hard-sample threshold keeps meaning
                     the same thing across variants; see `ablation_config.reward_mode`.

    `judge_fn(error_text, ai_pos, ai_rsn, ai_cor, gold_pos, gold_rsn, gold_cor) -> float`.

    Returns (total, code_reward, judge_score); judge_score is 0.0 when no judge was consulted.
    """
    parsed = parse_response(response)
    if not has_full_format(parsed):
        return 0.0, 0.0, 0.0

    code_reward = compute_code_reward(parsed, gold_pos, gold_corr)

    if reward_mode == REWARD_MODE_CODE_ONLY:
        return code_reward, code_reward, 0.0

    if judge_fn is None:
        raise ValueError(f"reward_mode={reward_mode!r} requires a judge_fn")

    judge_score = judge_fn(
        error_text, parsed["error_position"], parsed["specific_error_reason"],
        parsed["corrected_text"], gold_pos, gold_spec, gold_corr,
    )
    return code_reward * W_CODE + judge_score * W_JUDGE, code_reward, judge_score


# ==================== Hard-sample identity ====================
SAMPLE_FIELDS = ("error_text", "reasoning", "error_position", "specific_error_reason",
                 "corrected_content")


def structured_data_to_string(error_text, reasoning, gold_pos, gold_spec, gold_corr) -> str:
    """
    Serialise one sample in the `-*-` form stage 4 writes into `hard_samples.txt`, which is what
    `txt2json.py` parses. Used by the static-diagnosis path, which has to produce a dump stage 5
    can consume even though stage 4 never ran.

    The separator is load-bearing in both this function and `sample_key` below. Concatenating the
    five fields with nothing between them would let two different samples collapse onto the same
    string -- an empty field adjacent to a populated one yields the same result as the fields merely
    shifted -- and a colliding key silently drops a hard sample from the pool.
    """
    return f"{error_text}-*-{reasoning}-*-{gold_pos}-*-{gold_spec}-*-{gold_corr}"


def sample_key(sample: dict) -> str:
    """
    Stable identity of a sample: all five gold fields, separated by a character that cannot occur
    in report text.

    Deliberately a different separator from `structured_data_to_string`: this key is the stage-6
    pool's identity and is persisted in `iteration_pool.json` across iterations, so changing it
    would invalidate any pool file already written. Kept byte-identical to the committed stage-6
    behaviour.
    """
    return "\x01".join(str(sample.get(f, "")) for f in SAMPLE_FIELDS)

import json
import random
import time
from typing import List, Dict
from openai import OpenAI

# -------------------------- Basic Configuration --------------------------
# Reuse DeepSeek API configuration from your existing environment; other models can also be substituted
client = OpenAI(
    api_key="xx-xxxxxxxxxxxxxxxxxxxxx",  # Replace with your real API Key
    base_url="https://api.deepseek.com"
)

# Augmentation ratio: strictly follow the paper's 1:5 setting
AUGMENT_RATIO = 5
# Model selection: prioritize general large models with strong reasoning ability to guarantee the rationality of professional error generation
AUGMENT_MODEL = "deepseek-chat"

# Four standard error types (fully consistent with paper description)
ERROR_TYPES = [
    "clause_citation_error",       # Clause citation error
    "test_parameter_deviation",    # Test parameter deviation
    "failure_criterion_misuse",    # Failure criterion misapplication
    "cross_chapter_inconsistency"  # Cross‑chapter data inconsistency
]

# -------------------------- Core Prompt Definition --------------------------
AUGMENT_SYSTEM_PROMPT = """You are a senior expert in AEC-Q automotive-grade chip testing and test report auditing.
Your task is to generate realistic and professional error-injected test report fragments based on the given correct standard text.

## Core Requirements
1. Inject only one type of specified error into the correct text. The error must be subtle, realistic and conform to real‑world AEC‑Q auditing scenarios.
2. The error must be technically reasonable, which is easy to be ignored by inexperienced auditors but violates AEC‑Q specifications.
3. Do not change the overall structure and professional expression of the original text. Only modify the target error part.
4. The corrected_content field must be exactly the same as the original input correct text, as the standard answer.

## Error Type Definitions
1. clause_citation_error: Wrong AEC‑Q standard number, sub‑clause number or standard name. E.g., write AEC‑Q100‑001 as AEC‑Q100‑002.
2. test_parameter_deviation: Minor deviation of test parameters, such as sample size, temperature value, Cpk threshold, voltage, duration. E.g., change 5 devices to 4 devices, change Cpk≥1.67 to Cpk≥1.5.
3. failure_criterion_misuse: Wrong pass/fail threshold, reversed comparison direction, or wrong compliance conclusion. E.g., change "meets the requirements" to "does not meet the requirements", reverse ">" and "<".
4. cross_chapter_inconsistency: Data contradiction between different parts of the text, such as inconsistent parameter values before and after, mismatched device types and test items.

## Output Format
Strictly output a single JSON object with NO extra text, NO markdown code blocks. The JSON must contain 5 fields:
- error_text: The text with injected error
- reasoning: Step‑by‑step auditing reasoning, first compare with domain knowledge, then locate the error unit
- error_position: The smallest complete text segment containing the error
- specific_error_reason: Brief explanation of why it is an error, with AEC‑Q standard basis
- corrected_content: Exactly the original correct text (standard answer)
"""

# -------------------------- Function for Generating Single Error Variant --------------------------
def generate_one_variant(correct_text: str, error_type: str, max_retries: int = 3) -> Dict:
    """
    Call LLM to generate one augmented sample with specified error type
    :param correct_text: Original ground‑truth text (corresponds to corrected_content in sample)
    :param error_type: Target error category
    :param max_retries: Maximum retry attempts for API failure
    """
    user_prompt = f"""Correct standard text:
{correct_text}

Please inject a "{error_type}" error into the above text. Generate the result in required JSON format."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=AUGMENT_MODEL,
                messages=[
                    {"role": "system", "content": AUGMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,  # Balance diversity and professional correctness
                stream=False
            )
            content = response.choices[0].message.content.strip()

            # Clean possible markdown code‑block wrapper
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            sample = json.loads(content)

            # Validate completeness of required json fields
            required_fields = ["error_text", "reasoning", "error_position", "specific_error_reason", "corrected_content"]
            if all(field in sample for field in required_fields):
                # Force ground‑truth answer to be identical to source text
                sample["corrected_content"] = correct_text
                return sample
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {str(e)}")
            time.sleep(1)

    # Fallback template when LLM generation exhausts retries
    print(f"  LLM generation failed after {max_retries} retries, fallback to rule‑based.")
    return {
        "error_text": correct_text + " [Error injection failed placeholder]",
        "reasoning": "Fallback sample due to generation failure.",
        "error_position": "N/A",
        "specific_error_reason": "N/A",
        "corrected_content": correct_text
    }

# -------------------------- Batch Augmentation for One Hard Sample --------------------------
def augment_single_sample(original_sample: Dict) -> List[Dict]:
    """
    Generate 5 augmented variants from one source hard sample: four defined error types + one random variant
    :param original_sample: Source hard‑sample dictionary loaded from json
    """
    correct_text = original_sample["corrected_content"]
    augmented_samples = []

    # Generate 4 variants corresponding to four predefined error types
    for err_type in ERROR_TYPES:
        variant = generate_one_variant(correct_text, err_type)
        augmented_samples.append(variant)
        time.sleep(0.5)  # Throttle api request rate

    # Fifth variant: randomly pick error type to increase diversity
    random_type = random.choice(ERROR_TYPES)
    variant = generate_one_variant(correct_text, random_type)
    augmented_samples.append(variant)

    # Filter fallback placeholder samples; keep filling until reaching target count
    valid_samples = [s for s in augmented_samples if s["error_position"] != "N/A"]
    while len(valid_samples) < AUGMENT_RATIO:
        random_type = random.choice(ERROR_TYPES)
        variant = generate_one_variant(correct_text, random_type)
        if variant["error_position"] != "N/A":
            valid_samples.append(variant)

    return valid_samples[:AUGMENT_RATIO]

# -------------------------- Main Entry --------------------------
def main():
    input_path = "../../data/train/Hard_samples/origin_hard_samples.json"
    output_path = "../../data/train/Hard_samples/agumented_hard_samples.json"

    # Load original hard‑sample dataset
    with open(input_path, "r", encoding="utf-8") as f:
        origin_samples = json.load(f)

    print(f"Loaded {len(origin_samples)} original hard samples.")
    print(f"Generating {AUGMENT_RATIO} variants per sample, total expected: {len(origin_samples)*AUGMENT_RATIO}")

    # Iterate and perform sample augmentation
    all_augmented = []
    for idx, sample in enumerate(origin_samples):
        print(f"\nProcessing sample {idx+1}/{len(origin_samples)}...")
        variants = augment_single_sample(sample)
        all_augmented.extend(variants)
        print(f"  Generated {len(variants)} valid variants.")

    print(f"\nAugmentation completed. Total valid augmented samples: {len(all_augmented)}")

    # Persist augmented dataset, keep format consistent with source data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_augmented, f, indent=4, ensure_ascii=False)

    print(f"Augmented samples saved to: {output_path}")


if __name__ == "__main__":
    main()

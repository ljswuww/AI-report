import json
import re
import random
from typing import List, Dict, Tuple, Optional

# Fix random seed for reproducibility
random.seed(42)

# -------------------------- Helper Utility Functions --------------------------
def invert_compare(op: str) -> str:
    """Reverse comparison operator for failure‑criterion error injection"""
    if '>=' in op:
        return '<='
    elif '<=' in op:
        return '>='
    elif '>' in op:
        return '<'
    elif '<' in op:
        return '>'
    else:
        return '!='

# -------------------------- Four Error Injection Functions (strictly follow paper description) --------------------------
def inject_clause_citation_error(ground_truth: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Inject clause‑citation error: modify AEC‑Q standard number / sub‑clause number
    Corresponding to paper: clause citation errors
    """
    pattern = r'AEC-Q(\d+)(-\d+)?'
    matches = list(re.finditer(pattern, ground_truth))
    if not matches:
        return None

    match = random.choice(matches)
    original_citation = match.group(0)
    main_num = match.group(1)
    sub_num = match.group(2)

    # Generate incorrect standard reference number
    main_num_int = int(main_num)
    wrong_main = main_num_int + random.choice([-1, 1, 10, -10])
    wrong_main_str = f"{wrong_main:03d}" if len(main_num) == 3 else str(wrong_main)

    if sub_num:
        sub_num_int = int(sub_num[1:])
        wrong_sub = sub_num_int + random.choice([-1, 1, 2, -2])
        wrong_sub_str = f"-{wrong_sub:03d}"
    else:
        wrong_sub_str = ""

    wrong_citation = f"AEC-Q{wrong_main_str}{wrong_sub_str}"

    # Generate error‑containing text and corresponding annotations
    error_text = ground_truth[:match.start()] + wrong_citation + ground_truth[match.end():]
    error_position = wrong_citation
    error_reason = f"Incorrect AEC-Q standard clause citation. The correct reference should be '{original_citation}', but it is wrongly written as '{wrong_citation}'."
    reasoning = f"【Domain Knowledge Comparison】According to professional AEC-Q standard specifications, the standard clause citation in the text has an error. The correct standard designation is '{original_citation}', while the text incorrectly uses '{wrong_citation}', which does not conform to the correct standard reference specification. 【Error Unit Localization】The error is located in the standard citation segment."

    return error_text, error_position, error_reason, reasoning


def inject_test_parameter_deviation(ground_truth: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Inject test‑parameter deviation: modify sample count, temperature, Cpk, voltage and other numerical parameters
    Corresponding to paper: test parameter deviations
    """
    # Prioritize patterns matching business‑relevant parameters
    param_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:devices|samples|bond points|lots|units)',
        r'Cpk\s*=\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*°C',
        r'(\d+(?:\.\d+)?)\s*V',
        r'(\d+(?:\.\d+)?)\s*mA',
        r'(\d+(?:\.\d+)?)\s*hours?'
    ]

    for pattern in param_patterns:
        matches = list(re.finditer(pattern, ground_truth))
        if matches:
            match = random.choice(matches)
            original_val_str = match.group(1)
            original_val = float(original_val_str)

            # Generate deviated value (integer ±1~3; float ±20%~40%)
            if original_val.is_integer():
                deviation = random.randint(1, max(1, int(original_val * 0.3)))
                wrong_val = original_val + random.choice([-deviation, deviation])
                wrong_val_str = str(int(wrong_val))
            else:
                deviation = original_val * random.uniform(0.2, 0.4)
                wrong_val = original_val + random.choice([-deviation, deviation])
                wrong_val_str = f"{wrong_val:.1f}"

            # Replace target numeric value
            val_start = match.start(1)
            val_end = match.end(1)
            error_text = ground_truth[:val_start] + wrong_val_str + ground_truth[val_end:]
            error_position = wrong_val_str + ground_truth[val_end:match.end()]

            error_reason = f"Test parameter deviation. The correct value should be {original_val_str}, but it is wrongly set to {wrong_val_str}, which does not meet AEC-Q parameter requirements."
            reasoning = f"【Domain Knowledge Comparison】According to AEC-Q test specifications, the test parameter in the text has a deviation. The correct parameter value should be {original_val_str}, while the text incorrectly uses {wrong_val_str}, which fails to satisfy the standard parameter requirements. 【Error Unit Localization】The error is located in the test parameter segment."

            return error_text, error_position, error_reason, reasoning

    # Fallback: match generic numeric values
    general_pattern = r'\b(\d+(?:\.\d+)?)\b'
    matches = list(re.finditer(general_pattern, ground_truth))
    if matches:
        match = random.choice(matches)
        original_val_str = match.group(1)
        original_val = float(original_val_str)
        if original_val == 0:
            return None

        wrong_val = original_val * random.uniform(0.6, 0.8)
        wrong_val_str = f"{wrong_val:.1f}" if '.' in original_val_str else str(int(wrong_val))

        error_text = ground_truth[:match.start()] + wrong_val_str + ground_truth[match.end():]
        error_position = wrong_val_str
        error_reason = f"Test parameter deviation. The correct value should be {original_val_str}, but it is wrongly written as {wrong_val_str}."
        reasoning = f"【Domain Knowledge Comparison】According to relevant test specifications, the parameter value in the text is incorrect. The correct value should be {original_val_str}, while the text gives {wrong_val_str}. 【Error Unit Localization】The error is located at the numerical parameter."

        return error_text, error_position, error_reason, reasoning

    return None


def inject_failure_criterion_misuse(ground_truth: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Inject failure‑criterion misapplication: modify pass/fail threshold, comparison direction or compliance conclusion
    Corresponding to paper: failure criterion misapplication
    """
    criterion_patterns = [
        (r'(Cpk\s*)([><=]+)\s*(\d+(?:\.\d+)?)',
         lambda m: (m.group(1), invert_compare(m.group(2)), m.group(3))),
        (r'(\d+)\s*failure(s)?',
         lambda m: (str(max(0, int(m.group(1)) + random.choice([-1, 1]))), m.group(2))),
        (r'meets\s+the\s+standard\s+requirements',
         lambda m: 'does not meet the standard requirements'),
        (r'passes?\s+the\s+test',
         lambda m: 'fails the test'),
        (r'compliant\s+with\s+AEC-Q',
         lambda m: 'not compliant with AEC-Q')
    ]

    for pattern, repl_func in criterion_patterns:
        matches = list(re.finditer(pattern, ground_truth, re.IGNORECASE))
        if matches:
            match = random.choice(matches)
            original_segment = match.group(0)
            wrong_segment = ''.join(repl_func(match))

            error_text = ground_truth[:match.start()] + wrong_segment + ground_truth[match.end():]
            error_position = wrong_segment
            error_reason = f"Failure criterion misapplication. The correct criterion should be '{original_segment}', but it is wrongly set to '{wrong_segment}', resulting in incorrect compliance judgment."
            reasoning = f"【Domain Knowledge Comparison】According to AEC-Q test failure criteria, the judgment criterion in the text is misused. The correct criterion should be '{original_segment}', while the text incorrectly uses '{wrong_segment}', which will lead to wrong compliance conclusion. 【Error Unit Localization】The error is located in the failure criterion segment."

            return error_text, error_position, error_reason, reasoning

    return None


def inject_cross_chapter_inconsistency(ground_truth: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Inject cross‑chapter data inconsistency: create contradictory parameters or type conflicts within text
    Corresponding to paper: cross‑chapter data inconsistency
    """
    # Strategy 1: find duplicated numeric parameters and modify one occurrence
    param_pattern = r'(\d+(?:\.\d+)?)\s*(devices|samples|bond points|lots|°C|V|mA)'
    matches = list(re.finditer(param_pattern, ground_truth, re.IGNORECASE))

    if len(matches) >= 2:
        match1, match2 = random.sample(matches, 2)
        original_val = match1.group(1)
        unit = match1.group(2)

        val = float(original_val)
        wrong_val = val + random.choice([-1, 1, -2, 2]) if val.is_integer() else val * 0.8
        wrong_val_str = str(int(wrong_val)) if val.is_integer() else f"{wrong_val:.1f}"

        error_text = ground_truth[:match1.start()] + wrong_val_str + " " + unit + ground_truth[match1.end():]
        error_position = wrong_val_str + " " + unit

        error_reason = f"Cross-chapter data inconsistency. The {unit} parameter is recorded as {match2.group(1)} elsewhere in the text, but it is incorrectly written as {wrong_val_str} here, resulting in conflicting data across sections."
        reasoning = f"【Domain Knowledge Comparison】By cross-checking the data throughout the text, there is an inconsistency in the {unit} parameter. The value recorded in another part is {match2.group(1)}, while the value here is {wrong_val_str}, which does not match. 【Error Unit Localization】The error is located at the inconsistent parameter segment."

        return error_text, error_position, error_reason, reasoning

    # Strategy 2: modify device/package type to trigger type contradiction
    type_patterns = [r'hermetic packages?', r'plastic packages?', r'BGA devices?']
    for pattern in type_patterns:
        matches = list(re.finditer(pattern, ground_truth, re.IGNORECASE))
        if matches:
            match = random.choice(matches)
            original_type = match.group(0)

            if 'hermetic' in original_type.lower():
                wrong_type = 'plastic packages' if 'package' in original_type.lower() else 'plastic package'
            elif 'plastic' in original_type.lower():
                wrong_type = 'hermetic packages' if 'package' in original_type.lower() else 'hermetic package'
            elif 'bga' in original_type.lower():
                wrong_type = 'QFP devices' if 'device' in original_type.lower() else 'QFP device'
            else:
                continue

            error_text = ground_truth[:match.start()] + wrong_type + ground_truth[match.end():]
            error_position = wrong_type
            error_reason = f"Cross-chapter data inconsistency. The device/package type is described as '{original_type}' in the standard context, but it is wrongly written as '{wrong_type}' here, causing type mismatch."
            reasoning = f"【Domain Knowledge Comparison】By cross-verifying the device type description in the text, there is an inconsistency. The correct type should be '{original_type}', while it is incorrectly described as '{wrong_type}' here, which conflicts with the overall specification. 【Error Unit Localization】The error is located at the device type description segment."

            return error_text, error_position, error_reason, reasoning

    return None

# -------------------------- Core Augmentation Logic --------------------------
def augment_single_sample(original_sample: Dict, variant_num: int = 5) -> List[Dict]:
    """
    Generate variant_num augmented variants for one single hard sample (default ratio 1:5 as in paper)
    :param original_sample: Source hard‑sample dictionary
    :param variant_num: Number of variants to generate, following the 1:5 ratio from paper
    """
    # Inject errors based on ground‑truth text to guarantee annotation correctness
    ground_truth = original_sample['corrected_content']
    augmented_samples = []

    # Generate one base variant for each of four error categories
    error_functions = [
        inject_clause_citation_error,
        inject_test_parameter_deviation,
        inject_failure_criterion_misuse,
        inject_cross_chapter_inconsistency
    ]

    for func in error_functions:
        result = func(ground_truth)
        if result:
            error_text, error_pos, error_reason, reasoning = result
            augmented_samples.append({
                "error_text": error_text,
                "reasoning": reasoning,
                "error_position": error_pos,
                "specific_error_reason": error_reason,
                "corrected_content": ground_truth
            })

    # Generate remaining variants: randomly select error types for diverse positions / magnitudes, or create mixed‑error samples
    remaining = variant_num - len(augmented_samples)
    for _ in range(remaining):
        random.shuffle(error_functions)
        for func in error_functions:
            result = func(ground_truth)
            if result:
                error_text, error_pos, error_reason, reasoning = result
                # Skip if exactly identical sample already exists
                if not any(s['error_text'] == error_text for s in augmented_samples):
                    augmented_samples.append({
                        "error_text": error_text,
                        "reasoning": reasoning,
                        "error_position": error_pos,
                        "specific_error_reason": error_reason,
                        "corrected_content": ground_truth
                    })
                    break

    # Extreme fallback: create mixed‑error samples if insufficient valid variants can be injected
    while len(augmented_samples) < variant_num:
        if augmented_samples:
            base_sample = random.choice(augmented_samples)
            # Inject a second error onto an existing erroneous sample
            for func in error_functions:
                result = func(base_sample['error_text'])
                if result:
                    error_text, error_pos, error_reason, reasoning = result
                    augmented_samples.append({
                        "error_text": error_text,
                        "reasoning": base_sample['reasoning'] + " " + reasoning,
                        "error_position": f"{base_sample['error_position']}; {error_pos}",
                        "specific_error_reason": f"{base_sample['specific_error_reason']} Meanwhile, {error_reason.lower()}",
                        "corrected_content": ground_truth
                    })
                    break
        else:
            # Fallback: keep original sample if no error can be injected (rare corner case)
            augmented_samples.append(original_sample.copy())

    return augmented_samples[:variant_num]

# -------------------------- Main Entry Point --------------------------
def main():
    # Path configuration. Routed through `override` so an ablation variant can run its own
    # augmentation into its own working directory; with no ABLATION_* set the pipeline defaults
    # below apply and a direct run is unchanged.
    # `ablation_config` lives one level up; running this file directly puts only `utils/` on
    # sys.path, so the parent directory has to be added explicitly.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from ablation_config import override

    input_path = override("augment.INPUT_JSON",
                          "../../data/train/Hard_samples/origin_hard_samples.json")
    output_path = override("augment.OUTPUT_JSON",
                           "../../data/train/Hard_samples/augmented_hard_samples.json")

    # Load original hard‑sample dataset
    with open(input_path, 'r', encoding='utf-8') as f:
        origin_hard_samples = json.load(f)
    print(f"Loaded {len(origin_hard_samples)} original hard samples.")

    # Perform batch‑wise sample augmentation
    all_augmented = []
    for idx, sample in enumerate(origin_hard_samples):
        variants = augment_single_sample(sample, variant_num=5)
        all_augmented.extend(variants)
        print(f"Sample {idx+1}/{len(origin_hard_samples)}: generated {len(variants)} variants.")

    print(f"\nAugmentation completed. Total augmented samples: {len(all_augmented)}")

    # Persist augmented dataset to disk
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_augmented, f, indent=4, ensure_ascii=False)
    print(f"Augmented samples saved to: {output_path}")


if __name__ == "__main__":
    main()

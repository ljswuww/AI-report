# -*- coding: utf-8 -*-
"""
Regression test for the training reward (Eqs. 1-7).

The reward is the one component every stage and every ablation variant depends on, and it is easy to
break silently: a changed n-gram construction or a shifted weight still produces a number, just a
different one, and nothing downstream would complain. The expected values below are therefore pinned.

They were not derived from this implementation. Each was first reproduced by executing the reward
block exactly as committed in `6model_iteration.py` and comparing outputs over 4000 randomised
n-gram triples plus fixed text and judge-reply cases; the pinned values here are that verified
behaviour, frozen. A change to the reward that alters any of them is a change to the paper's
definition and should be deliberate.

Run either way:

    python test_reward_utils.py        # prints a summary
    pytest test_reward_utils.py        # collects the test_* functions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reward_utils as R  # noqa: E402

# (prediction, gold, expected 3-gram F1)
THREE_GRAM_CASES = [
    # one quantity differs; the rest of the sentence still matches
    ("测试报告显示 VDD 为 3.3V。", "测试报告显示 VDD 为 5.0V。", 0.8571428571428571),
    # identical text is a perfect match
    ("测试报告显示 VDD 为 3.3V。", "测试报告显示 VDD 为 3.3V。", 1.0),
    # both sides shorter than 3 tokens: falls back to exact match, not n-gram overlap
    ("参数偏差", "参数偏离", 0.5),
    ("ab", "ab", 1.0),
    ("ab", "cd", 0.0),
    # an empty side scores 0 rather than a vacuous match
    ("", "none", 0.0),
    ("none", "", 0.0),
    # multiset, not set: a repeated phrase is penalised
    ("重复重复重复重复", "重复重复重复", 0.8),
]

# (judge reply, expected parsed score)
JUDGE_REPLY_CASES = [
    ("<score>0.7</score>", 0.7),
    ("<score>0.0</score>", 0.0),
    ("<score>1.0</score>", 1.0),
    # a judge that echoes the prompt's value list gave no verdict -> the safe failure mode
    ("<score>0.0 / 0.3 / 0.7 / 1.0</score>", 0.0),
    # unclosed tag, or no tags at all
    ("<score>0.7", 0.7),
    ("no tags 0.3", 0.3),
    ("", 0.0),
]

WELL_FORMED = ("<reasoning>r</reasoning><error_position>位置A</error_position>"
               "<specific_error_reason>原因</specific_error_reason>"
               "<corrected_text>改正后的文本</corrected_text>")
MALFORMED = "no tags at all"
COMPLIANT = ("<reasoning>r</reasoning><error_position>none</error_position>"
             "<specific_error_reason>none</specific_error_reason>"
             "<corrected_text>原文</corrected_text>")


def test_three_gram_f1():
    for pred, gold, expected in THREE_GRAM_CASES:
        got = R.calculate_3gram_f1(pred, gold)
        assert abs(got - expected) < 1e-12, f"f1({pred!r}, {gold!r}) = {got!r}, expected {expected!r}"


def test_judge_score_parsing():
    for reply, expected in JUDGE_REPLY_CASES:
        got = R.extract_xml_score(reply)
        assert got == expected, f"extract_xml_score({reply!r}) = {got!r}, expected {expected!r}"


def test_code_reward_scale_and_credits():
    # a perfect correction saturates the 0-10 code scale
    assert R.compute_code_reward(R.parse_response(WELL_FORMED), "位置A", "改正后的文本") == 10.0
    # a compliant report that correctly answers "none" is credited in full, not scored against
    # an empty reference
    assert R.compute_code_reward(R.parse_response(COMPLIANT), "", "原文") == 10.0
    # a missing tag is not eligible, so it is not content-scored
    assert R.compute_mixed_reward(MALFORMED, "原文", "位置A", "原因", "改正后",
                                  judge_fn=lambda *a: 1.0) == (0.0, 0.0, 0.0)


def test_hybrid_weights():
    # Eq. 7: R_total = 0.1 * R_code + 9.0 * S_judge
    total, code, judge = R.compute_mixed_reward(WELL_FORMED, "原文", "位置A", "原因", "改正后的文本",
                                                judge_fn=lambda *a: 0.7)
    assert code == 10.0 and judge == 0.7
    assert abs(total - (code * 0.1 + judge * 9.0)) < 1e-12, total


def test_code_only_mode_drops_the_judge():
    def explode(*args):
        raise AssertionError("the judge must not be consulted in code-only mode")

    total, code, judge = R.compute_mixed_reward(
        WELL_FORMED, "原文", "位置A", "原因", "改正后的文本",
        judge_fn=explode, reward_mode=R.REWARD_MODE_CODE_ONLY,
    )
    # the code component alone, on its native scale -- not rescaled
    assert total == code == 10.0
    assert judge == 0.0

    # the format gate belongs to the output contract, so it still applies with the judge removed
    assert R.compute_mixed_reward(
        MALFORMED, "原文", "位置A", "原因", "改正后",
        judge_fn=explode, reward_mode=R.REWARD_MODE_CODE_ONLY,
    ) == (0.0, 0.0, 0.0)


def test_hybrid_requires_a_judge():
    try:
        R.compute_mixed_reward(WELL_FORMED, "原文", "位置A", "原因", "改正后", judge_fn=None)
    except ValueError:
        return
    raise AssertionError("hybrid mode with no judge_fn should raise, not silently score")


def test_hard_sample_key_is_stable():
    sample = {"error_text": "a", "reasoning": "b", "error_position": "c",
              "specific_error_reason": "d", "corrected_content": "e"}
    assert R.sample_key(sample) == "\x01".join("abcde")
    # the dump encoding is a different delimiter, and must be the one txt2json.py parses
    assert R.structured_data_to_string("a", "b", "c", "d", "e") == "a-*-b-*-c-*-d-*-e"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

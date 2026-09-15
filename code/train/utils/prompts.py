# -*- coding: utf-8 -*-
"""
The auditing system prompt, for callers that are not one of the six pipeline stage scripts.

The stage scripts each embed this prompt as their own module constant, which is deliberate: they are
written to be runnable as standalone scripts and are executed as separate processes. This module is
the copy for everything else -- currently static hard-sample diagnosis, which has to present the
model with the same input paradigm the GRPO stages used, or it would be diagnosing a differently
prompted policy.

Note for anyone diffing the copies: the stage scripts' versions are not byte-identical to each
other. Compared with the file in `prompts/` and with this copy:

- `2cold_start_SFT.py` is reworded in eight places. It is a paraphrase, not a whitespace variant,
  and it is the prompt the model is actually supervised on in stage 2. It is left as it is because
  editing it would change the token sequence of an existing training run.
- `5DAP_for_hardsamples.py` carries the same wording under the constant name
  `SYSTEM_PROMPT_TEMPLATE`, with four section headings losing a second space (`###  TASKS ###` ->
  `### TASKS ###`). Headings are markers, not instructions, so this is a spacing variant.
- The remaining copies differ in blank lines and trailing whitespace only, which does not change
  the rendered prompt in any way a model can observe, but does mean a naive string comparison
  between the copies will report a difference. This copy matches `6model_iteration.py`.
"""

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

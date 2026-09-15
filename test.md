# Experimental Details

Supplementary implementation details for the closed-loop reinforcement-learning pipeline for
intelligent auditing of AEC-Q-compliant automotive-grade chip test reports.

This document records the environment, data construction, per-stage configurations, reward design,
hard-sample pool mechanics, and evaluation protocol. Figures reported in the main text are not
repeated here.

---

## 1. Computational environment

| Item | Value |
|---|---|
| Base model | Qwen2.5-14B-Instruct (`model/base/qwen2.5-14b-instruct`) |
| Hardware | 3 × NVIDIA A100, 80 GB each (240 GB total); `CUDA_VISIBLE_DEVICES=0,1,2` |
| OS | CentOS Linux 7 |
| CUDA | 11.8 |
| PyTorch | 2.6.0 |
| Mixed precision | BF16 (`bf16=True`), TF32 enabled |
| Optimiser | `adamw_torch_fused` (all Trainer stages) |
| Hub mirror | `HF_ENDPOINT=https://hf-mirror.com` |

All stages perform **full-parameter** fine-tuning of the same 14B checkpoint: no quantisation, no
LoRA or other adapter modules, and no model parallelism beyond `device_map="auto"`. Gradient
checkpointing is enabled for cold-start SFT and both GRPO stages, and disabled for the two
domain-adaptive pretraining stages (which are memory-bound rather than activation-bound).

---

## 2. Data

### 2.1 Domain-adaptive pretraining (DAP) corpus — 10,587 samples

Merged at load time from two directories, 18 JSON files in total:

| Directory | Files | Samples |
|---|---|---|
| `data/train/DAP_data/few_shot_ft` | 10 | 6,853 |
| `data/train/DAP_data/few_shot_short_length` | 8 | 3,734 |
| **Total** | **18** | **10,587** |

Files are partitioned by token tier — 256, 512, 1024 and 2048 — so that each segment respects the
model's input-length constraint. Within a tier, two record types are interleaved: raw overlapping
segments, and knowledge question–answer pairs (`*_noQA.json` files). Every record is a single
`{"text": ...}` field.

Tokenisation: `max_length=2048`, `padding="max_length"`, and `labels = input_ids` — an unmasked
causal language modelling objective over the whole sequence. A 0.1 % split is held out
(`train_test_split(test_size=0.001)`) as a training-time loss probe only; it is not used for model
selection.

### 2.2 Chain-of-thought (CoT) supervision data

All CoT records use the same five-field schema: `error_text`, `reasoning`, `error_position`,
`specific_error_reason`, `corrected_content`.

| File | Samples | Consumed by |
|---|---|---|
| `data/train/RL_data/Cold_start/cold_start_samples.json` | 50 | Stage 2 (cold-start SFT) |
| `data/train/RL_data/GRPO_format/RL__format_samples.json` | 465 | Stage 3 (format GRPO) |
| `data/train/RL_data/GRPO_content/RL_content_samples.json` | 652 | Stage 4 (content GRPO), Stage 6 (CIO) |

The format-stage file name contains a double underscore (`RL__format_samples.json`).

### 2.3 Hard-sample data

| File | Samples | Role |
|---|---|---|
| `data/train/Hard_samples/origin_hard_samples.json` | 30 | Source pool of diagnosed hard samples |
| `data/train/Hard_samples/hard_samples.txt` | 30 | Same records, flushed by stage 4 |
| `data/train/Hard_samples/agumented_hard_samples.json` | 30 | Augmented set consumed by stage 5 |

Stage 4 dumps its accumulated hard-sample set to `hard_samples.txt` in a `finally` block, so the
file survives an interrupted run. Records are joined with the `*|||*\n` separator, and the five
fields inside each record are joined with `-*-`. `txt2json.py` converts that dump back into the
JSON five-field schema. Hard dumps live under `data/train/Hard_samples/` rather than alongside the
training scripts, since they are produced by training but consumed as data.

**Augmentation.** The 1:5 ratio is implemented in two interchangeable scripts:

- `code/train/utils/Rules_augmented_hardsamples.py` — deterministic and offline.
  `augment_single_sample(sample, variant_num=5)` produces one variant per error category, then
  fills the remainder with randomly drawn categories; a mixed-error construction is used as a
  fallback if too few valid single-error variants can be injected.
- `code/train/utils/LLM_augmented_hardsamples.py` — `AUGMENT_RATIO = 5`, four variants for the
  four categories plus one randomly chosen fifth. Each generation is retried three times, and the
  script falls back to the rule-based generator when retries are exhausted, so the ratio holds
  even under API failure.

Both scripts operate on four error categories: numeric main parameter, numeric sub-parameter, unit,
and parameter deviation. Numeric categories perturb the original value by ±1, ±2 or ±10 for the
main parameter and ±1 or ±2 for a sub-parameter.

### 2.4 Evaluation set

`data/test/test_samples.json` — 200 records, same five-field schema as the CoT data (without the
`answer` alias). No training script loads this path; it is fully isolated from every stage,
including the closed-loop iterations.

167 records carry a gold error span. The remaining 33 are compliant reports with no error: the
gold `error_position` and `specific_error_reason` are empty strings and `corrected_content` is
identical to `error_text`. These are scored and reported as a separate cohort (see §7.3).

---

## 3. Structured output schema

Every stage drives the model to a fixed four-tag output. The tag order is part of the contract:

```
<reasoning>              reasoning process
<error_position>         smallest complete error-containing segment, or "none"
<specific_error_reason>  brief error cause, or "none"
<corrected_text>         complete revised text
```

The system prompt is stored in
`prompts/LLM Prompt for Intelligent Auditing of AEC-Q Test Reports.txt` and is embedded verbatim as
a module constant in `1DAP.py`, `2cold_start_SFT.py`, `3grpo_format.py`, `4grpo_content.py`,
`5DAP_for_hardsamples.py` and `code/test/test.py`, so downstream stages see an in-distribution
prompt. `{Input_text}` is the only substitution slot.

---

## 4. Stage configurations

Scripts run from `code/train` unless noted. Each stage initialises from the previous stage's
output directory; no stage re-initialises from a public checkpoint after stage 1.

### Stage 1 — Domain-adaptive pretraining

| | |
|---|---|
| Script | `1DAP.py` |
| Init from | Qwen2.5-14B-Instruct |
| Data | 10,587 DAP samples |
| Output | `model/output/1DAP_output` |
| Sequence length | 2048 |
| Epochs | 10 |
| Batch size | 2 per device × gradient accumulation 4 (effective 8) |
| Learning rate | 3 × 10⁻⁵ |
| Scheduler | `cosine_with_restarts` |
| Warm-up | 250 steps |
| Weight decay | 0 |
| Gradient clipping | 1.0 |
| Gradient checkpointing | off |
| Eval / save interval | every 20 / 150 steps |

### Stage 2 — Cold-start fine-tuning (CSFT)

| | |
|---|---|
| Script | `2cold_start_SFT.py` |
| Init from | `1DAP_output` |
| Data | 50 cold-start records, repeated twice (×2) |
| Output | `model/output/2Cold_start_output` |
| Sequence length | 1024 |
| Epochs | 20 |
| Batch size | 2 per device × gradient accumulation 4 (effective 8) |
| Learning rate | 3 × 10⁻⁵ |
| Scheduler | `cosine_with_restarts` |
| Warm-up | 10 steps |
| Weight decay | 0.03 |
| Gradient clipping | 1.0 |
| Gradient checkpointing | on |
| Held-out split | 2 % |

**Answer-only loss masking.** The chat template is applied manually, then the assistant span is
located with a sliding-window scan over the token sequence for
`<|im_start|>assistant\n`. Every position before the end of that marker is set to `-100` in the
label tensor, so the loss is computed on the response only. If the marker is not found anywhere in
a sequence, all of its labels are masked, which drops that sequence from the gradient rather than
letting it train on prompt tokens.

### Stage 3 — Format-oriented GRPO

| | |
|---|---|
| Script | `3grpo_format.py` |
| Init from | `2Cold_start_output` |
| Data | 465 format-stage CoT records |
| Output | `model/output/3GRPO_format_output` |
| Epochs | 10 |
| Batch size | 2 per device × gradient accumulation 2 |
| Group size (`num_generations`) | 4 |
| Generation batch size | 4 |
| Max prompt / completion length | 896 / 1280 |
| Learning rate | 5 × 10⁻⁶ |
| Adam β₁ / β₂ | 0.9 / 0.95 |
| Weight decay | 0.01 |
| Warm-up ratio | 0.05 |
| Scheduler | `cosine` |
| Gradient clipping | 1.0 |
| Reward | Format reward only (§5.1) |

### Stage 4 — Content-oriented GRPO

| | |
|---|---|
| Script | `4grpo_content.py` |
| Init from | `3GRPO_format_output` |
| Data | 652 content-stage CoT records |
| Output | `model/output/4GRPO_content_output` |
| Epochs | 10 |
| Batch size | 2 per device × gradient accumulation 2 |
| Group size (`num_generations`) | 4 |
| Generation batch size | 4 |
| Max prompt / completion length | 896 / 1280 |
| Learning rate | 5 × 10⁻⁶ |
| Adam β₁ / β₂ | 0.9 / 0.95 |
| Weight decay | 0.01 |
| Warm-up ratio | 0.05 |
| Scheduler | `cosine` |
| Gradient clipping | 1.0 |
| Reward | Hybrid reward (§5.2) |

Sampling configuration for both GRPO stages, set on the model's generation config rather than in
`GRPOConfig`: `temperature = 1.3`, `top_p = 0.93`, `top_k = -1`, `do_sample = True`,
`repetition_penalty = 1.1`, `num_beams = 1`, `max_new_tokens = 1280`. The deliberately high
temperature widens exploration in a task whose reward is sparse and largely binary.

### Stage 5 — Hard-sample targeted optimisation (HSO)

| | |
|---|---|
| Script | `5DAP_for_hardsamples.py` |
| Init from | `4GRPO_content_output` |
| Data | `agumented_hard_samples.json`, converted on the fly |
| Output | `model/output/5DAP4HS_output` |
| Sequence length | 2048 |
| Epochs | 5 |
| Batch size | 2 per device × gradient accumulation 4 |
| Learning rate | 3 × 10⁻⁵ |
| Scheduler | `cosine_with_restarts` |
| Warm-up | 250 steps |
| Weight decay | 0 |
| Gradient clipping | 1.0 |
| Objective | Unmasked causal LM (`labels = input_ids`) |

Each augmented record is rendered as *system prompt with `{Input_text}` substituted* + the four-tag
response, with empty location/reason fields written as the literal `none`. This is the same
sequence form used in DAP, so the stage is a targeted continuation of pretraining on the error
patterns the policy still fails, rather than a supervised fine-tuning pass.

### Stage 6 — Closed-loop iterative optimisation (CIO)

| | |
|---|---|
| Script | `6model_iteration.py` |
| Init from | `5DAP4HS_output` |
| Data | 652 base CoT records ∪ accumulated augmented hard samples |
| Output | `model/output/6Iteration_output/iter_{1..N}`, plus `final/` |
| Iterations | 3 |
| Epochs per iteration | 3 |
| Train / validation split | 9 : 1, fixed by seed 42 |
| Reward | Hybrid reward, training-style judge (§5.2) |
| Augmentation mode | `rules` (`AUGMENT_MODE`), 5 variants per sample |
| Convergence tolerance | 3 % |
| Convergence patience | 2 rounds |
| Hard-pool ratio bound | 5 % |

Per-iteration GRPO hyperparameters follow stage 4, with epochs reduced to 3 because the loop
retrains up to three times. The train/validation split is fixed by seed so that every iteration is
scored on exactly the same validation samples; a drifting validation set would make the
round-over-round reward comparison meaningless.

Contract of the loop, per iteration:

1. Train GRPO on the base CoT set unioned with the accumulated augmented hard samples. The
   checkpoint lands in `iter_N`.
2. Refresh the hard-sample pool from the rewards observed during that iteration (§6).
3. Augment the newly diagnosed hard samples at 1:5 and merge them into the accumulated set that
   seeds the next iteration.
4. Evaluate the mixed reward on the fixed validation split.
5. Apply the convergence test; stop early if it passes.

A `finally` block always writes the hard-sample pool, the accumulated augmented set, the
per-iteration report, and the `final/` model, so an interrupted or non-converged run still yields
a usable artifact.

Note that with `CONVERGENCE_PATIENCE = 2` and `MAX_ITERATIONS = 3`, the convergence criterion can
first be satisfied at the third round — the loop therefore runs all three iterations unless the
patience or the iteration budget is changed.

---

## 5. Reward design

### 5.1 Format reward (stage 3)

Every present and well-formed tag scores 1.8 points (four tags → 7.2), and a further 2.8-point
bonus is granted when all four tags are present **and** their opening positions appear in the
required order `reasoning → error_position → specific_error_reason → corrected_text`. The total is
clamped to [0, 10].

This stage carries no content signal: it teaches the output contract only, which is why it
precedes the content stage.

### 5.2 Hybrid reward (stages 4 and 6)

**Eligibility gate.** If any of the four tags is missing, the reward for that sample is 0.0 and no
further scoring happens.

**Code-based component**, range [0, 10]:

```
code_reward = 0.5  +  pos_score  +  corr_score
```

- `0.5` — flat format credit for producing all four tags.
- `pos_score` — `8.0 × 3-gram-F1(predicted error position, gold error position)`. When the gold
  position is empty (compliant report), the term is instead `8.0` if the predicted position
  normalises to `none`, and `0.0` otherwise.
- `corr_score` — `1.5 × 3-gram-F1(predicted corrected text, gold corrected text)`.

**3-gram F1** (both components). Text is first space-padded around every CJK character in the
U+4E00–U+9FFF range, then split on whitespace; Latin runs and numerals survive as single tokens.
Order-3 n-grams are formed by joining each consecutive triple with `-`. Matching uses multisets, so
repeated n-grams are counted with their multiplicities and no de-duplication is applied. With
`common` the multiset intersection size:

```
precision = common / |pred_grams|      recall = common / |gold_grams|
F1        = 2 · precision · recall / (precision + recall)
```

If both texts are shorter than three tokens, the term degrades to exact string equality (1.0 or
0.0). An empty candidate or reference yields 0.0.

**LLM-as-a-judge component**, range {0.0, 0.3, 0.7, 1.0}. Step-wise cumulative scoring over the
three pre-matched elements, with later steps skipped once an earlier one fails:

| Step | Element | Points |
|---|---|---|
| 1 | Error location matches | 0.3, else 0.0 and stop |
| 2 | Error reason equivalent | +0.4 (total 0.7), else hold 0.3 and stop |
| 3 | Corrected text consistent and free of new errors | +0.3 (total 1.0), else hold 0.7 and stop |

Equivalence is judged semantically: wording, diction and sentence structure may differ, provided
the same text span, the same core error point, and an equivalent professional correction are
identified. Replies are scored 0.0 unless a `<score>` value is parseable into the valid set.
The judge prompt is `prompts/LLM‑as‑a-judge prompt used during GRPO training.txt`; the backbone is
`deepseek-chat` through the OpenAI-compatible endpoint `https://api.deepseek.com`.

**Aggregation:**

```
total_reward = code_reward × 0.1  +  judge_reward × 9.0        ∈ [0, 10]
```

so the two signals stand in a 0.1 : 0.9 ratio. The judge term dominates deliberately: the 3-gram
F1 term is a surface-alignment proxy that a policy can raise by copying the reference wording,
whereas the judge responds to whether the correction is *logically* right.

Both stages also maintain an exponential moving average of the mean reward with α = 0.9, logged
alongside the per-step minimum and maximum; it is a training diagnostic and does not enter the
reward.

---

## 6. Hard-sample pool

A sample is **hard** when the best reward in its own rollout group stays below the threshold of
**9.0 on the 0–10 scale** (equivalently 0.9 normalised). Every reward call, each observed prompt is
added to the pool if hard and **evicted** if it later clears the threshold, so the pool tracks the
current weakness of the policy rather than accumulating a monotone history.

Group bookkeeping follows the trainer's batching semantics. TRL invokes the reward function once
per *generation batch*, and that batch holds `num_generations` copies of each of
`generation_batch_size ÷ num_generations` unique prompts, with dataset columns replicated to match.
With `generation_batch_size = num_generations = 4`, a batch carries exactly one unique prompt, so
the group maximum is the batch maximum. `core_reward_wrapper` in stage 4 reads the batch-maximum
reward and attributes it to the first replicated row; `6model_iteration.py` instead derives the
group size from the batch at runtime and indexes each group by its own prompt, so a raised
`generation_batch_size` cannot silently drop hard samples.

Pool identity is the `-*-`-joined five-tuple `error_text / reasoning / error_position /
specific_error_reason / corrected_content` — the same key used for the stage-4 dump.

**Stage 6 pool mechanics.** The pool is seeded from stage 4's hard-sample dump and extended at
each iteration with newly diagnosed samples, which are augmented at 1:5 before seeding the next
GRPO round. The accumulated augmented set is persisted to
`data/train/Hard_samples/iteration_augmented.json` and the live pool to
`data/train/Hard_samples/iteration_pool.json`, so a run can be resumed and audited.

**Convergence test**, evaluated after each iteration on the fixed validation split:

1. The mixed reward fluctuates by less than 3 % relative to the previous round, for two
   consecutive rounds; **and**
2. the hard-sample pool holds fewer than 5 % of the total sample pool.

Both conditions must hold. The first bounds round-over-round improvement, the second bounds the
residual difficulty of the data the policy still fails, so a run cannot converge by flattening the
reward while leaving a large hard core unaddressed.

---

## 7. Evaluation protocol

### 7.1 Metric

**Error Correction Accuracy (ECA)** is the primary metric:

```
ECA = N_correct / N_total × 100 %
```

with `N_total` the size of the isolated test set and `N_correct` the count of samples the
ECA-dedicated judge assigns a binary score of 1.0.

ECA is deliberately all-or-nothing: a sample is incorrect if **any** of the four pre-matched
elements — error position, error cause, reasoning process, corrected text — fails the consistency
check. Semantic equivalence rather than literal string matching is applied, so paraphrase does not
incur a penalty, but a sample cannot earn credit for getting three elements right. Only fully
accurate audit results are admissible in automotive-grade certification, which is what this
criterion mirrors.

**3-gram F1** on the corrected text is reported as a complementary fine-grained metric, using the
same construction as the training reward (§5.2). Because it admits partial matching, it is
systematically higher than ECA; the two are read together rather than as substitutes.

### 7.2 Judge decoupling

The ECA evaluator is fully decoupled from the training-phase judge:

| | Training judge | ECA judge |
|---|---|---|
| Backbone | `deepseek-chat` | `qwen3.8-max` |
| Scaling rule | step-wise cumulative {0.0, 0.3, 0.7, 1.0} | all-or-nothing {0.0, 1.0} |
| Prompt | `prompts/LLM‑as‑a-judge prompt used during GRPO training.txt` | `prompts/LLM‑as‑a-judge prompt used during computing ECA.txt` |

Both prompts share the same structural framework — three ordered checks over the same equivalence
criteria — but differ in backbone *and* in scoring rule, so the policy is never evaluated by the
signal it was optimised against. This is what rules out circular validation: the ECA judge cannot
reward the surface patterns the GRPO judge was tuned to accept.

The judge performs verification only. It does not re-audit the report, does not localise errors
independently, and does not consult clause text — it compares the model's four predicted elements
against the gold annotation. Keeping its task load low avoids degrading judgement with a second,
harder task.

### 7.3 Inference and score extraction

`code/test/test.py`, run from `code/test`:

- **Decoding**: greedy (`DO_SAMPLE = False`), `max_new_tokens = 1280`, one sample per record. Greedy
  decoding is required for the reported ECA to be reproducible and free of sampling noise.
- **Field extraction**: each of the four tags is read by locating its delimiters and stripping
  whitespace; a missing delimiter yields the sentinel `not find`. Any sample with an unparseable
  tag is counted as malformed and scored 0 — it is never silently repaired.
- **Judge call**: OpenAI-compatible client against Alibaba Cloud Model Studio (Bailian /
  DashScope). Credentials are read from the environment (`DASHSCOPE_API_KEY`, optionally
  `DASHSCOPE_BASE_URL` for a workspace-scoped MaaS endpoint); no key is stored in the repository.
  The client is constructed before the model load, so a missing credential fails fast instead of
  after a multi-minute model load. Each call is retried three times with exponential backoff, and
  the sample scores 0.0 if all retries are exhausted.
- **Verdict parsing**: numeric verdicts are normalised before comparison, so a judge writing
  `1`, `1.0` or `1.00` all map to 1.0 and `0`, `0.0`, `0.00` all map to 0.0. A `<score>` block
  containing several numbers — a judge echoing the prompt's value list, for instance — counts only
  an explicitly written `0.0` or `1.0`, and otherwise falls back to 0.0, the safe failure mode.
- **No-error records**: the 33 compliant reports in the test set have empty gold location and
  reason fields, which the prompt has no clause for. They are rendered to the judge as an explicit
  phrase (`none (this text is compliant and contains no error)`) rather than as empty strings, and
  are reported as their own `no_error` cohort alongside the `with_error` cohort, so an
  auditor-style "no error found" response is neither rewarded nor punished by a formatting
  accident.
- **Reporting**: alongside overall ECA, the summary reports the format compliance rate, the number
  of malformed outputs, ECA restricted to format-compliant outputs (which separates a formatting
  failure from a genuine auditing failure), and per-cohort ECA and format rates.
- **Write-through**: per-sample records — including the judge's raw reply, the full model response
  and all gold and predicted fields — are appended to `eca_results/eca_per_sample.jsonl` after
  every sample, so a crash or a killed job keeps its progress and judge outputs remain auditable
  against expert labels. The aggregate lands in `eca_results/eca_summary.json`.

---

## 8. Execution order and artifacts

Run from `code/train`, in order; the ECA script runs from `code/test`.

| # | Command | Produces |
|---|---|---|
| 1 | `python 1DAP.py` | `model/output/1DAP_output` |
| 2 | `python 2cold_start_SFT.py` | `model/output/2Cold_start_output` |
| 3 | `python 3grpo_format.py` | `model/output/3GRPO_format_output`, `reward.txt` |
| 4 | `python 4grpo_content.py` | `model/output/4GRPO_content_output`, `data/train/Hard_samples/hard_samples.txt` |
| 5 | `python ../../data/train/Hard_samples/txt2json.py` | `data/train/Hard_samples/origin_hard_samples.json` |
| 6 | `python utils/Rules_augmented_hardsamples.py` | `data/train/Hard_samples/agumented_hard_samples.json` |
| 7 | `python 5DAP_for_hardsamples.py` | `model/output/5DAP4HS_output` |
| 8 | `python 6model_iteration.py` | `model/output/6Iteration_output/iter_{1..3}`, `final/`, `iteration_report.json`, `iteration_pool.json`, `iteration_augmented.json` |
| 9 | `cd ../test && python test.py` | `code/test/eca_results/eca_per_sample.jsonl`, `eca_summary.json` |

Only stage 4 flushes a hard-sample dump; the format stage writes just its reward history. Step 6
may be run with `utils/LLM_augmented_hardsamples.py` instead — the two generators implement the
same 1:5 ratio and the same error categories.

Stage 6 initialises from the stage-5 output and runs the hard-sample loop internally over three
iterations, so it re-diagnoses and re-augments at each round rather than consuming steps 5–6 again;
per-iteration augmentation uses the same rule-based generator with `AUGMENT_VARIANT_NUM = 5` and
accumulates into `iteration_augmented.json`.

Model directory chain: `qwen2.5-14b-instruct` → `1DAP_output` → `2Cold_start_output` →
`3GRPO_format_output` → `4GRPO_content_output` → `5DAP4HS_output` → `6Iteration_output/final`.

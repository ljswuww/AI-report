# Closed-Loop Reinforcement Learning-Enhanced LLM for Intelligent Auditing of AEC-Q-Compliant Automotive-Grade Chip Test Reports

A six-stage training pipeline that turns `Qwen2.5-14B-Instruct` into an auditor of automotive-grade
chip test reports. The model reads a test report segment, decides whether it violates its governing
AEC-Q clause, localises the smallest error-containing span, states the cause, and returns the corrected
text — in one fixed four-tag response:

```
<reasoning>              reasoning process
<error_position>         smallest complete error-containing segment, or "none"
<specific_error_reason>  brief error cause, or "none"
<corrected_text>         complete revised text
```

Stages 3–6 optimise that behaviour with reinforcement learning, and stage 6 runs the whole
optimisation as a closed loop: each iteration diagnoses the samples the current policy still fails,
augments them, and folds them into the next round.

> **Data notice.** Automotive-grade chip test reports are company-proprietary documents, and the
> certification records they belong to are covered by confidentiality agreements. The complete
> dataset used in this work **cannot be released**, and the raw real-world audit records are withheld
> in full. What this repository ships is a small set of **expert-constructed simulated samples** and
> **de-identified excerpts** in the same schema, plus the AEC-Q reference standards and one worked
> example. See [§2](#2-data) for exactly what is and is not included.

---

## Contents

| Path | What it is |
|---|---|
| [`code/train/`](code/train/) | The six training stages and the ablation runner |
| [`code/train/utils/`](code/train/utils/) | Reward, prompts, hard-sample augmentation |
| [`code/test/test.py`](code/test/test.py) | ECA evaluation harness |
| [`data/`](data/) | The released data: simulated samples and de-identified excerpts |
| [`data source/`](data%20source/) | AEC-Q reference standards (public PDFs) |
| [`auditing example/`](auditing%20example/) | One worked auditing example, input to output |
| [`prompts/`](prompts/) | The system prompt and both LLM-as-a-judge prompts |
| `*_DETAILS.md` | Setup documentation: pipeline, baselines, ablations, judge validation |

---

## 1. Method: the six-stage pipeline

Every stage does full-parameter training of the same 14B checkpoint — no quantisation, no LoRA. Each
stage initialises from the previous stage's output directory, so the stages form one chain:

| # | Stage | Script | Initialises from | Consumes | Writes |
|---|---|---|---|---|---|
| 1 | Domain-adaptive pretraining (DAP) | [`1DAP.py`](code/train/1DAP.py) | `qwen2.5-14b-instruct` | DAP corpus | `1DAP_output` |
| 2 | Cold-start fine-tuning (CSFT) | [`2cold_start_SFT.py`](code/train/2cold_start_SFT.py) | `1DAP_output` | 50 core CoT samples | `2Cold_start_output` |
| 3 | Format-oriented GRPO | [`3grpo_format.py`](code/train/3grpo_format.py) | `2Cold_start_output` | format-stage RL set | `3GRPO_format_output` |
| 4 | Content-oriented GRPO | [`4grpo_content.py`](code/train/4grpo_content.py) | `3GRPO_format_output` | content-stage RL set | `4GRPO_content_output` |
| 5 | Hard-sample targeted optimisation | [`5DAP_for_hardsamples.py`](code/train/5DAP_for_hardsamples.py) | `4GRPO_content_output` | augmented hard samples | `5DAP4HS_output` |
| 6 | Closed-loop iterative optimisation | [`6model_iteration.py`](code/train/6model_iteration.py) | `5DAP4HS_output` | content-stage RL set + accumulated pool | `6Iteration_output/final` |

All outputs land under `code/model/output/`. The evaluated artifact is `6Iteration_output/final`.

- **Stages 1 and 5** train with an unmasked causal-LM objective over domain text (`labels = input_ids`):
  stage 1 over the AEC-Q corpus, stage 5 over the augmented hard samples.
- **Stage 2** is answer-only supervised fine-tuning: the prompt is conditioned on and masked out of the
  loss, so only the response tokens are scored.
- **Stages 3, 4 and 6** use TRL's `GRPOConfig` / `GRPOTrainer`. Stage 3 optimises format compliance
  alone; stages 4 and 6 optimise the hybrid reward below.
- **Stage 6** iterates: each round samples the current policy, thresholds its rollout rewards to
  diagnose hard samples, augments them 1:5, accumulates them, and trains again from that set.

Full per-stage hyperparameters, data construction and evaluation protocol are in
[`EXPERIMENT_DETAILS.md`](EXPERIMENT_DETAILS.md).

---

## 2. Data

### 2.1 Disclosure and access

The corpus this pipeline was built on comes from two sources that cannot both be published:

- **Real-world audit records** — actual automotive-grade chip test reports and the audit results
  produced from them. These contain company-proprietary test data (sample counts, measured
  parameters, Cpk values, process details, customer and part identifiers) and form part of
  industry-confidential certification documentation. They are covered by **binding non-disclosure
  agreements**, so they are not in this repository in any form. No amount of de-identification was
  considered sufficient for the raw records.
- **Expert-constructed samples** — CoT audit samples written by domain experts around the
  AEC-Q standards, reproducing the defect patterns that occur in real reports (clause-citation
  errors, test-parameter deviations, failure-criterion misuse, cross-chapter inconsistency) without
  reproducing any customer's document.

**What is released here is therefore a small, illustrative subset:** the expert-constructed simulated
samples, de-identified excerpts in the same five-field schema, the reference standards, and one worked
example. It is enough to run every script in the repository end to end and to see the exact input and
output contract. It is **not** the corpus behind any reported number — the scale, the provenance and
the difficulty distribution of the full set are not reproducible from this repository.

**What that means in practice.** A clone of this repository can execute the whole pipeline, but cannot
reproduce the reported results, because the training and evaluation data it needs is not here. Results
reported elsewhere were obtained on the full internal corpus, which remains with the data owners.

If you have your own licensed report corpus, the pipeline accepts it directly: every data path is
configurable ([§9](#9-configuration-reference)) and the schema is the five fields shown in §2.2.

### 2.2 What is released

| File | Records | Kind | Role |
|---|---|---|---|
| `data/train/DAP_data/few_shot_ft/*.json` | 6,853 | Simulated, built over the AEC-Q standards | Domain corpus for stage 1 |
| `data/train/DAP_data/few_shot_short_length/*.json` | 3,734 | Simulated, same source, short-length tiers | Domain corpus for stage 1 |
| `data/train/RL_data/Cold_start/cold_start_samples.json` | 50 | Expert-constructed | Stage 2 cold-start set |
| `data/train/RL_data/GRPO_format/RL_format_samples.json` | 465 | De-identified excerpts + expert-constructed | Stage 3 RL set |
| `data/train/RL_data/GRPO_content/RL_content_samples.json` | 652 | De-identified excerpts + expert-constructed | Stages 4 and 6 RL set |
| `data/train/Hard_samples/hard_samples.txt` | 30 | Produced by stage 4 | Its roll-out dump, in the wire format below |
| `data/train/Hard_samples/origin_hard_samples.json` | 30 | Produced by `txt2json.py` | The same records in JSON |
| `data/train/Hard_samples/augmented_hard_samples.json` | 30 | Produced by the augmentation script | Stage 5 training set |
| `data/test/test_samples.json` | 200 | De-identified excerpts + expert-constructed | Evaluation set |

The counts are of the files as shipped. Both are subsets of the internal corpus, not samples from it
in a statistical sense, so the shipped data does not support a difficulty-distribution claim.

**Schema.** Every CoT record carries the same five fields, and nothing else:

```json
{
  "error_text": "...",                  // the report segment, with the error present
  "reasoning": "...",                   // expert chain of thought
  "error_position": "...",              // smallest error-containing span, "" if compliant
  "specific_error_reason": "...",       // brief cause, "" if compliant
  "corrected_content": "..."            // the corrected report
}
```

Compliant reports are represented rather than excluded: their `error_position` and
`specific_error_reason` are empty and `corrected_content` equals `error_text`. The evaluation harness
renders those empty fields to the judge as an explicit no-error phrase and reports them as their own
cohort, so answering "none" on a compliant report is scored on its own terms.

**The other DAP folders.** `data/train/DAP_data/` also holds several earlier split variants
(`few_shot`, `few_shot1`, `few_shot_before`, `few_shot_mix`, `few_shot_semantic`). They are retained for
reference; `1DAP.py` merges only the two folders listed above.

### 2.3 Reference standards — `data source/`

The AEC-Q documents the domain corpus and the error taxonomy are built on, in their published form:

| File | Document |
|---|---|
| `AEC Q100-Rev J1.pdf` | AEC-Q100 Rev J1 — stress test qualification for integrated circuits |
| `AEC_Q100_Rev_H_Base_Document.pdf` | AEC-Q100 Rev H base document |
| `AEC-Q101 Rev - E.pdf` | AEC-Q101 Rev E — discrete semiconductors |
| `AEC_Q101_Rev_E_Base_Document.pdf` | AEC-Q101 Rev E base document |
| `AEC_Q100_QTP_Template_Rev_H.pdf` | AEC-Q100 qualification test plan template |

These are standards-body publications, not customer documents, and they can be redistributed as
published. They are the source of the clause text, parameter limits and units the task is defined
against, and of the knowledge that judgment in this domain requires — the pipeline learns it in
stage 1 rather than relying on retrieval at inference.

### 2.4 Worked example — `auditing example/`

[`auditing example/auditing example.txt`](auditing%20example/auditing%20example.txt) holds one real
auditing instance in full: the input report text, followed by the model's four-tag output, with the
error localised and the correction applied. `auditing example.png` is the same case rendered. It is
included so that the input-to-output contract is visible without running anything, and it shows the
error category that a rule engine cannot reach: the report states a sampling plan that is internally
consistent (Cpk 2.2 against a 1.67 limit, zero failures) and passes every threshold check, but the
sample was drawn from the wrong number of devices — a violation that exists only in relation to the
standard.

### 2.5 Building your own dataset

Nothing about the pipeline depends on the shipped samples. To run it on a corpus of your own:

1. Put your CoT data in the five-field schema of §2.2, one JSON array per stage's file.
2. Point the stages at it with the `ABLATION_*` environment variables of
   [§9](#9-configuration-reference) — no source edit is needed.
3. For stage 1, supply `{"text": ...}` records rendered with the system prompt, as `1DAP.py` expects.

The two files most likely to need replacing are `GRPO_content/RL_content_samples.json` (the RL split,
also the pool stage 4 diagnoses over) and `test_samples.json` (evaluation).

---

## 3. Reward design

The task is scored by a hybrid reward: a deterministic code component on a 0–10 scale, plus an
LLM-as-a-judge component on 0–1, weighted 0.1 : 0.9.

```
total_reward = code_reward × 0.1  +  judge_reward × 9.0        ∈ [0, 10]
```

**Format gate.** A response missing any of the four tags scores 0 for the whole reward and is not
content-scored. The output contract is enforced, not merely requested.

**Code component** (`code/train/utils/reward_utils.py`): `0.5` format credit, plus up to `8.0` for
3-gram F1 between the predicted and the gold error position, plus up to `1.5` for 3-gram F1 on the
corrected text, clamped to 0–10. A compliant report correctly answered "none" is given the full
position credit rather than scored against an empty reference.

**Judge component.** The judge compares the four elements the model produced against the gold
annotation. It performs no audit of its own — it does not localise errors or look up clauses — which
keeps its task bounded and keeps the signal about auditing quality rather than about the judge's
domain mastery. The training judge scores cumulatively (`0.0 / 0.3 / 0.7 / 1.0`, one point per passed
check).

**The measured metric is scored by a different judge.** Evaluation uses ECA — error-correction
accuracy, all-or-nothing (`0.0 / 1.0`) — computed by a separate judge with a different backbone and a
different scoring rule, so the policy is never evaluated by the signal it was optimised against.
3-gram F1 is reported alongside it as a complementary fine-grained metric. Both judges, both prompts
and the decoupling argument are specified in [`EXPERIMENT_DETAILS.md`](EXPERIMENT_DETAILS.md) §5 and
§7.

**The reward has a regression test.** `code/train/utils/test_reward_utils.py` pins the reward's
behaviour on fixed inputs, so a change to the n-gram construction or a weight cannot pass silently:

```bash
cd code/train/utils
python test_reward_utils.py        # or: pytest test_reward_utils.py
```

---

## 4. Environment

| Item | Value used |
|---|---|
| Base model | `Qwen/Qwen2.5-14B-Instruct` |
| Hardware | 3 × NVIDIA A100 80 GB |
| OS | CentOS Linux 7 |
| CUDA | 11.8 |
| PyTorch | 2.6.0 |
| Precision | BF16, TF32 enabled |
| Optimiser | `adamw_torch_fused` |
| Libraries | `transformers`, `datasets`, `trl`, `torch`, `numpy`, `openai`, `modelscope` (download) |

The repository ships **no dependency lockfile**, so pin these yourself; the versions above are what
the pipeline was run with. `HF_ENDPOINT=https://hf-mirror.com` is set in the training scripts and
`code/test/test.py`.

Download the base model into the path the stages expect:

```bash
cd code/model/base
bash download.sh                 # ModelScope; lands in ./qwen2.5-14b-instruct
```

### Credentials

Two judges are hosted API models, and they are configured differently on purpose:

- **Training judge** (`code/train/4grpo_content.py`, `6model_iteration.py`,
  `ablation_static_diagnose.py`) — the client is constructed in the source with a **placeholder key**.
  Substitute your own key there.
- **ECA judge** (`code/test/test.py`) — reads the key from the environment, so nothing is committed:

  ```bash
  export DASHSCOPE_API_KEY=sk-xxx
  # optional, if your workspace has a scoped endpoint:
  export DASHSCOPE_BASE_URL=https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
  ```

No credential of any kind is stored in this repository.

---

## 5. Running the pipeline

Every stage is a standalone script whose paths are relative to `code/train`, so **run them from that
directory**, in order:

```bash
cd code/train

python 1DAP.py                    # 1  domain-adaptive pretraining
python 2cold_start_SFT.py         # 2  cold-start fine-tuning
python 3grpo_format.py            # 3  format-oriented GRPO
python 4grpo_content.py           # 4  content-oriented GRPO  -> dumps hard_samples.txt

# hard-sample path: dump -> JSON -> 1:5 augmentation
python ../../data/train/Hard_samples/txt2json.py
python utils/Rules_augmented_hardsamples.py

python 5DAP_for_hardsamples.py    # 5  targeted optimisation on the augmented set
python 6model_iteration.py        # 6  closed-loop iteration -> 6Iteration_output/final
```

The two steps between stages 4 and 5 are the hard-sample path: stage 4 flushes its accumulated hard
samples to `data/train/Hard_samples/hard_samples.txt` with the `*|||*\n` record separator and `-*-`
between the five fields; `txt2json.py` converts that dump back into the JSON schema;
`Rules_augmented_hardsamples.py` expands each sample 1:5 and writes the set stage 5 trains on.

Each stage writes its checkpoint to `code/model/output/<stage>/` and can be re-run independently as
long as its input checkpoint exists. Stages 5 and 6 read the augmented set, so skipping the
augmentation step leaves them with nothing to load.

### Configuration reference

Every path and mode is overridable by environment variable, using the convention `ABLATION_<KEY>`
where dots in the key become underscores:

```bash
ABLATION_1DAP_MODEL_PATH=/path/to/another/checkpoint   python 1DAP.py
ABLATION_4GRPO_DATA_PATH=/my/own/rl_split.json         python 4grpo_content.py
ABLATION_TEST_MODEL_PATH=../model/output/6Iteration_output/iter_2   python ../test/test.py
```

With no `ABLATION_*` variable set, every script behaves exactly as it does in the commands above. The
resolution rule lives in [`code/train/ablation_config.py`](code/train/ablation_config.py); the full
key table is exercised by [`code/train/run_ablation.py`](code/train/run_ablation.py).

---

## 6. Evaluation

```bash
cd code/test
python test.py
```

`test.py` loads the checkpoint at `MODEL_PATH`, generates one greedy completion per test sample with
`max_new_tokens = 1280`, sends the four extracted elements to the ECA judge with the gold annotation,
and writes:

| Output | Contents |
|---|---|
| `eca_results/eca_per_sample.jsonl` | One record per sample: gold fields, the full model response, the four predicted elements, the judge's raw reply, the verdict |
| `eca_results/eca_summary.json` | Overall ECA, format compliance rate, ECA restricted to format-compliant outputs, and per-cohort figures |

Records are appended after every sample, so a killed job keeps its progress. Because the judge's raw
reply is retained per sample, any two runs can be compared item by item rather than only in aggregate
— which is what makes a disagreement inspectable rather than merely countable.

---

## 7. Ablations

`code/train/run_ablation.py` drives the six one-module-removed variants (`no-DAP`, `no-CSFT`,
`no-T-GRPO`, `no-HSO`, `no-HR`, `no-CIO`) plus `full`. It reimplements no stage: each step launches an
ordinary stage script as a subprocess with `ABLATION_*` variables set, so a variant's shape lives in
one table instead of in six edited copies of the pipeline.

```bash
cd code/train
python run_ablation.py --list                          # variants and their stage plans
python run_ablation.py --variant no-T-GRPO --dry-run   # print the plan, run nothing
python run_ablation.py --variant no-T-GRPO             # train it
python run_ablation.py --all                           # every variant, in sequence
python run_ablation.py --variant no-CIO --evaluate     # score its final checkpoint
```

Each variant writes to its own subtree under `code/model/output/ablation/<variant>/` so no two can
overwrite each other, and `--evaluate` gives each its own record directory. A step whose checkpoint
already exists is skipped, so an interrupted variant resumes. `--reuse-full` points the stages that are
provably identical to the full run's at that run's checkpoints instead of retraining them; it is
opt-in. Two variants need substitutions rather than omissions — `no-T-GRPO` requires a static
diagnosis pass over the RL split, and `no-HR` swaps the hybrid reward for its code component — both of
which are implemented, not optional extras. Variant definitions, boundary decisions and the
reasoning behind each substitution are in [`ABLATION_DETAILS.md`](ABLATION_DETAILS.md).

---

## 8. Companion documents

| Document | Covers |
|---|---|
| [`EXPERIMENT_DETAILS.md`](EXPERIMENT_DETAILS.md) | Environment, data construction, per-stage configuration, reward design, hard-sample pool, evaluation protocol |
| [`BASELINE_DETAILS.md`](BASELINE_DETAILS.md) | The six comparison methods: base model, rule-based engine, DAP+SFT+RAG, DAP+SFT+DPO, GLM-5.2-MoE + KG, and ours |
| [`ABLATION_DETAILS.md`](ABLATION_DETAILS.md) | The six one-module-removed variants, what "removing" a module means, and what is held constant |
| [`JUDGE_VALIDATION_DETAILS.md`](JUDGE_VALIDATION_DETAILS.md) | The human validation of the judge: sampling, expert annotation, binary mapping, agreement metrics |

---

## 9. Reproducibility notes

**What is deterministic.** The dataset shuffle in stages 3 and 4 uses `seed=42`, and the stage-6
train/validation split uses `VAL_SEED = 42`, so every iteration of the loop is scored on the same
validation samples. Test-time inference is greedy, so a given checkpoint scores the same way every
time. The hard-sample augmentation seeds `random` with 42.

**What is not.** The held-out splits taken inside stages 1, 2 and 5 are created without a seed and
therefore vary between runs; they serve as training-loss probes only and affect neither model selection
nor any reported figure. Judge replies come from a hosted model and are not bit-reproducible even at
temperature 0.

**What cannot be reproduced from this repository.** The data. See [§2.1](#21-disclosure-and-access):
the corpus behind the reported figures is under NDA and is not here, and the shipped samples are a
small illustrative subset rather than a sample of it.

**Known open items.** The evaluation set as documented is larger than the file shipped in
`data/test/test_samples.json`, and the test items carry no sub-set label, so a stratified split of the
evaluation set cannot be reconstructed from the data here. Both are noted where they matter, in
[`EXPERIMENT_DETAILS.md`](EXPERIMENT_DETAILS.md) §2.4 and
[`JUDGE_VALIDATION_DETAILS.md`](JUDGE_VALIDATION_DETAILS.md) §10.

---

## 10. License

MIT — see [`LICENSE`](LICENSE). The license covers the code in this repository. The AEC-Q documents
under `data source/` remain the property of their publishers and are included as published standards,
not as licensed material from this repository; the released sample data is included for research use,
and its terms are those of the data release rather than of the code license.

# Ablation Implementation Details

Setup details for the module ablation study. This document specifies how each ablated variant was
built and configured so that the comparison is reproducible; measured values are not repeated here.

The pipeline itself, its six stages, its data and its reward design are specified in
`EXPERIMENT_DETAILS.md`. The evaluation harness, judge and decoding settings are shared with every
other experiment and are stated there once (§7). This document adds only what is specific to the
ablations: what "removing" a module means operationally, what substitutes for it, and what each
variant is scored on.

---

## 1. Design

**One module at a time.** Six variants are built, each removing exactly one of the six core modules
while every other stage keeps the settings it has in the full pipeline. A seventh entry is the full
pipeline, unmodified. No variant removes two modules and no variant changes a retained stage's
hyperparameters, data or reward.

**"Removed" means skipped, not replaced.** A variant omits the named module's stage entirely rather
than substituting a different implementation of it. What changes is therefore always the
**initialisation edge** — which checkpoint the next surviving stage starts from — plus, in two
cases, the source of a downstream input that the removed stage used to produce (§3 and §6).

**No module is ablated by disabling it in place.** Switching off the judge inside the reward
function, or zeroing a stage's learning rate, would leave the surrounding machinery in place and
would not measure the module's contribution. The stage is removed from the run.

**Everything downstream still runs.** A variant is a complete pipeline, trained end to end, and its
final checkpoint is evaluated like any other. No variant is evaluated at an intermediate stage,
except where the removed module *is* the last one (§8).

**Naming.** Variants are labelled by the removed module: `no-DAP`, `no-CSFT`, `no-T-GRPO`,
`no-HSO`, `no-HR`, `no-CIO`, and `full` for the unmodified pipeline.

---

## 2. Variant matrix

| Variant | Module removed | Stages skipped | Substitution | Final checkpoint evaluated |
|---|---|---|---|---|
| `full` | — | — | — | `6Iteration_output/final` |
| `no-DAP` | DAP | 1 | — | `6Iteration_output/final` |
| `no-CSFT` | CSFT | 2 | — | `6Iteration_output/final` |
| `no-T-GRPO` | T-GRPO | 3 and 4 | static hard-sample diagnosis (§5) | `6Iteration_output/final` |
| `no-HSO` | HSO | 5, plus its diagnosis and augmentation | stage 6 runs with an empty pool seed (§6) | `6Iteration_output/final` |
| `no-HR` | HR | none | code-only reward in stages 4 and 6 (§7) | `6Iteration_output/final` |
| `no-CIO` | CIO | 6 | — | `5DAP4HS_output` |

Two modules occupy more than one stage, which is why two rows list more than one stage. T-GRPO is a
two-stage module, so removing it skips both GRPO stages together. HSO comprises hard-sample
diagnosis, its 1:5 augmentation and the targeted pretraining stage, so removing it skips the stage
and retires the data pipeline that feeds it. HR is the only module that is not a stage at all: it is
a component of the stage-4 and stage-6 reward, so its variant changes the reward rather than the
pipeline topology, and no stage is skipped.

The "initialisation edge" that changes in each variant follows from the table: the next surviving
stage starts from whichever upstream checkpoint is still produced.

---

## 3. `no-DAP`

The domain-adaptive pretraining stage is skipped and the pipeline begins at cold-start fine-tuning,
which now initialises directly from the released base checkpoint.

| | |
|---|---|
| Stage 1 | not run |
| Stage 2 init from | the base checkpoint, in place of `1DAP_output` |
| Stages 3–6 | unchanged, chained as usual |
| DAP corpus | unused in this variant |

Operationally this is a single change: the model path at the top of the cold-start script points at
the base checkpoint instead of the DAP output.

Note that the targeted continued-pretraining stage (stage 5) is **not** affected. Both stage 1 and
stage 5 train with a causal LM objective over domain text, but they are separate modules serving
different roles, and only the pretraining corpus stage is removed here.

**What it isolates.** Whether domain knowledge must be internalised before the task-specific stages
can work, or whether cold-start fine-tuning on task data can carry that load on its own.

---

## 4. `no-CSFT`

The cold-start stage is skipped. Stage 3 now initialises from `1DAP_output` rather than
`2Cold_start_output`.

| | |
|---|---|
| Stage 2 | not run |
| Stage 3 init from | `1DAP_output` |
| Stages 4–6 | unchanged |
| Cold-start samples | unused in this variant |

The model therefore reaches reinforcement learning with domain knowledge but no prior exposure to
the task paradigm or the four-tag output contract. The format stage still supplies a format-driven
signal, so the contract is not left untaught — what is missing is the supervised warm start on
expert trajectories.

**What it isolates.** Whether supervised adaptation to the task paradigm accelerates and stabilises
the subsequent RL stages, or whether RL alone reaches the same place.

---

## 5. `no-T-GRPO`

Both GRPO stages are skipped. Stage 5 initialises from `2Cold_start_output`.

| | |
|---|---|
| Stages 3 and 4 | not run |
| Stage 5 init from | `2Cold_start_output` |
| Stage 6 | runs, with the substitution below |
| Hard-sample dump | not produced — stage 4 normally writes it |

**Substitution required: the source of hard-sample diagnosis.** Hard samples are normally diagnosed
from the rewards observed during the content-GRPO rollouts. With that stage removed there are no
such rewards, so the pipeline has no pool to augment for stage 5 and no pool to seed stage 6 with.
Leaving both empty would confound the ablation with a change to HSO's and CIO's starting state.

Instead the pool is produced by **static diagnosis**: run the variant's current checkpoint greedily
over the RL training split, score each sample with the hybrid reward, and admit the samples whose
score falls below the same hard-sample threshold used in the full pipeline. The threshold, the
augmentation ratio and the pool key are all unchanged; only the origin of the scores differs — a
greedy pass over a fixed checkpoint, instead of grouped rollouts of a policy mid-update.

The variant then follows the normal hard-sample path: the statically diagnosed pool is augmented at
1:5, that set trains stage 5, and stage 6 seeds its accumulated set from it exactly as usual.

The substitute is implemented in `code/train/ablation_static_diagnose.py`. It imports the reward
from `code/train/utils/reward_utils.py` rather than restating it, so the pool cannot be selected by
a reward that has drifted away from the one the GRPO stages use. It writes its dump in exactly the
encoding stage 4 writes, so `txt2json.py` and everything after it are untouched by the
substitution. Two properties of the substitution are worth stating because they are not incidental:

- It decodes greedily, so each sample yields one completion. There is no group and no group maximum,
  which is the quantity stage 4 thresholds. The rule applied here is therefore "this sample's score
  is below the threshold", the single-rollout case of the same rule.
- It scores every sample of the RL split, which is the set stage 4 trains and diagnoses over. The
  split is not narrowed to the loop's held-out training side, because narrowing it would make this
  variant's pool cleaner than the full pipeline's rather than comparable to it.

**What it isolates.** The contribution of the reinforcement-learning stages to structured reasoning
and auditing quality, with the hard-sample module and the closed loop otherwise intact. The
substitution is deliberate and must be reported alongside the result: in this variant hard samples
are identified *before* any policy improvement has occurred, so the pool reflects the weaknesses of a
cold-start model rather than those of a partially optimised one.

---

## 6. `no-HSO`

The hard-sample module is removed. Stage 6 initialises from `4GRPO_content_output` and the data
pipeline that HSO feeds is retired along with it.

| | |
|---|---|
| Stage 5 | not run |
| Stage 6 init from | `4GRPO_content_output` in place of `5DAP4HS_output` |
| Hard-sample diagnosis, JSON conversion, 1:5 augmentation | not run |
| Stage 6 pool seed | empty |
| Stage 6 accumulated augmented set | empty |

**Module boundary.** Three things are removed together, because they constitute one module:

1. Diagnosis — thresholding the content-GRPO rollout rewards to decide which samples are hard.
2. Augmentation — expanding each diagnosed sample into five variants at 1:5.
3. Targeted pretraining — training a dedicated checkpoint on the augmented set (stage 5).

The augmented set is not a by-product of the pretraining stage. It is built by a separate
augmentation step that consumes the diagnosis dump, and it is read by both stage 5 and stage 6. So
removing the module removes the diagnosis and the augmentation, not merely the training stage —
otherwise the variant would drop the checkpoint while silently keeping the data path, and would
differ from the full pipeline by less than the module definition implies.

**On the boundary with CIO.** Stage 6 has its own hard-sample mechanism: each iteration re-diagnoses
hard samples from its own rollouts, augments them, and accumulates them for the next round. That
mechanism belongs to CIO and is **retained** here. What this variant removes is the standalone
diagnose-augment-pretrain pass that runs before the loop; the loop's own use of hard samples is a
different module's mechanism and stays.

The boundary is worth stating plainly because the two are easy to conflate. This variant is
*not* hard-sample-free: the loop still finds and augments hard samples, it just starts from an
empty pool with nothing inherited.

The empty start is enforced by a switch, not by deleting files: stage 6 seeds its accumulated
augmented set from the augmented hard-sample file unless `ABLATION_AUGMENTED_SEED_ENABLED=0`, which
is what this variant sets. Without it the loop would still pick up the file the removed module
produced and would silently import part of that module's effect back in. The pool itself needs no
switch — it is seeded from `iteration_pool.json`, and this variant gets its own path for that file,
so it starts empty because nothing has written it yet.

**What it isolates.** Whether a dedicated diagnose-augment-pretrain round earns its cost over and
above what the closed loop achieves with the same samples and the same threshold.

---

## 7. `no-HR`

The pipeline topology is unchanged. The hybrid reward is replaced in stages 4 and 6 by its
code-based component alone, so the LLM-as-a-judge term is never consulted during training.

| | |
|---|---|
| Stage 3 reward | unchanged (format reward carries no judge term) |
| Stage 4 reward | code-based component only |
| Stage 6 reward | code-based component only |
| Judge calls during training | none |
| Hard-sample threshold | unchanged |

**The substitution.** In the full pipeline the aggregated reward is

```
total_reward = code_reward × 0.1  +  judge_reward × 9.0
```

and the ablated variant drops the judge term and rescales the code term back onto the same 0–10
scale:

```
total_reward = code_reward
```

Both `code_reward` and the aggregate live on 0–10, so this substitution removes the semantic signal
without moving the reward scale. That is deliberate: the hard-sample threshold is defined in
absolute reward units, and rescaling the reward would change the diagnosis of which samples are hard
at the same time as it removes the judge — two variables at once. Keeping the scale fixed keeps the
pool's membership criterion comparable across variants, although the threshold naturally means
something stricter when the only component left is surface alignment.

**Evaluation is unaffected.** The judge that scores the test set is a separate evaluator with its
own backbone and its own all-or-nothing prompt (§7.2 of `EXPERIMENT_DETAILS.md`). Removing the
training judge does not remove the ECA judge, and every variant including this one is scored by the
same evaluator.

**What it isolates.** The contribution of semantic judgement to the training signal — specifically,
whether surface alignment alone is enough to select for corrections that are logically sound rather
than merely similar to the reference wording.

**Configuration chosen.** The substitution above keeps the format eligibility gate and the format
credit inside `code_reward`, on the grounds that those belong to the output contract rather than to
HR. The stricter alternative — scoring the corrected text with 3-gram F1 and nothing else, dropping
the gate too — removes the mechanism that enforces the four-tag contract, so a decline under that
configuration would conflate a lost reward signal with a lost format constraint. The first is
implemented; the second is available by a one-line change if a stricter reading is preferred. The
two are not interchangeable, so whichever is used must be stated with the result.

The branch is in `4grpo_content.py:RewardManager.core_correction_reward` and, for the loop, in
`utils/reward_utils.py:compute_mixed_reward`. Both read the mode from `ABLATION_REWARD_MODE`, which
`ablation_config.py` resolves; no reward mode is enabled by default, so an ordinary run is hybrid.

---

## 8. `no-CIO`

The closed-loop iterative stage is skipped and the pipeline stops after the hard-sample targeted
optimisation stage. The checkpoint produced by stage 5 is the evaluated artifact.

| | |
|---|---|
| Stage 6 | not run |
| Evaluated checkpoint | `5DAP4HS_output` |
| Hard-sample pool, augmented set, iteration report | not produced |

This is the only variant evaluated at a checkpoint other than the end of the full pipeline, because
the removed module is the final one. It measures the pipeline's state after a single targeted
optimisation round, before any iteration.

**What it isolates.** What the closed loop contributes beyond the single-round result — both the
iteration itself and the additional hard samples it diagnoses along the way.

---

## 9. What is held constant

Every variant shares, unchanged:

- The isolated test set, in full, with no filtering.
- The system prompt, the four-tag output contract, and the tag ordering requirement.
- Decoding at evaluation: greedy, `max_new_tokens = 1280`, one generation per sample.
- The ECA judge: backbone, all-or-nothing prompt, verdict parser, retry policy.
- The 3-gram F1 construction used both in the reward and as the auxiliary metric.
- The hard-sample threshold, the augmentation ratio and the pool key.
- The validation split seed, so all variants are compared on the same held-out samples.
- Malformed-output handling: a response missing any of the four tags scores 0, in every variant.
- The retained stages' hyperparameters, data and epoch counts.

Only the removed module's presence, its initialisation edge, and the documented substitutions
differ: static hard-sample diagnosis for `no-T-GRPO` (§5), an empty pool seed for `no-HSO` (§6), and
a code-only training reward for `no-HR` (§7).

---

## 10. Output layout and evaluation

Variants must not share output directories. Each writes to its own subtree so that no variant can
overwrite another's checkpoint or read it by accident:

```
model/output/ablation/
    full/  no-DAP/  no-CSFT/  no-T-GRPO/  no-HSO/  no-HR/  no-CIO/
```

Each subtree mirrors the full pipeline's layout, so inside a variant the stage outputs keep their
usual relative positions and only the root changes.

Every variant is then evaluated by pointing the evaluation script's model path at that variant's
final checkpoint and running it unmodified. The harness takes an arbitrary checkpoint, so the
evaluation path is identical across variants — nothing about the scoring changes between them, and
no variant is scored with a tailored prompt or a second judge.

Because each run writes one record per test sample and retains the judge's raw reply, any two
variants can be compared sample by sample rather than only in aggregate. That is what allows a
difference between two variants to be inspected — for instance, whether a drop comes from a handful
of samples whose output format broke, or from a broad decline in correction quality.

---

## 11. Reproduction order and cost

Each variant is a full pipeline run. None can be derived from another's checkpoint, because the
ablation changes where the chain starts, so all six must be trained.

| Variant | Stages trained | Extra cost | Note |
|---|---|---|---|
| `full` | 1–6 | — | Already required by the main experiments; reused as the reference row |
| `no-DAP` | 2–6 | — | Saves the most expensive single stage: full-parameter 14B pretraining over the domain corpus |
| `no-CSFT` | 1, 3–6 | — | Skips only the cheapest training stage, so cost is close to `full` |
| `no-T-GRPO` | 1, 2, 5, 6 | one greedy pass over the RL split for static diagnosis | Skips both GRPO stages; stage 6 still iterates three times |
| `no-HSO` | 1–4, 6 | — | Cost close to `full` |
| `no-HR` | 1–6 | — | Training cost close to `full`, but no judge API calls during training |
| `no-CIO` | 1–5 | — | Cheapest overall: no iteration loop |

**Stage reuse.** Where a variant and the full pipeline reach a stage through the same
initialisation edge with the same data and settings, that stage's output is identical and does not
need to be retrained. In practice:

- `no-HR`, `no-HSO` and `no-CIO` all share stages 1–4 (and `no-CIO` also shares stage 5) with the
  full run, because the modifications are all downstream of the content-GRPO stage.
- `no-CSFT` shares stage 1 with the full run, since removing the cold-start stage does not change how
  DAP is trained.
- `no-DAP` shares nothing: every downstream stage starts from a different checkpoint.
- `no-T-GRPO` shares stages 1 and 2 only; its stage 5 and stage 6 diverge because the GRPO stages
  that would otherwise sit between them are gone.

Reuse is an optimisation, not a requirement. If a variant is trained from scratch, the result is the
same, provided the seeds and settings match — the shared stages are deterministic apart from the
unseeded internal held-out splits noted in `EXPERIMENT_DETAILS.md` §1, which affect only a
training-loss probe.

**How the variants are run.** `code/train/run_ablation.py` encodes the plan above and drives it. It
reimplements no stage: each step launches the ordinary stage script as a subprocess with `ABLATION_*`
environment variables set, and those variables are resolved at the top of each script by
`code/train/ablation_config.py`. A stage run under the runner therefore executes the same code path
as a direct run, and the variant's shape lives in one table rather than in six edited copies of the
pipeline. With no `ABLATION_*` variable set, every script behaves exactly as it did before.

```
cd code/train
python run_ablation.py --list                         # the variants and their stage plans
python run_ablation.py --variant no-T-GRPO --dry-run  # print the plan, run nothing
python run_ablation.py --variant no-T-GRPO            # train it
python run_ablation.py --all                          # train every variant in sequence
python run_ablation.py --variant no-CIO --evaluate    # score its final checkpoint
```

Three properties of the runner are worth knowing before using it:

- **Resumption.** A step whose checkpoint already exists is skipped, so an interrupted variant
  resumes where it stopped, and a failed stage stops its variant rather than launching the next stage
  against a checkpoint that was never written. `--force` re-runs everything.
- **`--reuse-full` is off by default.** It points the stages listed as shareable at the full run's
  existing checkpoints and skips retraining them. It is opt-in because it couples a variant's result
  to a run that happened elsewhere, and only the stages whose inputs and settings are provably
  identical are eligible. Note that the dump-conversion and augmentation steps are not skipped by
  it, since whether they feed a reused stage is not known until the plan is resolved; they are pure
  text transforms and cost seconds.
- **The two substitutions are not optional extras.** `no-T-GRPO` cannot run without static
  diagnosis — the stages that would have produced its pool are gone — and its diagnosed pool empty
  is reported as a warning, because an empty pool would silently also remove HSO and make the
  variant mean something other than what its name says.

Every variant's evaluation differs only in which checkpoint the harness is pointed at. The harness
reads `ABLATION_TEST_MODEL_PATH` and `ABLATION_TEST_RESULT_DIR`, so `--evaluate` gives each variant
its own record directory and no two variants can overwrite one another's per-sample output.

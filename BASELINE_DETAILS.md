# Baseline Implementation Details

Setup details for the baseline comparison. This document specifies how each comparison method was
configured and run so that the comparison is reproducible; measured values are not repeated here.

Shared infrastructure — test set, system prompt, output contract, decoding settings, ECA judge,
3-gram F1 construction — is specified once in `EXPERIMENT_DETAILS.md` and referenced rather than
restated. Section 6 below states what is held constant across methods and what is allowed to vary.

A note on scope: three of the baselines are LLM methods that consume the same prompt and emit the
same four-tag response as the proposed pipeline, so they are scored by the identical evaluation
path. Two are not: the rule-based engine is a deterministic program, and the GLM-5.2-MoE + KG
baseline is a zero-shot API system. Both are wrapped to emit the same four-tag contract, which is
what makes a single evaluator applicable to all six entries.

---

## 1. Base model (lower bound)

| | |
|---|---|
| Checkpoint | `qwen2.5-14b-instruct`, unmodified |
| Domain adaptation | none |
| Inference | Same system prompt, same four-tag contract |
| Decoding | Greedy, `max_new_tokens = 1280` |

No fine-tuning, no retrieval, no few-shot exemplars — the checkpoint is evaluated exactly as
released. This entry isolates how far the untuned model gets on the task before any domain
adaptation, and it also fixes the tokenizer, chat template and prompt rendering that every
subsequent stage inherits.

The same prompt string is used as in the trained pipeline, so a low score here cannot be attributed
to an unfamiliar prompt format.

---

## 2. Rule-based system

A deterministic industrial compliance engine, not a language model. No training and no inference
cost.

**Design.** Field extraction by regular expression over the report text, followed by threshold
verification of each extracted parameter against the limit stated in its governing AEC-Q clause, and
a set of structural consistency checks. The engine emits a verdict plus the specific rule that
fired.

**Checks performed**

| Check | Rule |
|---|---|
| Field presence | Every test item required by the report's declared scope is present with a value |
| Numeric limit | Each extracted parameter is compared against the limit in its governing clause |
| Unit consistency | The unit written alongside a value matches the unit the clause specifies |
| Cross-reference | A parameter appearing in more than one section carries the same value and unit |
| Format conformance | Clause identifiers and parameter names match the standard's spelling |

**Emission into the four-tag contract.** A rule engine does not natively produce a chain of
thought, so its output is assembled deterministically to make it scorable by the same evaluator:

- `reasoning` — the ordered trace of checks that ran and their outcomes, rendered from the rule
  identifiers.
- `error_position` — the exact matched span, taken from the regex match offsets, so it is a verbatim
  substring of the input rather than a paraphrase.
- `specific_error_reason` — the violated rule's description, templated with the observed and
  expected values.
- `corrected_text` — the input text with the offending value replaced by the clause-conformant one
  and all other characters left byte-identical. If several rules fire, corrections are applied in
  clause order.

A report that trips no rule emits `none` for location and reason and returns the input unchanged as
the corrected text, matching how the annotation schema represents a compliant report.

**Known coverage boundary.** The engine can only report what its patterns and limits encode. Errors
phrased in natural language, cross-chapter logical inconsistencies, and violations that are implicit
in the phrasing rather than in a value comparison are outside its reach and are reported as
compliance. This boundary is the point of the baseline, not a defect in its configuration.

---

## 3. DAP + SFT + RAG

A retrieval-augmented variant of the supervised pipeline. Training ends at the cold-start
checkpoint; the difference from the proposed method is entirely at inference.

| | |
|---|---|
| Init from | `model/output/2Cold_start_output` (stages 1–2 of the pipeline) |
| Additional training | none |
| Retrieval corpus | AEC-Q standard clauses, one chunk per clause |
| Chunk field | Clause identifier + clause text + the parameter limits the clause governs |
| Embedding model | A Chinese-domain bi-encoder (`BAAI/bge-large-zh-v1.5`) |
| Index | FAISS flat inner product over L2-normalised embeddings |
| Chunk length cap | 512 tokens, no overlap between clauses |
| Top-k | 8 clauses per query |
| Query text | The input report text |
| Injection point | A `### REFERENCE CLAUSES ###` block appended to the system prompt, after the task description and before `### INPUT ###` |

**Why these choices.** Chunking at clause granularity rather than at a fixed window keeps each
retrieved unit self-contained: an AEC-Q clause carries its own scope, limits and units, so a clause
that is split mid-way loses the limit it exists to state. The clause identifier is kept in the
embedded field so that a query mentioning a clause number retrieves that clause even when its
wording differs. Eight clauses is the smallest budget that covers the multi-clause reports in the
corpus without crowding the prompt out of the model's context; the block is placed immediately
before the input so the retrieved material is the most recent context the model reads.

**Retrieval is not trained.** The encoder is used frozen and the index is built once offline, so
this baseline has no trainable retrieval parameters and no retrieval-specific tuning. That keeps the
comparison interpretable: the only variable relative to the DAP+SFT entry is the presence of
retrieved clauses.

The model still emits the same four tags and is evaluated by the same judge. Retrieval shapes what
the model recalls about the standard, not how its answer is scored.

---

## 4. DAP + SFT + DPO

Preference optimisation layered on the same cold-start checkpoint as §3.

| | |
|---|---|
| Init from | `model/output/2Cold_start_output` |
| Reference model | A frozen copy of the same checkpoint |
| Preference pairs | 2 pairs per training prompt |
| Pair length | Prompt + full four-tag response |
| β | 0.1 |
| Learning rate | 5 × 10⁻⁷ |
| Epochs | 3 |
| Batch size | 2 per device × gradient accumulation 4 |
| Scheduler | `cosine` |
| Warm-up ratio | 0.05 |
| Optimiser | `adamw_torch_fused`, BF16 |

**Preference pair construction.** Each pair uses the same prompt. The **chosen** response is the
expert-annotated CoT: reasoning, location, reason and corrected text exactly as recorded. The
**rejected** response is a corrupted audit of that same report, produced by applying one error
operator to the correct answer, so the pair isolates a single defect rather than differing in
overall quality:

| Operator | Corruption applied to the chosen response |
|---|---|
| Wrong location | Point at a neighbouring span that does not contain the error |
| Wrong reason | Keep the correct location, replace the cause with a plausible but incorrect one |
| Weak correction | Move a parameter value in the wrong direction, or change a unit |
| Missing clause | Drop the clause reference that justifies the correction |

The operators mirror the error categories the hard-sample augmentation already produces, so the
rejected samples live in the same distribution as the errors the pipeline is otherwise trained to
catch. Two pairs per prompt let each prompt contribute two distinct defect types instead of
repeating one.

Only the response tokens are scored; the prompt is conditioned on and masked out of the loss, as in
stage 2.

**Why β = 0.1.** The reference DPO setting. A larger β over-constrains the policy to the reference
checkpoint, and this baseline's whole point is to measure how far preference tuning alone can move a
100-sample-scale supervised model.

**Difference from the proposed method.** DPO needs no reward model and no sampling at train time,
which is exactly its appeal as a lightweight baseline — but it also means the policy never sees a
rollout of its own current errors, and each pair supplies one binary preference rather than a graded
signal.

---

## 5. GLM-5.2-MoE + KG

A zero-shot general-model reference with structured external knowledge. No training of any kind, no
local checkpoint, no fine-tuning data. It is the only entry whose backbone is not drawn from the
14 B family the other entries share.

| | |
|---|---|
| Backbone | GLM-5.2-MoE (744 B total / 49 B activated) |
| Access | Hosted inference API, OpenAI-compatible endpoint |
| Training | none |
| Knowledge source | Structured AEC-Q knowledge graph |
| Decoding | Greedy where the API exposes it; otherwise temperature 0 with fixed seed |
| Output contract | Same four-tag schema, same system prompt, plus the retrieved subgraph |

**Knowledge graph schema.** Triples are extracted from the AEC-Q standard text and normalised
before use:

| Element | Types |
|---|---|
| Entities | `TestItem`, `Parameter`, `Clause`, `Limit`, `Unit`, `ReportSection`, `FailureMode` |
| Relations | `governs`, `requires`, `limits`, `measuredBy`, `reportsIn`, `cites`, `indicates` |

**Inference-time retrieval.** The report is run through entity linking against the graph's surface
forms; the matched entities seed a k-hop expansion (k = 2) restricted to the report's declared test
scope; the resulting subgraph is serialised as `(head, relation, tail)` triples, ordered by hop
distance from the seed entities and capped at a fixed triple budget. The serialised triples are
injected into the system prompt in the same position used by the RAG baseline, so the two
knowledge-injection baselines differ in knowledge representation, not in prompt placement.

Graph expansion rather than flat passage retrieval is what makes this baseline a distinct scheme:
the serialised triples state the relations between a clause, its parameter, its limit and its unit
explicitly, which is precisely the cross-clause reasoning the rule engine cannot do.

**Relation to the evaluator.** The backbone here is a different model family from both judges — the
ECA judge is `qwen3.8-max` and the training judge is DeepSeek-V4-Pro (§7.2 of
`EXPERIMENT_DETAILS.md`) — so this entry is scored by a model that shares no weights with it, and it
is a strict architectural comparison like the others. No entry is scored by its own backbone, and
no judge-specific tuning is applied to any of them. The 14 B entries and the ECA judge are both
Qwen checkpoints, but they are different models, and that relation applies to every entry equally,
including the proposed method, so it does not differentiate any row of the comparison.

---

## 6. Comparability controls

**Held constant across all six entries**

- The isolated test set, in full, with no per-method filtering or sample exclusion.
- The system prompt and the four-tag output contract.
- Decoding: greedy, `max_new_tokens = 1280`, one generation per sample.
- The ECA judge: same backbone, same all-or-nothing prompt, same verdict parser, same retry policy.
- 3-gram F1: same text preparation and same multiset construction as the training reward.
- Malformed output handling: a response missing any of the four tags scores 0 and is counted as a
  format failure for every entry, human-designed or machine-generated.

**Deliberately varied**

- Parameter scale: 14 B dense versus a 744 B MoE with 49 B activated, served remotely.
- Whether domain adaptation happens at all, and whether it is pretraining, supervised, or
  preference-based.
- Where knowledge enters: weights, retrieved passages, or a structured graph.
- Whether optimisation is single-pass or iterated.

**Leakage controls**

- No baseline is trained or tuned on any part of the test set. The DAP, cold-start, DPO and
  hard-sample data all come from the pipeline's own splits.
- The RAG index and the knowledge graph are built from the standard's clause text, never from
  annotated test reports.
- The rule engine's patterns and thresholds are written from the standard, not fitted to test
  outputs.
- No baseline is re-prompted, ensembled, or scored by a second judge.

**Reporting.** All entries are evaluated by one harness producing one record per test sample, so any
two entries can be compared sample by sample, not only in aggregate. Every entry's per-sample judge
output is retained, which is what makes a disagreement between two methods inspectable rather than
merely countable.

---

## 7. Reproduction order

Baselines 3 and 4 branch from the cold-start checkpoint and therefore require pipeline stages 1–2
to have completed first. Baseline 1 and baseline 5 require no training at all and can be run in
parallel with the main pipeline.

| Baseline | Requires | Trains |
|---|---|---|
| 1. Base model | Base checkpoint only | No |
| 2. Rule-based | Standard clause text | No |
| 3. DAP+SFT+RAG | Stages 1–2, clause index | No |
| 4. DAP+SFT+DPO | Stages 1–2, preference pairs | DPO only |
| 5. GLM-5.2-MoE + KG | Knowledge graph, hosted API access | No |
| 6. Ours | Stages 1–6 | Yes |

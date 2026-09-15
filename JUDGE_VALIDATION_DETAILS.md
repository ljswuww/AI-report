# Judge Validation Details

Setup details for the judge reliability and human validation experiment. This document specifies how
the validation set was drawn, how the human reference was produced, how each machine judge's output
was turned into a binary label, and which metrics were computed; measured values are not repeated
here.

The evaluation protocol itself — the ECA criterion, the two judge prompts, the judge backbones and
the parsing rules — is defined once in `EXPERIMENT_DETAILS.md` §7.2 and referenced rather than
restated. This document covers only what that section does not: the human validation procedure and
the agreement analysis built on it.

The experiment exists because an LLM-scored metric is only as credible as the scorer. Two specific
risks motivate it: the ECA judge may have preferences of its own that the pipeline has learned to
satisfy, and an evaluator related to the training loop would make the reported result circular. The
design below addresses both by making the reference a human one and by testing the two judges against
each other as well as against it.

---

## 1. What is being validated

Three separate claims are tested, and they need separate evidence:

1. **The human reference is stable.** Two domain experts labelling independently must agree closely
   enough for their consensus to be usable as a gold standard. Without this, a judge's agreement
   with the reference measures nothing.
2. **The ECA judge is a valid substitute for expert scoring.** This is the claim that carries the
   reported ECA figure: if the judge agrees with experts about as often as experts agree with each
   other, then the metric is measuring the auditing quality rather than the judge's idiosyncrasies.
3. **The training reward and the reported metric are not the same instrument.** The training judge
   shapes the policy; the ECA judge scores it. If their verdict patterns were interchangeable, the
   reported number would be partly a restatement of the objective, and no amount of agreement with
   humans would remove that.

---

## 2. Sampling design

**Source and size.** The validation set is drawn from the isolated test set that every comparison
shares, which holds 400 items. Half of it — 200 items — is annotated and used here. No sampled item is
used in training in any stage, and no item is drawn from the DAP corpus, the CoT supervision data or
the hard-sample data.

The reported metric is computed over the full 400; this experiment validates the instrument on 200 of
them. The two therefore rest on different item counts — 400 for the metric, 200 for the agreement
figures — and the annotated half is not a separate evaluation set. It is a stratified sample of the
very set the metric is reported on, which is what lets the agreement result be read as evidence about
that metric rather than about an unrelated subset.

**Why half.** Labelling the full test set would double the expert time for a sharper estimate of a
quantity that is already bounded by the agreement the two experts reach with each other (§3), and the
half that is drawn is stratified, so the estimate covers the difficulty range of the whole set. The
bound this leaves is stated in §9: the figures describe 200 items, and the confidence interval around
each kappa is a function of that.

**Stratification.** The test set is not homogeneous, so the sample is stratified into the same three
subsets the test set is documented as carrying (§2.4 of `EXPERIMENT_DETAILS.md`):

| Stratum | Content | Share of the 400 |
|---|---|---|
| Real-world test report cases | Report segments taken from actual automotive-grade chip test reports | 180 |
| Expert-constructed hard cases | Cases written by domain experts to exercise the error categories the report auditing task targets | 130 |
| Out-of-distribution unseen scenarios | Items whose phrasing or setting falls outside the patterns the training data covers | 90 |

Stratifying rather than sampling uniformly is what makes the sample representative of the *difficulty*
range: the hard-case stratum is under a third of the test set but a much larger share of the cases
where a judge is most likely to disagree with a human expert, and a uniform draw would under-weight
exactly those items.

**Selection.** Within each stratum the items are drawn at random, with allocation proportional to
stratum size, and the draw is seeded so the sampled subset is fixed and can be reused identically by
every comparison. Proportional allocation over the three strata above fixes the per-stratum counts of
the sample, and those counts are part of the design rather than an outcome to be read off afterwards —
they should be recorded with the sampled indices, since a draw that happens to over-sample the
out-of-distribution stratum would report a lower agreement for reasons that have nothing to do with
the judge. Every pair in §7 is scored on the same subset, so no pair has an easier or harder set of
items than another.

**Unit of sampling.** One item is one complete case: the erroneous text plus its gold annotation. It
is not one field. The all-or-nothing criterion is defined over the whole item, and splitting it into
per-field judgements would change what the human and the machine are both being asked.

**Compliant reports must be represented in proportion.** Some test items are reports that contain no
error, where the correct audit is `none` for location and reason and the input unchanged as the
corrected text. These are a distinct decision — never spotting an error is a different failure from
misidentifying one — and the human makes that decision natively while the judge prompt has no branch
for it (§7.3 of `EXPERIMENT_DETAILS.md` describes the remedy: empty gold fields are rendered as an
explicit no-error phrase before the judge sees them, and the cohort is reported separately). The
stratified draw must preserve their share,
and their agreement is reported as its own cohort rather than folded into the pooled figure.

**The sampled subset is not training data.** Nothing about the validation set is fed back into any
stage; the items are read once, labelled, and compared.

---

## 3. Human annotation protocol

**Annotators.** Two senior experts in the AEC-Q domain, working independently. Neither is involved in
building or tuning the pipeline.

**What they see.** For each sampled item, the same three things the judge is given: the erroneous
text, the audited output being evaluated (its four elements), and the gold annotation. They do not
see any judge's score, any judge's reasoning, or each other's labels.

**What they decide.** A single binary verdict per item, applying the same all-or-nothing ECA criterion
that the evaluation uses: the audit is **correct** when the located span, the stated cause and the
corrected text are all right and the correction introduces no new error; otherwise **incorrect**. The
labels are therefore commensurable with the judge's output by construction — both parties answer the
same question under the same rule, which is what makes an agreement statistic meaningful rather than
merely arithmetic.

**Independence before consensus.** The two sets of labels are produced without discussion, and
Cohen's kappa is computed on this initial pair. That figure is the inter-annotator agreement, and it
serves as the ceiling for every other comparison in §7: judges cannot be expected to agree with a
consensus reference more often than the two humans who produced it agreed with each other in advance.

**Consensus.** Items where the two experts disagreed are then re-examined jointly against the standard
and resolved to a single label, giving each item one gold verdict. Two properties of this step are
worth recording, because they are easy to leave implicit:

- The gold standard is post-consensus, not independent of the annotators. It is a considered human
  judgement, and it inherits the ceiling above — it is not external ground truth.
- The number of items that reached consensus this way should be reported alongside the result. It is
  the size of the set on which the reference was negotiated rather than merely recorded, and a
  disagreement statistic computed mostly over negotiated items rests on a weaker reference than one
  computed over items both experts labelled the same way.

**Label propagation.** The labels attach to the model outputs that the judges scored, item by item.
The responses are not regenerated for the validation, so the human labels and the judge verdicts
describe the same objects and a disagreement can be traced to a specific response.

---

## 4. The judges under comparison

Both judges are the production scorers, called with their production configuration. Nothing is
re-tuned, re-prompted or given extra context for this experiment.

| | ECA-dedicated judge | Training-phase judge |
|---|---|---|
| Role | Scores the final reported metric | Supplies the reward during stages 4 and 6 |
| Backbone | `qwen3.8-max` (2.4 T total / 95 B activated) | DeepSeek-V4-Pro (1.6 T total / 49 B activated) |
| Endpoint | Hosted API, OpenAI-compatible (`code/test/test.py`) | Hosted API (`code/train/4grpo_content.py`, `code/train/6model_iteration.py`) |
| Prompt | All-or-nothing (`prompts/LLM‑as‑a‑judge prompt used during computing ECA.txt`) | Cumulative (`prompts/LLM‑as‑a‑judge prompt used during GRPO training.txt`) |
| Aggregation | One failing step ends the judgement at 0.0 | Each passing step adds points; a failing step ends the judgement at the level reached |
| Output | `{0.0, 1.0}` | `{0.0, 0.3, 0.7, 1.0}` |
| Parse fallback | Unparseable reply → 0.0 | Unparseable reply → 0.0 |

**Model identifiers.** The ECA judge is requested by the id `qwen3.8-max`. The training judge is
requested as `deepseek-chat` against `api.deepseek.com`, the provider's chat alias, which resolves to
the DeepSeek-V4-Pro version named above; the code therefore carries the alias while this document
names the version, and the two refer to the same judge. The identifiers are what to look up when
either judge is re-run.

The two prompts ask the *same* question. Both run the same three ordered checks — error location,
then error reason, then corrected text — and both use the same equivalence criteria, under which
different wording is accepted provided the located span, the core cause and the corrected content
match. What differs is the aggregation: the ECA judge scores the answer as a unit, while the training
judge scores it as three accumulating steps. That is the intended design — the reward must be graded
during learning, and the reported metric must be all-or-nothing — but it is also the reason the
judge-versus-judge comparison has to be read carefully (§7, §9).

In training, the training judge's reply is consumed as the judge component of the hybrid reward in
`EXPERIMENT_DETAILS.md` §5.2. In the ECA evaluation, the ECA judge's reply is consumed directly as the
per-sample verdict. Neither judge performs an audit of its own: both compare the elements the model
already produced against the gold annotation, and neither localises errors independently or consults
clause text. That keeps the judge's task bounded and keeps the comparison about auditing quality
rather than about the judge's domain mastery.

---

## 5. Making the two judges comparable: the binary mapping

Cohen's kappa is defined on a categorical scale, so the training judge's four-level output has to be
collapsed before it can enter the comparison. The mapping used is:

| Judge | Raw output | Binary label |
|---|---|---|
| ECA-dedicated | 1.0 | correct |
| ECA-dedicated | 0.0 | incorrect |
| Training-phase | 1.0 | correct |
| Training-phase | 0.0, 0.3, 0.7 | incorrect |

**Why 1.0 is the cut for the training judge.** 1.0 means all three checks passed, which is the same
condition the ECA criterion states for `correct`. Cutting anywhere lower would label an audit
"correct" while the ECA judge, applying its rule, calls it incorrect, and the two judges' labels would
then be measuring different things.

**The consequence, stated plainly.** The mapping is a full-mark threshold, so what is being validated
for the training judge is *whether it awarded full marks*, not how well its graded scores track human
severity. The two judges are not symmetric inputs to the comparison: the ECA judge's binary output is
native to its prompt, while the training judge's is derived. Both are valid, but a disagreement
between them can originate in the threshold rather than in the two models' judgement, and the
judge-versus-judge figure should be read as evidence about the decision rules rather than as a
measurement of how differently the two models "think".

**No partial credit anywhere.** Nothing in this experiment uses the 0.3 or 0.7 levels as a score. If a
graded comparison is wanted later, it must be a separate analysis with a weighted kappa, since an
unweighted kappa over two scales of different granularity would treat a 0.7 and a 0.0 as identical
errors.

**Malformed replies.** Both parsers return 0.0 when a judge reply carries no readable verdict, so a
judge-side parsing failure enters the comparison as `incorrect`. This is the same safe failure mode
the pipeline uses, but it means the agreement figures include any judge that failed to answer; the
count of such replies should be checked before the result is read, and it is recoverable because both
harnesses retain the raw reply.

---

## 6. Metrics

**Cohen's kappa (primary).** Reported for every pair in §7. It measures agreement on the same items
after removing the agreement expected by chance:

```
kappa = (p_observed - p_expected) / (1 - p_expected)
```

The chance term is what makes it the primary metric here. Under an all-or-nothing criterion the
outcome is skewed — a large majority of items are `incorrect` for most methods, and a majority are
`correct` for a strong one — so two raters can agree on a high proportion of items while having almost
no real agreement beyond what the base rates already imply. Raw accuracy cannot distinguish that case
from genuine agreement; kappa can. Conventional descriptive bands exist for it, but the figures are
reported as numbers, with the agreement ceiling of §3 alongside them, rather than characterised by a
label.

**Overall accuracy (secondary).** The proportion of items whose binary label matches the reference's.
It is reported because it is directly interpretable — it is the fraction of items the judge and the
human would have handled identically — and because it is the quantity that connects to the reported
ECA figure, which is itself a proportion. It is secondary because it is not chance-corrected.

**Both are computed on identical item sets.** The same sampled items, the same model responses, the
same binary rule, for all four pairs. A pair's figures are therefore comparable to any other pair's.

---

## 7. Comparisons performed, and what each one supports

| Pair | Establishes |
|---|---|
| Expert 1 ↔ Expert 2 | The agreement ceiling. Chance-corrected agreement between the two independent labelings, computed before consensus |
| ECA judge ↔ human gold standard | The validity of the reported metric. This is the comparison the reported figure rests on |
| Training-phase judge ↔ human gold standard | The validity of the reward signal, and that the objective the policy optimised was itself aligned with expert judgement |
| ECA judge ↔ training-phase judge | That the two scorers are not interchangeable |

The last row is the decoupling check, and its interpretation is narrow on purpose. A figure below the
two judges' respective agreement with humans is consistent with the two evaluation systems using
different decision rules, which is what the design intends: the policy is not optimised against the
instrument that reports it. It does not follow that the two judges are independent in the statistical
sense — they are different models under different prompts by construction, and §5's mapping
guarantees some disagreement mechanically. What the row supports is the narrower claim that the
verdicts of the reported metric are not produced by the decision rule that shaped the policy.

Two supporting observations belong with this table and can be read off the same records without new
labelling:

- **The ordering of agreement levels.** A judge whose agreement with humans is close to the experts'
  own agreement with each other is performing within the noise of the reference; one far below it is
  introducing disagreement of its own.
- **Where the disagreements fall.** Because every pair is scored on the same items, disagreements can
  be inspected case by case: whether they concentrate in one stratum, on the compliant-report cohort,
  or on the items the experts themselves had to negotiate. A pooled figure hides all of that.

---

## 8. Held constant across the whole experiment

- The sampled item set, the model responses being judged, and the binary rule — identical for every
  pair.
- The judge prompts: the same text the pipeline and the evaluation use in production, with no
  tailoring for the validation.
- The judge configuration: backbone, endpoint, retry policy and verdict parser as configured for the
  judge's production role.
- The raw judge replies, which are retained so that any disagreement can be read back rather than only
  counted.
- The three judged elements — location, reason, corrected text — and the equivalence criteria that
  define them.
- Human independence: the annotators never see judge output, and the judges never see the human
  labels. The gold labels are established before the judge comparison is run.

---

## 9. Caveats that must accompany the result

- **Scale.** The 200-item sample is large enough for the agreement estimate to be meaningful but not
  for it to be sharp; the confidence interval around each kappa should be reported with it, and read
  before any difference between two kappas is treated as a difference.
- **Pooling hides the strata.** A pooled kappa over a stratified sample is a weighted average across
  strata of very different difficulty, and it can mask a judge that is reliable on real report cases
  and unreliable on the hard and out-of-distribution strata. Per-stratum figures should be reported.
- **The reference is a ceiling, not ground truth.** The gold standard is the experts' post-consensus
  label. Agreement with it measures agreement with a considered human judgement, not correctness
  against the standard, and no judge can exceed the agreement the two experts had before negotiating.
- **The mappings are not symmetric.** The training judge's binary label comes from a full-mark
  threshold on a graded scale (§5). Its kappa is a statement about that derived label, not about the
  graded signal.
- **The compliant-report cohort is a different decision.** Items with no error require the model to
  decline; this is a distinct behaviour with its own error mode, and its agreement should be reported
  separately rather than absorbed into the pooled figure.
- **Agreement is not correctness.** Two raters, one human and one machine, can agree on a reading of
  the standard that is itself wrong. The experiment bounds the judge's deviation from expert practice;
  it does not certify the practice.
- **No full-set annotation was performed.** Labelling all 400 items would cost far more expert time
  than it would add confidence, so the validation rests on the 200-item stratified sample, and the
  figures describe that sample. The stratified draw is what makes them informative about the whole
  test set; it does not make them measured on it.

---

## 10. Status in this repository

**No script for this experiment is committed.** The sampling, the expert labelling, the consensus step
and the agreement computation were carried out outside the repository, and this document records the
procedure so that the setup is reproducible rather than only the outcome.

What the repository does provide is the input the analysis consumes. `code/test/test.py` writes one
record per test item and retains the judge's raw reply alongside the four predicted elements and the
gold annotation, so the objects being compared are recoverable from a completed evaluation run. The
two judge prompts are held in `prompts/` and mirrored in `code/train/utils/reward_utils.py` (training)
and in `code/test/test.py` (ECA), which is what makes §4's claim that no prompt was tailored for the
validation checkable.

One discrepancy is worth recording rather than smoothing over: the test file in this repository,
`data/test/test_samples.json`, holds 200 records, while the test set described in §2 holds 400. The
file is therefore a subset or a partial export, and which of the two it is has to be settled before
§2's draw can be reconstructed from it — a 200-item draw from a 400-item set and a 200-item test set
are not the same object, and only the first is what this document describes.

Three things would have to be added before the experiment could be re-run from this repository:

1. **A recorded mapping from the file to the test set.** Which of the 400 items the 200 records
   correspond to, so that the annotated half can be located in the data rather than inferred.
2. **A stratum label on the items.** The test data carries the five gold fields and nothing else, so
   the three strata of §2 exist only in the sampling as performed, and the sampled subset's
   per-stratum counts are recorded nowhere. Without a `subset` field on each item — or a recorded
   sample manifest giving the drawn indices, their strata and the seed — the sampled subset cannot be
   reconstructed, and a re-run would produce a different sample rather than reproduce this one.
3. **A sampling and agreement script.** The draw, the binary mapping of §5 and the kappa computation
   of §6 are each small and deterministic, and none of them is in the repository today.

The measurement the experiment yields is not a property of the pipeline's code but of the judges'
behaviour on a fixed set of items, so the item set and the labels, not the scripts, are what determine
whether a rerun agrees.

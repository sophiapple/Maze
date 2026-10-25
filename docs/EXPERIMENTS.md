# Included experiments

## Common setting

The three published cohorts use nonthinking Qwen3.5-4B and four complete exemplars per target prompt.
All decisions in the archived planning policies use input-side features, not correctness labels or realized output lengths.
The exported records contain one evaluated deployment outcome per question and arm.
A three-request deployment is therefore one correctness record, not three independent questions.

| Cohort | Questions | Included arms | Role | Maze policy |
|---|---:|---:|---|---|
| `gpqa99_v206` | 99 | 10 | Representative development cohort from GPQA Diamond | Four-head energy-based prompt choice, executed once or repeated in three requests |
| `mmlu32_v160` | 32 | 3 | Development cohort from MMLU-Pro | Calibrated input-only utility choice, repeated in three requests |
| `gpqa32_v166` | 32 | 4 | Post-hoc cached GPQA operating point | Frozen v121 selected prompt, three actual v134 requests; retrieval, Raw MV8, and legacy v108 planner retained |

The original experiment identities are retained in the configuration and provenance files.
The cohorts are not pooled into a larger accuracy estimate.
They are not substitutes for the full benchmark populations used elsewhere in Maze research.

## Inference policies

**Single-call:** one completion from the chosen prompt.

**SC3:** three distinct physical requests, with distinct fixed seeds, for the same complete prompt; one vote per question.

**Raw MV$N$:** $N$ requests using separately sampled random prompt paths; one vote per question.

**Retrieval SC3:** the retrieval-selected prompt is repeated in three requests.

For these compact configurations, invalid parsed answers do not vote, and tied valid answers are resolved by first occurrence in the frozen response order.
All-invalid responses remain incorrect; their cost is retained.
The saved ledger holds the evaluated final correctness, not the raw responses needed to independently reconstruct those votes.

## Cost accounting

`processed_tokens = input_tokens + output_tokens + query_encoder_tokens`.
Target counters sum over every physical request in a deployment.
Encoder processing is recorded once per query where present in the experiment's accounting.
Offline bank/system encoding is separate.
Unknown end-to-end latency, training cost, and complete serving cost are not inferred from token counts.
Token ratios are measured component ratios, not hardware-independent latency claims.

The GPQA three-call Maze comparison with Raw MV8 combines 71/99 versus 69/99 correct deployments with fewer physical requests and fewer recorded processed tokens.
The exact token ratio is available in the machine-readable report.
The MMLU-Pro comparison with Retrieval SC3 has 25/32 versus 24/32 correct deployments at the same three-call budget.

## Reconstructed comparisons

See [the comparison table](../results/reproduced/comparisons.md), [CSV](../results/reproduced/comparisons.csv), and [complete JSON report](../results/reproduced/summary.json).
All 21 comparisons are declared in [configs/results.json](../configs/results.json).
They include each Maze policy against every named non-Maze control in its cohort. Nine positive contrasts are highlighted in the README; all 21 remain visible. These cohorts and variants remain selected development evidence, not an exhaustive survey of all project methods or settings.
All data used in these comparisons are included, and every question in each declared arm is retained.

## Record schema

```json
{
  "cohort": "gpqa99_v206",
  "question": "q001",
  "arm": "Maze_SC3",
  "correct": 1,
  "calls": 3,
  "input_tokens": 7026,
  "output_tokens": 4635,
  "query_encoder_tokens": 247,
  "processed_tokens": 11908,
  "invalid_calls": 0,
  "truncated_calls": 0
}
```

This example is an actual included numeric record.
Question labels are cohort-local ordinals, not public dataset question identifiers.
Provenance binds the export to its source records without disclosing benchmark text or answers.

## GPQA32 reconstruction and exposure

The v166 operating point was defined after prior aggregate outcomes, by voting the three already completed, fixed v121/v134 selected-prompt requests once. Its original one-call mean over three draws is unchanged. It is a post-hoc development point, not a new prospective or representative experiment. All 32 questions are retained, including six incorrect Maze votes. The earlier v108 planner is also reconstructed from its three same-prompt calls as a disclosed legacy control.

The four policies produce 128 deployment outcomes from 544 request observations. There are 487 distinct cached request IDs; 57 observations reuse a request appearing in another arm. Per-policy deployment costs include every consumed call, while physical-cache accounting deduplicates IDs. The deduplicated token subtotal covers only these exported calls, not complete historical research acquisition. Opaque answer classes are assigned consistently within each question; null classes abstain, and ties use earliest nonnull occurrence. Correctness is still a recorded source field.

GPQA32 and GPQA99 share 13 original query identities, published as an ordinal mapping in [cohort_overlap.json](../evidence/development/cohort_overlap.json). The cohorts use different banks and development histories, so the smaller cohort is not independent confirmation of the larger one. Across cohorts there are 163 question memberships and 150 distinct source IDs. Multiple policies and comparisons reuse the same outcomes.

The GPQA99 ledger contains ten complete arms, including retrieval-one-call, retrieval-SC3, and uniform-Raw-SC3. The README highlights positive contrasts while all control results remain in the full report. The cohorts and variants were chosen after outcomes were known; no claim of prospective selection is made.

## Fresh-generation recipe differs from the paper

The closed GPQA32 plan used temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, repetition penalty 1.0, and a 16,384 output-token cap. The paper's default setting is temperature 0.3 and a 4,096 cap. Do not substitute the development configuration or any generic demo for exact paper replication. Historical checkpoint/feature/renderer/target bindings must be supplied for a model rerun.

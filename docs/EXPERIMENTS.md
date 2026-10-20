# Included experiments

## Common setting

The two published cohorts use nonthinking Qwen3.5-4B and four complete exemplars per target prompt.
All decisions in the archived planning policies use input-side features, not correctness labels or realized output lengths.
The exported records contain one evaluated deployment outcome per question and arm.
A three-request deployment is therefore one correctness record, not three independent questions.

| Cohort | Questions | Included arms | Role | Maze policy |
|---|---:|---:|---|---|
| `gpqa99_v206` | 99 | 7 | Representative development cohort from GPQA Diamond | Four-head energy-based prompt choice, executed once or repeated in three requests |
| `mmlu32_v160` | 32 | 3 | Development cohort from MMLU-Pro | Calibrated input-only utility choice, repeated in three requests |

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
The seven comparisons are declared in [configs/results.json](../configs/results.json).
They are a selected set of observed-positive contrasts, not an exhaustive survey of all available methods or settings.
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

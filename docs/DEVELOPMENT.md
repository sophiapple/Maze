# Developing the Maze project

The project separates reusable computation from frozen experiment definitions.
A paper is one research milestone, not a namespace for every future implementation.

## Core boundaries

| Interface | Responsibility | Must not consume |
|---|---|---|
| `FeatureStore` | Node/view features, input identities, encoder recipe | Query answers or response outcomes |
| `NumpyEnergy`-compatible scorer | `config`, `project(states)`, `score_projected(cache, indices)` | Generated answers, evaluation correctness, realized output length |
| `allocate` | Turn ranked paths into physical request slots | Evaluation labels |
| `Generator.generate` | Execute one frozen request and return `Generation` | Evaluation answer file |
| `grade` | Parse completed responses, compare with separate labels | Authority to change prompt selection or reissue calls |

The feature store, scorer, and generator can be replaced independently.
A new representation should have a new recipe identity.
A new inference policy must state its candidate budget, call budget, repetition counts, and tie rule.

## Add a scorer variant

Implement the scorer interface and add a configuration under `configs/scorers/`.
Keep input dimensions, normalization, boundary handling, and readout semantics explicit.
Add a numerical test against an independent expression or a preserved implementation.
Do not silently replace scalar reduction with vector reduction, or zero-start features with an encoded system node.
For trainable scorers, export only parameters that were actually fitted and record the query identities used in fitting and validation.

## Add a target adapter

Return `Generation(text, input_tokens, output_tokens, finish_reason, response_model)`.
The input is a frozen request dictionary with complete messages, target identity, decoding settings, and seed.
Do not retry inside the adapter or discard invalid/length-capped generations.
Use `None` when a cost counter is unknown.
Never store credentials in a configuration or response record.

## Add an experiment

Use a new stable experiment identifier and store its configuration before evaluating it.
Record the benchmark split, bank, encoder, feature recipe, scorer, proposal, allocation, target mode, and evaluator.
Keep the complete question cohort for each reported arm.
Publish the numeric evidence needed to reconstruct a claimed measurement, respecting the source data's redistribution terms.
Distinguish result reconstruction from independent regrading and fresh model execution.
An observed gain on one cohort is not automatically a claim for another model, dataset, or allocation policy.

## Test and release

```bash
python -m unittest discover -s tests -v
python scripts/reproduce_results.py --output runs/check-results
python scripts/audit_repository.py
```

Preserved scorer files are kept under `reproduction/compact/` and bound to their original identities.
Modify the reusable implementation, not the preserved copy, when adding new behavior.
Version experiment configurations separately from the Python API.
Update the result manifest only after reviewing a deliberate data or configuration change.
See [publishing](PUBLISHING.md) for creating an independent repository under a new account.

# Using Maze

## 1. Install and check the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
maze --help
maze demo --output runs/demo
maze train-demo --output runs/train-demo
```

The two demos are synthetic integration examples, not additional measured results.
Every output directory is new; commands refuse to overwrite an existing result.
On Windows, activate the environment using `.venv\Scripts\activate`.

## 2. Prepare authorized data

Obtain benchmark data under its own access terms and preserve its original split boundaries.
The converter reads local JSONL only; it neither downloads nor republishes benchmark questions.

```bash
python scripts/prepare_dataset.py --input data/raw/train.jsonl \
  --benchmark mmlu_pro --split train --output data/train
python scripts/prepare_dataset.py --input data/raw/validation.jsonl \
  --benchmark mmlu_pro --split validation --output data/validation
python scripts/prepare_dataset.py --input data/raw/test.jsonl \
  --benchmark mmlu_pro --split test --output data/test
```

Adapters accept MMLU-Pro, GPQA, GSM8K, MATH-500, AIME, or canonical records.
They produce unlabeled `queries.jsonl` and a separate `answers.jsonl`; only the training split produces `bank.jsonl`.
Training queries may also occur in the training bank, but their own exemplar is excluded from every path.
Validation/test questions must be excluded from the bank.
Near duplicates require dataset-specific checks beyond the normalized-text identity check supplied here.

A canonical exemplar has `id`, `question`, `answer`, optional `rationale`, and optional `choices`.
A query accepts only `id`, `question`, and optional `choices`.
The query loader rejects embedded answers or outcome fields.

## 3. Collect offline supervision

Prepare fixed local-edit candidate pools for training-only queries:

```bash
python scripts/propose_local_edits.py --bank data/train/bank.jsonl \
  --queries data/train/queries.jsonl --shots 4 --count 32 \
  --output data/train/candidates.json
```

Copy [offline_labels32.json](../configs/policies/offline_labels32.json) to `configs/local/offline-labeling.json`.
It fixes `selection: "uniform"`, `policy: "topk"`, `calls: 32`, `pool_size: 32`, and `purpose: "offline_training"` so every proposed candidate receives one labeling request.
Set `model` and `model_revision` to your target's actual serving identity and freeze the decoding configuration.
This is an offline labeling budget, not a claim of 32-call online Maze inference.

```bash
maze prepare --bank data/train/bank.jsonl --queries data/train/queries.jsonl \
  --system data/system.txt --config configs/local/offline-labeling.json \
  --candidates data/train/candidates.json --output runs/train-plan.json

# The following step executes real model requests and may incur cost.
export MAZE_API_BASE_URL='http://127.0.0.1:8000/v1'
maze collect --plan runs/train-plan.json --output runs/train-collection --allow-api-calls

python scripts/observations_from_run.py --plan runs/train-plan.json \
  --events runs/train-collection/events.jsonl --answers data/train/answers.jsonl \
  --split train --kind choice --output data/train/observations.jsonl
```

Repeat with disjoint validation queries, still using the training exemplar bank.
The converter evaluates each physical response and averages repeated outcomes for the same query/path.
The first path in the frozen request plan is the fixed reference for that query.
Local-edit training retains correctness preferences between locally related paths; the fixed chain proposal does not adapt its size based on the evaluation outcome.

For a remote compatible endpoint, use HTTPS and set `MAZE_API_KEY` in the environment.
Credentials are not written to plans or results.
No request is automatically retried: an interrupted call remains unresolved until the operator checks the server record.
After a process crash, remove a stale `.collector.lock` only after confirming no collector remains active; the unresolved-call check still applies.

## 4. Encode and train

Install the optional encoder dependencies:

```bash
python -m pip install -e '.[encoder]'
cat data/train/queries.jsonl data/validation/queries.jsonl > data/fit-queries.jsonl
cat data/train/observations.jsonl data/validation/observations.jsonl > data/fit-observations.jsonl

maze encode --bank data/train/bank.jsonl --queries data/fit-queries.jsonl \
  --system data/system.txt --model YOUR_ENCODER --revision YOUR_FIXED_REVISION \
  --layers 8,16,24,32 --start-mode zero_start_question_nodes \
  --device cuda --dtype bfloat16 --output data/fit-features

maze train --features data/fit-features --observations data/fit-observations.jsonl \
  --config configs/scorers/compact_vector.json --output runs/fitted-planner
```

Choose hidden-state tuple indices that actually exist in the selected encoder.
Index 0 is generally the embedding output; the final returned hidden state can include model-specific normalization.
The provided configurations use width 2560 as a concrete example; change `dim`, `views`, and the feature recipe together for another encoder.
The frozen encoder uses full-token mean pooling and float32 feature storage in this fresh workflow, with no silent node truncation.
These settings define a fresh feature recipe, not a substitute for unavailable archived tensors.

The training output contains a non-executable NPZ weight archive, a configuration/metadata JSON, and a loss history.
Only the planner receives gradients.
The independent scalar and full-QA configurations are trained through the same command with their respective node encoding, view count, and objective settings.

## 5. Freeze evaluation requests before generation

Encode evaluation queries with the same encoder revision and feature recipe but without their answers:

```bash
maze encode --bank data/train/bank.jsonl --queries data/test/queries.jsonl \
  --system data/system.txt --model YOUR_ENCODER --revision YOUR_FIXED_REVISION \
  --layers 8,16,24,32 --start-mode zero_start_question_nodes \
  --device cuda --dtype bfloat16 --output data/test-features

maze prepare --bank data/train/bank.jsonl --queries data/test/queries.jsonl \
  --system data/system.txt --features data/test-features \
  --checkpoint runs/fitted-planner --config configs/local/evaluation-repeat3.json \
  --output runs/evaluation-plan.json
```

Create `evaluation-repeat3.json` by copying [repeat3.json](../configs/policies/repeat3.json) and setting the actual target revision.
Repeat `--checkpoint` to use equal-weight consensus of compatible fitted heads.
The planner checks normalized-question overlap with checkpoint fitting and validation identities.
It serializes complete prompts and request seeds before any generation begins.
Choose service context limits that fit these complete prompts; this client never trims exemplars to fit a budget.

## 6. Execute and evaluate separately

```bash
maze collect --plan runs/evaluation-plan.json --output runs/evaluation-collection --allow-api-calls

maze grade --plan runs/evaluation-plan.json \
  --events runs/evaluation-collection/events.jsonl \
  --answers data/test/answers.jsonl --kind choice --output runs/evaluation.json
```

Use `numeric` for simple exact numeric answers, or `exact` for exact string matching.
Symbolic mathematical equivalence requires a task-specific evaluator; the generic exact/numeric parsers must not be relabeled as a full MATH evaluator.
All requests must be accounted for before a cohort is evaluated.
Invalid/truncated replies remain charged and are never silently replaced.
If the provider omits token usage, unknown costs remain null rather than being estimated from characters.

The runtime adapter is tested with a local mock server, not represented as a fresh real-model validation of these public benchmark settings.
A real model run produces new measurements under its own protocol.

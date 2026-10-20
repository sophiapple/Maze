# Examples

`maze demo --output runs/demo` exercises feature indexing, path scoring, repeated-prompt allocation, durable collection, and separate grading without any model download.
`maze train-demo --output runs/train-demo` additionally fits a tiny planner to invented targets, exports an NPZ checkpoint, and plans a disjoint synthetic query.

Both examples are software fixtures, not real benchmark evidence.
The published empirical records are under `evidence/development/` and are reproduced by `scripts/reproduce_results.py`.

Run `python examples/use_planner.py` for a small Python API example.

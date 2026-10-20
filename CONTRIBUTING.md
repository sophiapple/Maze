# Contributing

Contributions should make a specific part of Maze easier to use, understand, or evaluate.
Examples include an encoder adapter, a prompt-path scorer, a task evaluator, a call-allocation policy, or a complete experiment with reconstructable evidence.

Keep changes focused and include tests.
State which configuration the change affects and preserve previous configuration semantics.
Run the test suite and repository audit before opening a pull request.
A new result should identify its cohort, comparator, denominator, call budget, and exact reproduction path.
Do not describe a synthetic fixture or an aggregate-table check as a real benchmark rerun.

Do not commit credentials, unlicensed data, gated question text, raw private responses, or executable serialized model objects.
Use the NPZ/JSON interfaces for newly exported planner weights.
Report dataset and model access requirements in the experiment documentation.

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for interface contracts.

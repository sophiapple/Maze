# Data, checkpoints, and model access

## Included

This source tree includes the Maze implementations, independent references, preserved compact scorer code, tests, configurations, a paper pipeline image, and numeric records for the declared development experiments.
The 1,214 published outcome records contain cohort-local ordinal IDs, binary evaluated correctness, call counts, and token/counter measurements.
The GPQA32 supplement also includes opaque answer-equivalence classes and reused-call ordinals. These do not disclose answer text.
They contain no benchmark question text, gold answers, raw target responses, model weights, or credentials.

## Supplied by the user for new model runs

Fresh experiments require an authorized exemplar bank and query split, a frozen encoder and tokenizer, target-model access, and either a fitted scorer or the offline observations needed to train one.
The training and collection commands support these steps.
The original real-benchmark checkpoints, encoded-state bundles, and response archives are not included in this package.
The source archive therefore supports exact saved-outcome reconstruction, but not an asset-complete regeneration of the original target calls.

The [compact replay entry point](../reproduction/compact/replay.py) validates compatible independently supplied bundles and rejects mismatched feature or scorer identities.
A bundle digest identifies bytes; it does not grant redistribution rights.
No placeholder checkpoint is silently substituted when a real one is missing.

## Original sources

Models and datasets remain subject to their own terms and access conditions:

| Asset | Source |
|---|---|
| Qwen model family | [Qwen on Hugging Face](https://huggingface.co/Qwen) |
| GPQA | [Original dataset repository](https://huggingface.co/datasets/Idavidrein/gpqa) |
| MMLU-Pro | [Original dataset repository](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| GSM8K | [Original dataset repository](https://huggingface.co/datasets/openai/gsm8k) |
| MATH-500 | [Dataset repository](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| AIME | [Mathematical Association of America](https://maa.org/) |

Use the exact revision and applicable terms for the assets in your run.
Do not place gated benchmark content, restricted checkpoints, API keys, or private responses in a public pull request.
Store these assets outside Git or under ignored local directories.

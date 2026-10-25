# Camera-ready evidence map

This release binds the author's **Maze-NIPS26-Oct20-v0/main.tex** (SHA-256 `489de5afc295394e2497401607de671ea9185e585af6e26d01867f86b38d0f7a`). [Table excerpts](../evidence/paper/table_excerpts.tex) preserve all 15 original tables, including captions and qualifications. [Transcribed tables](../evidence/paper/tables.json) retain every row; [main-table CSV](../results/paper/main_results.csv) and [60 contrasts](../results/paper/main_contrasts.csv) are generated from them.

| Question | Evidence in the paper | Interpretation |
|---|---|---|
| Does input-side planning improve single-call accuracy? | Single-call table: 15 model–task cells; every Maze point exceeds Original, pFLAT, and EASE | Paper-reported prompt sensitivity at a fixed target-call budget |
| Can planning improve call efficiency? | Multi-call table: MultiMaze16 exceeds MV32 in all 15 cells | 50% fewer generation calls; token and latency costs remain separate |
| Is selection distinct from aggregation? | Shared 1,600-path pool: Maze top16 improves on random16 with the same aggregation | Oracle pick is diagnostic; within each selected set it remains above majority voting |
| Does structure help beyond a generic ranker? | Cached-representation rankers and matched-capacity sequence Transformer | Maze exceeds the matched Transformer means by 1.6, 2.9, and 1.6 pp on MMLU-Pro, GPQA, and MATH500 |
| How variable are the outcomes? | Five complete runs and paired intervals | Reported standard deviations are not standard errors; the one-call AIME–EASE interval crosses zero |
| Is online planning overhead small? | Qwen3-8B/MMLU-Pro, RTX4090: 15.8s Maze vs 15.5s Original; 254.6s MultiMaze16 vs 483.2s MV32 | Reported serving measurements exclude offline bank encoding and training; no new latency measurements here |
| Does it transfer to black-box targets? | GPT-5.1 and Gemini2.5Pro: zero-shot source scorer, adapted scorer, and eight-call policies | Adapted settings use 32 non-test queries / 512 completions per benchmark; do not call them zero-shot |
| What are the boundaries? | Component ablations, explored-pool diagnostics, capacity choices, encoder swaps, domain transfer, budget sweep, and split accounting | Preserve every control and uncertainty; pool diagnostics do not verify all theorem assumptions |

## Verify

```bash
python scripts/reproduce_paper.py --output runs/paper
python scripts/verify_release.py
# Optional: verify the full author-provided source as well.
python scripts/reproduce_paper.py --paper /path/to/main.tex --output runs/paper-source-check
```

These operations verify transcription and arithmetic, not the provenance of the paper's raw measurements. Original model checkpoints, exact benchmark features, per-run predictions, and response archives remain necessary for full benchmark regeneration. The [development evidence](EXPERIMENTS.md) is a separate set of scoped compact experiments and must not be relabeled as replication of these paper tables.

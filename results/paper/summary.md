# Camera-ready results: source-verified transcription

These are author-reported paper values. This command verifies transcription and arithmetic; it does not rerun the benchmarks.

| Paper comparison | Positive model–task cells | Gain range (pp) | Target calls |
|---|---:|---:|---:|
| Maze vs Original | 15/15 | +3.04 to +15.02 | 1 / 1 |
| Maze vs pFLAT | 15/15 | +0.99 to +5.88 | 1 / 1 |
| Maze vs EASE | 15/15 | +1.00 to +5.23 | 1 / 1 |
| MultiMaze vs Majority vote | 15/15 | +1.17 to +16.66 | 16 / 32 |

The 15 cells in each row cover three targets × five benchmarks. Comparisons share cells and are not independent replications.
All 105 main-table values, including random-proposal oracle values, are in `main_results.csv`. Oracle pick is diagnostic and sees a different proposal set from the main MultiMaze policy.
The complete 15 source tables retain matched-pool controls, ranker ablations, variability, latency, transfer, diagnostics, and split accounting in `evidence/paper/`.

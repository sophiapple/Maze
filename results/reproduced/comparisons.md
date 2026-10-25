| Cohort | Maze policy | Baseline | Maze accuracy | Baseline accuracy | Gain | Calls | Token ratio |
|---|---|---|---:|---:|---:|---:|---:|
| GPQA Diamond99 | Single-call Maze | Random prompt, 1 call | **65.66%** | 62.63% | +3.03 pp | 1 / 1 | 1.058× |
| GPQA Diamond99 | Single-call Maze | Random-path MV2 | **65.66%** | 63.64% | +2.02 pp | 1 / 2 | 0.506× |
| GPQA Diamond99 | MultiMaze, repeated prompt | Random-path MV3 | **71.72%** | 66.67% | +5.05 pp | 3 / 3 | 1.016× |
| GPQA Diamond99 | MultiMaze, repeated prompt | Random-path MV4 | **71.72%** | 69.70% | +2.02 pp | 3 / 4 | 0.762× |
| GPQA Diamond99 | MultiMaze, repeated prompt | Random-path MV8 | **71.72%** | 69.70% | +2.02 pp | 3 / 8 | 0.376× |
| MMLU-Pro32 | MultiMaze, repeated prompt | Retrieval SC3 | **78.12%** | 75.00% | +3.12 pp | 3 / 3 | 0.997× |
| MMLU-Pro32 | MultiMaze, repeated prompt | Random-prompt SC3 | **78.12%** | 65.62% | +12.50 pp | 3 / 3 | 1.105× |
| GPQA Diamond99 | Single-call Maze | Random-path MV3 | 65.66% | 66.67% | -1.01 pp | 1 / 3 | 0.343× |
| GPQA Diamond99 | Single-call Maze | Random-path MV4 | 65.66% | 69.70% | -4.04 pp | 1 / 4 | 0.258× |
| GPQA Diamond99 | Single-call Maze | Random-path MV8 | 65.66% | 69.70% | -4.04 pp | 1 / 8 | 0.127× |
| GPQA Diamond99 | Single-call Maze | Random-prompt SC3 | 65.66% | 71.72% | -6.06 pp | 1 / 3 | 0.356× |
| GPQA Diamond99 | Single-call Maze | Retrieval, 1 call | 65.66% | 72.73% | -7.07 pp | 1 / 1 | 0.997× |
| GPQA Diamond99 | Single-call Maze | Retrieval SC3 | 65.66% | 71.72% | -6.06 pp | 1 / 3 | 0.333× |
| GPQA Diamond99 | MultiMaze, repeated prompt | Random prompt, 1 call | **71.72%** | 62.63% | +9.09 pp | 3 / 1 | 3.131× |
| GPQA Diamond99 | MultiMaze, repeated prompt | Random-path MV2 | **71.72%** | 63.64% | +8.08 pp | 3 / 2 | 1.498× |
| GPQA Diamond99 | MultiMaze, repeated prompt | Random-prompt SC3 | 71.72% | 71.72% | +0.00 pp | 3 / 3 | 1.053× |
| GPQA Diamond99 | MultiMaze, repeated prompt | Retrieval, 1 call | 71.72% | 72.73% | -1.01 pp | 3 / 1 | 2.950× |
| GPQA Diamond99 | MultiMaze, repeated prompt | Retrieval SC3 | 71.72% | 71.72% | +0.00 pp | 3 / 3 | 0.985× |
| GPQA32 (post hoc) | MultiMaze, fixed selected prompt | Retrieval SC3 | **81.25%** | 68.75% | +12.50 pp | 3 / 3 | 1.248× |
| GPQA32 (post hoc) | MultiMaze, fixed selected prompt | Random-path MV8 | **81.25%** | 75.00% | +6.25 pp | 3 / 8 | 0.445× |
| GPQA32 (post hoc) | MultiMaze, fixed selected prompt | Legacy planner SC3 | **81.25%** | 68.75% | +12.50 pp | 3 / 3 | 1.132× |

# Compact scorer replay

These implementations preserve the compact geometric scorer and reference-relative calibration used in earlier Maze experiment configurations.
The files in `research/` are source-bound copies, and their numerical semantics are tested against the reusable implementations under `src/maze/`.

| Source | Behavior |
|---|---|
| `structured_energy_v118.py` | Four-view directed vector energy, node normalization, query-conditioned fusion, success and auxiliary heads |
| `structured_energy_v126_portable.py` | Full-QA 32-view input, all-view transition context, interior-only readout |
| `energy_residual_v142.py` | Query-balanced, intercept-free residual utility calibration |
| `replay.py` | Validated CPU inference from a compatible externally supplied state/checkpoint bundle |

```bash
python reproduction/compact/replay.py --bundle /path/to/authorized-bundle \
  --output runs/compact-scorer-replay.json
```

The original benchmark bundles are not included.
This command validates supplied bundles and recomputes scores and selected paths; it does not launch the target model.
To reconstruct the result tables from the included numeric records, use `python scripts/reproduce_results.py` instead.

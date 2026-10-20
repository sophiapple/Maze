"""Portable scalar and vector Maze energies on cached input representations.

All matrices use PyTorch's [output, input] storage convention. Scalar fusion
reduces channels before the cross-view nonlinearity; vector fusion does not.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import numpy as np
from maze.io import read_json, safe_member, sha256, write_json


@dataclass(frozen=True)
class EnergyConfig:
    family: str
    dim: int
    views: int
    rank: int = 64
    hidden: int = 96
    shots: int = 4
    window: int = 2
    start_mode: str = "encoded_system_full_qa"

    def __post_init__(self) -> None:
        if self.family not in {"paper_scalar", "compact_vector", "fullqa_vector"}:
            raise ValueError("Unknown energy family")
        for key in ("dim", "views", "rank", "hidden", "shots"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.window) is not int or self.window < 0:
            raise ValueError("window must be a nonnegative integer")
        if self.start_mode not in {"encoded_system_full_qa", "zero_start_question_nodes"}:
            raise ValueError("Unknown start-node representation")
        if self.family == "fullqa_vector" and self.views <= 2 * self.window:
            raise ValueError("Interior-only fusion needs at least one interior view")
        if self.family in {"paper_scalar", "fullqa_vector"} and self.start_mode != "encoded_system_full_qa":
            raise ValueError("This configuration requires the encoded-system/full-QA recipe")

    @property
    def active(self) -> tuple[int, ...]:
        if self.family == "fullqa_vector":
            return tuple(range(self.window, self.views - self.window))
        return tuple(range(self.views))

    def neighbors(self, layer: int, *, fusion: bool = False) -> tuple[int, ...]:
        if fusion and self.family == "paper_scalar" and not self.window <= layer < self.views - self.window:
            return ()
        return tuple(j for j in range(max(0, layer - self.window),
                                     min(self.views, layer + self.window + 1)) if j != layer)

    @property
    def score_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = {"mu.weight": (self.rank, self.dim), "mass.weight": (self.rank, self.dim),
                  "fusion.0.weight": (self.hidden, 2 * self.rank),
                  "fusion.2.weight": (self.rank, self.hidden)}
        if self.family != "paper_scalar":
            shapes.update({"energy_readout.weight": (1, self.rank), "energy_readout.bias": (1,)})
        return shapes

    @property
    def auxiliary_shapes(self) -> dict[str, tuple[int, ...]]:
        if self.family == "paper_scalar":
            return {}
        return {"anchor_heads.0.weight": (self.hidden, 4 * self.rank),
                "anchor_heads.0.bias": (self.hidden,), "anchor_heads.2.weight": (2, self.hidden),
                "anchor_heads.2.bias": (2,), "cost_head.0.weight": (self.hidden, 2 * self.rank),
                "cost_head.0.bias": (self.hidden,), "cost_head.2.weight": (1, self.hidden),
                "cost_head.2.bias": (1,)}


class NumpyEnergy:
    def __init__(self, config: EnergyConfig, weights: dict[str, np.ndarray]):
        self.config = config
        allowed = set(config.score_shapes) | set(config.auxiliary_shapes)
        if not set(config.score_shapes) <= set(weights) or not set(weights) <= allowed:
            raise ValueError("Unexpected or missing checkpoint parameters")
        shapes = config.score_shapes | config.auxiliary_shapes
        self.weights = {}
        for key, value in weights.items():
            array = np.asarray(value)
            if array.dtype.kind != "f" or array.shape != shapes[key] or not np.isfinite(array).all():
                raise ValueError(f"Invalid checkpoint array {key}")
            self.weights[key] = array.astype(np.float64, copy=False)

    def project(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(states, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != (self.config.views, self.config.dim) or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected finite node states [nodes, views, dim]")
        if self.config.family != "paper_scalar":
            x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8) * math.sqrt(self.config.dim)
        return x @ self.weights["mu.weight"].T, x @ self.weights["mass.weight"].T

    def _fusion(self, query: np.ndarray, energy: np.ndarray) -> np.ndarray:
        return np.tanh(np.concatenate([query, energy], axis=-1)
                       @ self.weights["fusion.0.weight"].T) @ self.weights["fusion.2.weight"].T

    def score_projected(self, projected: tuple[np.ndarray, np.ndarray],
                        indices: np.ndarray, *, cross_view: bool = True) -> np.ndarray:
        c = self.config
        metric_table, mass_table = projected
        idx = np.asarray(indices)
        if (idx.dtype.kind not in "iu" or idx.ndim != 2 or not len(idx)
                or idx.shape[1] != c.shots + 2 or idx.min() < 0
                or idx.max() >= len(metric_table)):
            raise ValueError("Invalid complete path-node indices")
        if any(len(set(row[1:-1])) != c.shots or row[-1] in row[1:-1] or row[0] in row[1:] for row in idx):
            raise ValueError("Each path needs distinct exemplars and a separate query node")
        if metric_table.shape != mass_table.shape or metric_table.shape[1:] != (c.views, c.rank):
            raise ValueError("Projected feature shape mismatch")
        if not np.isfinite(metric_table).all() or not np.isfinite(mass_table).all():
            raise ValueError("Nonfinite projected states")
        z, m = metric_table[idx], mass_table[idx]
        if c.start_mode == "zero_start_question_nodes" and (np.any(z[:, 0]) or np.any(m[:, 0])):
            raise ValueError("Zero-start recipe received nonzero start features")
        static = []
        for layer in range(c.views):
            d = z[:, :-1, layer] - z[:, 1:, layer]
            neighbors = c.neighbors(layer) if cross_view else ()
            cross = np.zeros_like(d)
            for j in neighbors:
                cross += z[:, :-1, j] - z[:, 1:, layer]
            d = d + np.tanh(cross)
            static.append(np.tanh(m[:, :-1, layer] * d).sum(axis=1))
        static = np.stack(static, axis=1)
        fused = []
        for layer in c.active:
            own = self._fusion(m[:, -1, layer], static[:, layer]) * static[:, layer]
            if c.family == "paper_scalar":
                own = own.sum(axis=-1)
            cross = np.zeros_like(own)
            for j in c.neighbors(layer, fusion=True) if cross_view else ():
                term = self._fusion(m[:, -1, layer], static[:, j]) * static[:, j]
                cross += term.sum(axis=-1) if c.family == "paper_scalar" else term
            fused.append(own + np.tanh(cross))
        whole = np.stack(fused, axis=1).sum(axis=1)
        if c.family == "paper_scalar":
            scores = whole
        else:
            whole = whole / math.sqrt(len(c.active))
            scores = (whole @ self.weights["energy_readout.weight"].T).ravel() + self.weights["energy_readout.bias"][0]
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite energy; no fallback ranking is applied")
        return scores

    def score(self, states: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return self.score_projected(self.project(states), indices)


def load_checkpoint(directory: str | Path) -> tuple[NumpyEnergy, dict]:
    root = Path(directory)
    info = read_json(root / "checkpoint.json")
    if info.get("schema") != 1:
        raise ValueError("Unsupported checkpoint schema")
    path = safe_member(root, info["weights"])
    if sha256(path) != info["weights_sha256"]:
        raise ValueError("Checkpoint digest mismatch")
    with np.load(path, allow_pickle=False) as archive:
        weights = {key: archive[key] for key in archive.files}
    return NumpyEnergy(EnergyConfig(**info["architecture"]), weights), info


def save_checkpoint(directory: str | Path, config: EnergyConfig,
                    weights: dict[str, np.ndarray], metadata: dict) -> None:
    NumpyEnergy(config, weights)  # Validate before creating any files.
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(root / "weights.npz", **weights)
    write_json(root / "checkpoint.json", {"schema": 1, "architecture": asdict(config),
               "weights": "weights.npz", "weights_sha256": sha256(root / "weights.npz"),
               "metadata": metadata})

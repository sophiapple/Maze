"""Indexed, input-only cached encoder features, independent of any test labels."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from maze.io import read_json, safe_member, sha256, write_json


class FeatureStore:
    def __init__(self, states: np.ndarray, nodes: list[dict], recipe: dict):
        self.states = np.asarray(states)
        if (self.states.dtype.kind != "f" or self.states.ndim != 3 or len(self.states) != len(nodes)
                or not all(self.states.shape) or not np.isfinite(self.states).all()):
            raise ValueError("Expected finite cached states [nodes, views, dim]")
        self.nodes, self.recipe = nodes, recipe
        self.lookup = {}
        for i, node in enumerate(nodes):
            if set(node) != {"id", "kind", "question_key"}:
                raise ValueError("Node metadata accepts only id, kind and question_key")
            if not isinstance(node["id"], str) or not node["id"] or node["id"] in self.lookup:
                raise ValueError("Node IDs must be unique nonempty strings")
            if node["kind"] not in {"system", "exemplar", "query"}:
                raise ValueError("Unknown node kind")
            key = node["question_key"]
            if node["kind"] != "system" and (not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key)):
                raise ValueError("Every exemplar/query needs a normalized-question SHA256")
            if node["kind"] == "system" and key is not None:
                raise ValueError("System nodes have no question key")
            self.lookup[node["id"]] = i
        if sum(n["kind"] == "system" for n in nodes) != 1:
            raise ValueError("Exactly one system/start node is required")
        self.start = next(i for i,n in enumerate(nodes) if n["kind"] == "system")
        if recipe.get("start_mode") == "zero_start_question_nodes":
            if np.any(self.states[self.start]):
                raise ValueError("Zero-start input is not zero")
        elif recipe.get("start_mode") == "encoded_system_full_qa":
            if not np.any(self.states[self.start]):
                raise ValueError("An encoded system cannot be silently replaced by zero")
        else:
            raise ValueError("Feature recipe must identify its start_mode")

    def path_indices(self, query_id: str, paths: list[tuple[str, ...] | list[str]]) -> np.ndarray:
        q = self.lookup[query_id]
        if self.nodes[q]["kind"] != "query":
            raise ValueError("query_id is not a query node")
        result = []
        for path in paths:
            if not path or len(set(path)) != len(path):
                raise ValueError("Exemplars in a path must be distinct")
            ids = [self.lookup[e] for e in path]
            if any(self.nodes[i]["kind"] != "exemplar" for i in ids):
                raise ValueError("Only exemplar nodes may appear in a path")
            keys = [self.nodes[i]["question_key"] for i in ids]
            if len(keys) != len(set(keys)) or self.nodes[q]["question_key"] in keys:
                raise ValueError("Duplicate question or self-exemplar leakage")
            result.append([self.start, *ids, q])
        if not result or len({len(x) for x in result}) != 1:
            raise ValueError("A score batch needs nonempty equal-length paths")
        return np.asarray(result, dtype=np.int64)

    @classmethod
    def load(cls, directory: str | Path) -> "FeatureStore":
        root = Path(directory)
        info = read_json(root / "features.json")
        if set(info) != {"schema", "arrays", "arrays_sha256", "nodes", "recipe"} or info["schema"] != 1:
            raise ValueError("Strict feature-manifest schema required; no outcome fields")
        path = safe_member(root, info["arrays"])
        if sha256(path) != info["arrays_sha256"]:
            raise ValueError("Feature array digest mismatch")
        with np.load(path, allow_pickle=False) as data:
            if data.files != ["states"]:
                raise ValueError("Feature archive may contain only the states array")
            return cls(data["states"], info["nodes"], info["recipe"])

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(root / "nodes.npz", states=self.states)
        write_json(root / "features.json", {"schema": 1, "arrays": "nodes.npz",
                   "arrays_sha256": sha256(root / "nodes.npz"), "nodes": self.nodes, "recipe": self.recipe})

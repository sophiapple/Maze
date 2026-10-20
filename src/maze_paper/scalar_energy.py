"""NumPy reference for the camera-ready *scalar-fusion* equations.

This new equation-level implementation is not an archived benchmark checkpoint.
Inputs are frozen, input-side states in [batch, path_node, view, feature] order.
Node 0 is the encoded system/start node; the last node is the query.
No node normalization, projection biases, vector readout, or layer averaging
is introduced. These would change the scalar configuration in the paper.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScalarWeights:
    relation: np.ndarray     # [d, r], W_mu
    mass: np.ndarray         # [d, r], W_m
    fusion_hidden: np.ndarray  # [2r, h], bias-free
    fusion_output: np.ndarray  # [h, r], bias-free

    def checked(self):
        arrays = tuple(np.asarray(a, dtype=np.float64) for a in (
            self.relation, self.mass, self.fusion_hidden, self.fusion_output))
        if any(a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all()
               for a in arrays):
            raise ValueError("weights must be finite nonempty matrices")
        mu, mass, hidden, output = arrays
        d, r = mu.shape
        if mass.shape != (d, r) or hidden.shape[0] != 2 * r:
            raise ValueError("incompatible relation, mass or hidden shapes")
        if output.shape != (hidden.shape[1], r):
            raise ValueError("incompatible fusion output shape")
        return arrays

    @property
    def parameter_count(self):
        return sum(a.size for a in self.checked())


def score_paths(states, weights):
    """Return one finite scalar per path, without reading outcomes or costs."""
    x = np.asarray(states, dtype=np.float64)
    mu, mass_w, hidden, output = weights.checked()
    if (x.ndim != 4 or x.shape[0] < 1 or x.shape[1] < 2
            or x.shape[2] < 1 or x.shape[3] != mu.shape[0]
            or not np.isfinite(x).all()):
        raise ValueError("states must be finite [batch, nodes>=2, views>=1, d]")
    z = x @ mu
    masses = x @ mass_w
    layers = x.shape[2]
    energy = np.empty((x.shape[0], layers, mu.shape[1]), dtype=np.float64)
    # Cross-view transition terms exist at boundaries using available neighbors.
    for layer in range(layers):
        neighbors = [j for j in range(max(0, layer - 2), min(layers, layer + 3))
                     if j != layer]
        cross = np.zeros_like(z[:, :-1, layer])
        for j in neighbors:
            cross += z[:, :-1, j] - z[:, 1:, layer]
        directed = z[:, :-1, layer] - z[:, 1:, layer] + np.tanh(cross)
        energy[:, layer] = np.tanh(masses[:, :-1, layer] * directed).sum(axis=1)

    def reduced_fusion(query_mass, path_energy):
        fused = np.tanh(np.concatenate((query_mass, path_energy), axis=-1)
                        @ hidden) @ output
        return (fused * path_energy).sum(axis=-1)

    result = np.zeros(x.shape[0], dtype=np.float64)
    for layer in range(layers):
        query_mass = masses[:, -1, layer]
        result += reduced_fusion(query_mass, energy[:, layer])
        # One-based interior set in the manuscript is {3, ..., L-2}.
        if 2 <= layer < layers - 2:
            cross_scalar = np.zeros(x.shape[0], dtype=np.float64)
            for j in (layer - 2, layer - 1, layer + 1, layer + 2):
                cross_scalar += reduced_fusion(query_mass, energy[:, j])
            result += np.tanh(cross_scalar)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite score; do not silently select a fallback")
    return result


def preference_loss(positive_scores, negative_scores):
    """Mean log(1+exp(s_minus-s_plus)) for observed correctness-flipping pairs."""
    plus = np.asarray(positive_scores, dtype=np.float64)
    minus = np.asarray(negative_scores, dtype=np.float64)
    if (plus.shape != minus.shape or plus.size == 0
            or not np.isfinite(plus).all() or not np.isfinite(minus).all()):
        raise ValueError("paired scores must be finite, nonempty and shape-matched")
    with np.errstate(over="ignore"):
        return float(np.logaddexp(0.0, minus - plus).mean())

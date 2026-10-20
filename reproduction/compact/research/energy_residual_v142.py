"""Fold-local residual calibration of frozen whole-path energy heads.

This is a small prompt-side utility head, not a causal harm model. Neural
geometries remain frozen; no answers or realized lengths enter selection.
"""
import math
import numpy as np


def question_weights(query_ids):
    counts = {q: query_ids.count(q) for q in set(query_ids)}
    return np.array([1.0 / counts[q] for q in query_ids], dtype=np.float64)


def fit_residual(x, y, query_ids, alpha=10.0):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert x.ndim == 2 and y.shape == (len(x),) and len(query_ids) == len(x)
    assert len(set(query_ids)) >= 2 and np.isfinite(x).all() and np.isfinite(y).all()
    assert np.max(np.abs(y)) <= 1 and alpha > 0
    weights = question_weights(list(query_ids))
    # No centering: all-zero anchor features must retain exactly zero utility.
    scale = np.sqrt((x * x * weights[:, None]).sum(0) / weights.sum()).clip(1e-6)
    z = x / scale
    lhs = (z.T * weights) @ z + alpha * np.eye(z.shape[1])
    rhs = z.T @ (weights * y)
    beta = np.linalg.solve(lhs, rhs)
    assert np.isfinite(beta).all()
    return dict(beta=beta.tolist(), scale=scale.tolist(), alpha=float(alpha),
                fit_queries=len(set(query_ids)), fit_conditions=len(y),
                calibration_weighting='Equal question mass; no all-correct downweighting')


def residual_scores(model, x):
    z = np.asarray(x, dtype=np.float64) / np.asarray(model['scale'])
    result = z @ np.asarray(model['beta'])
    assert np.isfinite(result).all()
    return result


def features(predictions, indices, anchor):
    """Only whitelisted frozen prompt-side scores are accepted."""
    rows = []
    for i in indices:
        row = []
        for p in predictions:
            assert set(p) == {'scores', 'uplift', 'harm'}
            sigmoid = lambda x: .5 * (1 + math.tanh(float(x) / 2))
            row.extend([sigmoid(p['scores'][i]) - sigmoid(p['scores'][anchor]),
                        float(p['uplift'][i]), float(p['harm'][i])])
        row.append(float(i != anchor))
        rows.append(row)
    out = np.asarray(rows, dtype=np.float64)
    assert np.array_equal(out[indices.index(anchor)], np.zeros(out.shape[1]))
    return out


def choose(scores, indices, anchor, keys, policy):
    assert len(scores) == len(indices) and len(indices) == len(set(indices))
    values = dict(zip(indices, map(float, scores)))
    assert all(math.isfinite(v) for v in values.values()) and values[anchor] == 0
    order = sorted(indices, key=lambda i: (-values[i], i != anchor, keys[i]))
    if policy == 'one_call':
        selected = order[:1]
    elif policy == 'top3':
        selected = order[:3]
    elif policy == 'anchor3':
        selected = [anchor] + [i for i in order if i != anchor][:2]
    else:
        raise ValueError(policy)
    assert len(selected) == len(set(selected)) == (1 if policy == 'one_call' else 3)
    return selected

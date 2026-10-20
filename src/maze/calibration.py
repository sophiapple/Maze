"""Small query-balanced residual utility calibration on held-out fitting records."""
from __future__ import annotations
from collections import Counter
import numpy as np


def fit_residual(x,y,query_keys,*,alpha:float=10.,held_out_keys=()):
    """Calibrate input-side forecasts against observed utility differences.

    The all-zero anchor stays at utility zero because no intercept is fitted.
    held_out_keys must contain the complete evaluation query membership.
    """
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);keys=list(query_keys)
    if x.ndim!=2 or not x.size or y.shape!=(len(x),) or len(keys)!=len(x) or len(set(keys))<2:
        raise ValueError('Calibration needs aligned rows from at least two fitting queries')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.max(np.abs(y))>1 or not np.isfinite(alpha) or alpha<=0:
        raise ValueError('Nonfinite inputs, out-of-range utility differences or invalid regularization')
    if any(not isinstance(k,str) or not k for k in keys):raise ValueError('Nonempty query identities required')
    if set(keys)&set(held_out_keys):raise ValueError('Calibration/evaluation question overlap')
    counts=Counter(keys);weights=np.array([1/counts[k] for k in keys])
    scale=np.sqrt((x*x*weights[:,None]).sum(0)/weights.sum()).clip(1e-6)
    z=x/scale
    beta=np.linalg.solve((z.T*weights)@z+alpha*np.eye(z.shape[1]),z.T@(weights*y))
    if not np.isfinite(beta).all():raise ValueError('Nonfinite calibrated weights')
    return {'beta':beta.tolist(),'scale':scale.tolist(),'alpha':alpha,
            'fit_question_keys':sorted(set(keys)),'held_out_question_keys':sorted(set(held_out_keys))}


def predict_residual(model,x):
    x=np.asarray(x,dtype=float);scale=np.asarray(model['scale']);beta=np.asarray(model['beta'])
    if x.ndim!=2 or beta.ndim!=1 or scale.shape!=beta.shape or x.shape[1]!=len(beta) or np.any(scale<=0):
        raise ValueError('Incompatible residual input shape')
    output=x/scale@beta
    if not np.isfinite(output).all():raise ValueError('Nonfinite residual utility')
    return output

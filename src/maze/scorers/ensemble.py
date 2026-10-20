"""Input-only consensus of independently fitted path scorers."""
from __future__ import annotations
import numpy as np
from .numpy_energy import NumpyEnergy


class EnergyEnsemble:
    def __init__(self,heads:list[NumpyEnergy],weights:list[float]|None=None):
        if not heads:raise ValueError('At least one trained head is required')
        if any(h.config!=heads[0].config for h in heads):
            raise ValueError('Consensus heads must share the same encoder/architecture contract')
        self.heads=heads;self.config=heads[0].config
        self.weights=np.ones(len(heads)) if weights is None else np.asarray(weights,dtype=float)
        if self.weights.shape!=(len(heads),) or not np.isfinite(self.weights).all() or np.any(self.weights<0) or self.weights.sum()<=0:
            raise ValueError('Consensus weights must be finite, nonnegative and have positive sum')
        self.weights=self.weights/self.weights.sum()

    def project(self,states):
        return [head.project(states) for head in self.heads]

    def score_projected(self,projected,indices,*,cross_view=True):
        if len(projected)!=len(self.heads):raise ValueError('Missing projected head features')
        values=np.stack([head.score_projected(p,indices,cross_view=cross_view) for head,p in zip(self.heads,projected)])
        return self.weights@values

    def score(self,states,indices):
        return self.score_projected(self.project(states),indices)

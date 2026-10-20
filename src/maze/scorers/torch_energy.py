"""Differentiable Maze energies with checkpoint-compatible shared projections."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .numpy_energy import EnergyConfig


class GeometricEnergy(nn.Module):
    """Trainable scalar/full-layer and compact/vector implementations.

    This implementation is checked numerically against both the independent
    NumPy paper reference and the preserved compact implementations.
    """
    def __init__(self, config: EnergyConfig):
        super().__init__()
        self.config = config
        d, r, h = config.dim, config.rank, config.hidden
        self.mu = nn.Linear(d, r, bias=False)
        self.mass = nn.Linear(d, r, bias=False)
        self.fusion = nn.Sequential(nn.Linear(2*r, h, bias=False), nn.Tanh(), nn.Linear(h, r, bias=False))
        if config.family != "paper_scalar":
            self.energy_readout = nn.Linear(r, 1)
            self.anchor_heads = nn.Sequential(nn.Linear(4*r, h), nn.Tanh(), nn.Linear(h, 2))
            self.cost_head = nn.Sequential(nn.Linear(2*r, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, states: torch.Tensor, anchors: torch.Tensor | None = None,
                *, cross_view: bool = True) -> dict[str, torch.Tensor]:
        c = self.config
        if states.ndim != 4 or tuple(states.shape[1:]) != (c.shots+2, c.views, c.dim):
            raise ValueError("Expected [paths, shots+2, views, dim]")
        if not torch.isfinite(states).all():
            raise ValueError("Nonfinite input features")
        if c.start_mode == "zero_start_question_nodes" and torch.any(states[:, 0] != 0):
            raise ValueError("Zero-start configuration received an encoded start")
        x = states
        if c.family != "paper_scalar":
            x = F.normalize(x, dim=-1, eps=1e-8) * math.sqrt(c.dim)
        z, m = self.mu(x), self.mass(x)
        energies = []
        for layer in range(c.views):
            distance = z[:, :-1, layer] - z[:, 1:, layer]
            neighbors = c.neighbors(layer) if cross_view else ()
            if neighbors:
                cross = sum(z[:, :-1, j] - z[:, 1:, layer] for j in neighbors)
                distance = distance + torch.tanh(cross)
            energies.append(torch.tanh(m[:, :-1, layer] * distance).sum(dim=1))
        static = torch.stack(energies, dim=1)
        fused = []
        for layer in c.active:
            query = m[:, -1, layer]
            own = self.fusion(torch.cat([query, static[:, layer]], dim=-1)) * static[:, layer]
            if c.family == "paper_scalar":
                own = own.sum(dim=-1)
            cross = torch.zeros_like(own)
            for j in c.neighbors(layer, fusion=True) if cross_view else ():
                term = self.fusion(torch.cat([query, static[:, j]], dim=-1)) * static[:, j]
                cross = cross + (term.sum(dim=-1) if c.family == "paper_scalar" else term)
            fused.append(own + torch.tanh(cross))
        whole = torch.stack(fused, dim=1).sum(dim=1)
        if c.family == "paper_scalar":
            return {"logits": whole, "static_energy": static}
        whole = whole / math.sqrt(len(c.active))
        query = m[:, -1, list(c.active)].mean(dim=1)
        logits = self.energy_readout(whole).squeeze(-1)
        result = {"logits": logits, "static_energy": static, "energy": whole,
                  "log_cost_scaled": F.softplus(self.cost_head(torch.cat([whole, query], -1)).squeeze(-1))}
        if anchors is not None:
            if anchors.shape != logits.shape or anchors.dtype != torch.long:
                raise ValueError("One integer anchor index is required per candidate")
            if torch.any(anchors < 0) or torch.any(anchors >= len(states)):
                raise ValueError("Anchor index outside the current query batch")
            anchor = whole[anchors]
            heads = self.anchor_heads(torch.cat([whole, anchor, whole-anchor, query], dim=-1))
            is_anchor = anchors == torch.arange(len(anchors), device=anchors.device)
            result.update(uplift=torch.where(is_anchor, 0., torch.tanh(heads[:, 0])),
                          harm_probability=torch.where(is_anchor, 0., torch.sigmoid(heads[:, 1])),
                          harm_logits=heads[:, 1])
        return result

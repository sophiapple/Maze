"""Full-QA, encoded-system, strict-interior W2 supervised path energy.

All 32 canonical views supply neighbors. Only layers 3--30 contribute to
the final energy and query mass. This has no causal order-edit supervision.
"""
import math
import torch
from maze_replay_geometry import StructuredEnergyV118, structured_loss, select_policy


class StructuredEnergyV126(StructuredEnergyV118):
    def __init__(self, dim=2560, views=32, rank=16, hidden=32,
                 *, start_mode='encoded_system_full_qa'):
        assert views == 32 and start_mode == 'encoded_system_full_qa'
        super().__init__(dim, views, rank, hidden, start_mode=start_mode)
        self.interior = tuple(range(2, 30))

    def geometry(self, metric, mass, *, cross_view=True):
        assert metric.shape == mass.shape and metric.shape[-3:] == (6, 32, self.rank)
        source, destination = metric[:, :-1], metric[:, 1:]
        distance = source - destination
        if cross_view:
            neighbor_source = torch.zeros_like(source)
            degree = torch.zeros(32, dtype=source.dtype, device=source.device)
            for offset in (-2, -1, 1, 2):
                lo, hi = max(0, -offset), min(32, 32 - offset)
                neighbor_source[:, :, lo:hi] += source[:, :, lo + offset:hi + offset]
                degree[lo:hi] += 1
            distance = distance + torch.tanh(neighbor_source - destination * degree[None, None, :, None])
        static = torch.tanh(mass[:, :-1] * distance).sum(1)
        query_mass = mass[:, -1, 2:30]
        center = static[:, 2:30]
        fused = self.fusion(torch.cat([query_mass, center], -1)) * center
        if cross_view:
            cross = torch.zeros_like(fused)
            for offset in (-2, -1, 1, 2):
                neighbor = static[:, 2 + offset:30 + offset]
                cross = cross + self.fusion(torch.cat([query_mass, neighbor], -1)) * neighbor
            fused = fused + torch.tanh(cross)
        whole = fused.sum(1) / math.sqrt(28)
        return whole, query_mass.mean(1), static

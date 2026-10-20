"""Versioned compact directed energy diagnostic, preserving the legacy scorer.

The cached arm is explicitly a question-node/zero-start ablation. It does not
claim encoded system or full QA/rationale nodes, nor causal ordering evidence.
Mass modulation uses Hadamard multiplication so energy remains rank-dimensional.
Cross-view fusion clips neighbors at the available-view boundaries; this is a
named finite-view adaptation of the manuscript, not its empty strict interior
set when L=4 and W=2.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class StructuredEnergyV118(nn.Module):
    def __init__(self, dim=2560, views=4, rank=64, hidden=96, *, start_mode):
        super().__init__()
        assert start_mode in ('zero_start_question_nodes', 'encoded_system_full_qa')
        self.init = dict(dim=dim, views=views, rank=rank, hidden=hidden, start_mode=start_mode)
        self.dim, self.views, self.rank = dim, views, rank
        self.start_mode = start_mode
        self.mu = nn.Linear(dim, rank, bias=False)
        self.mass = nn.Linear(dim, rank, bias=False)
        self.fusion = nn.Sequential(nn.Linear(2 * rank, hidden, bias=False), nn.Tanh(),
                                    nn.Linear(hidden, rank, bias=False))
        self.energy_readout = nn.Linear(rank, 1)
        self.anchor_heads = nn.Sequential(nn.Linear(4 * rank, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.cost_head = nn.Sequential(nn.Linear(2 * rank, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def project(self, table):
        """Frozen input states only; row-local normalization uses no labels/stats."""
        assert table.device.type == 'cpu' and table.shape[-2:] == (self.views, self.dim)
        states = F.normalize(table.float(), dim=-1, eps=1e-8) * math.sqrt(self.dim)
        return self.mu(states), self.mass(states)

    def geometry(self, metric, mass, *, cross_view=True):
        assert metric.shape == mass.shape and metric.shape[-3:] == (6, self.views, self.rank)
        source, destination = metric[:, :-1], metric[:, 1:]
        layer_energies = []
        for layer in range(self.views):
            distance = source[:, :, layer] - destination[:, :, layer]
            neighbors = [j for j in range(self.views) if 0 < abs(j - layer) <= 2]
            if cross_view and neighbors:
                cross = sum((source[:, :, j] - destination[:, :, layer] for j in neighbors))
                distance = distance + torch.tanh(cross)
            layer_energies.append(torch.tanh(mass[:, :-1, layer] * distance).sum(1))
        static = torch.stack(layer_energies, 1)
        query_mass = mass[:, -1]
        fused = []
        for layer in range(self.views):
            own = self.fusion(torch.cat([query_mass[:, layer], static[:, layer]], -1)) * static[:, layer]
            neighbors = [j for j in range(self.views) if 0 < abs(j - layer) <= 2]
            if cross_view and neighbors:
                cross = sum((self.fusion(torch.cat([query_mass[:, layer], static[:, j]], -1)) * static[:, j]
                             for j in neighbors))
                own = own + torch.tanh(cross)
            fused.append(own)
        whole = torch.stack(fused, 1).sum(1) / math.sqrt(self.views)
        return whole, query_mass.mean(1), static

    def forward_projected(self, metric_table, mass_table, node_indices, anchor_rows, *, cross_view=True, system_receipt=None):
        """No labels, outcomes, observed lengths or generated answers are accepted."""
        assert node_indices.ndim == 2 and node_indices.shape[1] == 6
        assert node_indices.shape[0] == anchor_rows.shape[0]
        assert anchor_rows.dtype == node_indices.dtype == torch.long
        if self.start_mode == 'zero_start_question_nodes':
            assert torch.all(metric_table[node_indices[:, 0]] == 0) and torch.all(mass_table[node_indices[:, 0]] == 0)
        else:
            assert system_receipt and system_receipt.get('exact_system_input_verified') is True
            assert system_receipt.get('full_qa_rationale_nodes_verified') is True
            assert torch.any(metric_table[node_indices[:, 0]] != 0) or torch.any(mass_table[node_indices[:, 0]] != 0)
        whole, query_mass, static = self.geometry(metric_table[node_indices], mass_table[node_indices], cross_view=cross_view)
        anchor = whole[anchor_rows]
        hidden = torch.cat([whole, anchor, whole - anchor, query_mass], -1)
        heads = self.anchor_heads(hidden)
        is_anchor = torch.arange(len(anchor_rows), device=anchor_rows.device) == anchor_rows
        success_logits = self.energy_readout(whole).squeeze(-1)
        uplift = torch.where(is_anchor, torch.zeros_like(heads[:, 0]), torch.tanh(heads[:, 0]))
        harm = torch.where(is_anchor, torch.zeros_like(heads[:, 1]), torch.sigmoid(heads[:, 1]))
        return dict(logits=success_logits, probability=torch.sigmoid(success_logits), energy=whole,
                    static_energy=static, uplift=uplift, harm_probability=harm, harm_logits=heads[:, 1],
                    log_cost_scaled=F.softplus(self.cost_head(torch.cat([whole, query_mass], -1)).squeeze(-1)))

    def forward(self, nodes, anchor_rows, *, cross_view=True, system_receipt=None):
        assert nodes.ndim == 4 and nodes.shape[1:] == (6, self.views, self.dim)
        if self.start_mode == 'zero_start_question_nodes':
            assert torch.all(nodes[:, 0] == 0)
        else:
            assert system_receipt and system_receipt.get('exact_system_input_verified') is True
            assert system_receipt.get('full_qa_rationale_nodes_verified') is True
            assert torch.any(nodes[:, 0] != 0), 'An encoded-system arm cannot silently use zero start'
        table = nodes.reshape(-1, self.views, self.dim)
        mu, mass = self.project(table)
        indices = torch.arange(len(table), dtype=torch.long).reshape(len(nodes), 6)
        return self.forward_projected(mu, mass, indices, anchor_rows, cross_view=cross_view, system_receipt=system_receipt)


def structured_loss(output, target, counts, anchors, weights, log_tokens_scaled,
                    paired_uplift, paired_harm, matched_mask, *, harm_trained):
    """Call with training/inner-validation query rows only; never outer labels."""
    def weighted(values, mask=None):
        w = weights if mask is None else weights * mask.float()
        return (values * w).sum() / w.sum().clamp_min(1e-8)
    bce = weighted(F.binary_cross_entropy_with_logits(output['logits'], target, reduction='none'))
    ranking_target = (1 + target - target[anchors]) / 2
    rank = weighted(F.binary_cross_entropy_with_logits(output['logits'] - output['logits'][anchors],
                                                     ranking_target, reduction='none'))
    cost = weighted((output['log_cost_scaled'] - log_tokens_scaled).square())
    uplift = weighted((output['uplift'] - paired_uplift).square(), matched_mask)
    harm = weighted(F.binary_cross_entropy_with_logits(output['harm_logits'], paired_harm, reduction='none'), matched_mask)
    loss = bce + .5 * rank + .1 * cost
    if harm_trained:
        loss = loss + .25 * uplift + .25 * harm
    return loss, dict(success_bce=bce, within_query_rank=rank, predicted_cost=cost,
                      paired_uplift=uplift, paired_harm=harm, local_order_loss=torch.zeros_like(loss))


def select_policy(scores, anchor, keys, policy, uplift=None, harm=None):
    """Prompt-side inference only, fixed call counts and deterministic ties."""
    assert set(scores) == set(keys) and anchor in scores
    assert all(math.isfinite(float(v)) for v in scores.values())
    ranked = sorted(scores, key=lambda i: (-float(scores[i]), i != anchor, keys[i]))
    if policy == 'one_call':
        result = ranked[:1]
    elif policy == 'top3':
        result = ranked[:3]
    elif policy == 'anchor3':
        result = [anchor] + [i for i in ranked if i != anchor][:2]
    elif policy == 'harm_guard':
        assert uplift is not None and harm is not None and set(uplift) == set(harm) == set(scores)
        guard = {i: float(uplift[i]) - .25 * float(harm[i]) for i in scores}
        guard[anchor] = 0.
        candidates = [i for i in guard if i != anchor and guard[i] > 0]
        result = [min(candidates, key=lambda i: (-guard[i], keys[i]))] if candidates else [anchor]
    else:
        raise ValueError(policy)
    assert len(result) == len(set(result)) == (3 if policy in ('top3', 'anchor3') else 1)
    return result

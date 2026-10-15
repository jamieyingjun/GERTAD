"""Common learned goal interface and auxiliary physical heads."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import lm_native, ce_native, model_api


class GoalAdapter(nn.Module):

    def __init__(self, kind, backbone, specs, mapdim, C, normalization):
        super().__init__()
        self.kind = kind
        self.backbone = backbone
        self.specs = specs
        self.C = C
        self.project = nn.Sequential(nn.Linear(128 if kind == 'gm' else 1600, 128), nn.LayerNorm(128), nn.GELU())
        self.map_point = nn.Sequential(nn.Linear(18, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.entity_head = nn.Sequential(nn.Linear(mapdim + 128 + 64, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.event_head = nn.Linear(64, 4)
        nn.init.zeros_(self.event_head.weight)
        nn.init.zeros_(self.event_head.bias)
        self.goal_head = nn.Sequential(nn.Linear(128 + 4 + 3 * C + 2 + 128 + 3 * C + 4, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer('map_mean', torch.tensor(normalization['mean'], dtype=torch.float32))
        self.register_buffer('map_sd', torch.tensor(normalization['sd'], dtype=torch.float32))

    def forward(self, b, goal, local_obs, components):
        h = model_api.embedding(self.kind, self.backbone, b)
        h = self.project(h)
        valid = b['entity_valid']
        B, E = valid.shape
        ef = (b['map_entity'] - self.map_mean) / self.map_sd
        x = b['x']
        T = x.shape[2]
        mask = torch.arange(T)[None, None, :] < b['lengths'][:, None, None]
        ph = self.map_point(x)
        average = (ph * mask[:, :, :, None]).sum(2) / b['lengths'][:, None, None]
        w = (x[:, :, :, 9] + 0.01) * mask
        w = w / w.sum(2, keepdim=True).clamp_min(1e-08)
        local = (ph * w[:, :, :, None]).sum(2)
        eh = self.entity_head(torch.cat([ef, h[:, None].expand(-1, E, -1), average, local], 2))
        r = self.event_head(eh)
        base = b['base']
        visit = base[:, :, 0] + r[:, :, 0]
        conditional = (base[:, :, 1] + r[:, :, 1]).clamp(0, 1)
        cost = (base[:, :, 2] + r[:, :, 2]).clamp_min(0)
        zprior = torch.where(base[:, :, 1] <= 1e-07, 6.0, -6.0)
        zero = torch.sigmoid(zprior + r[:, :, 3])
        first = torch.where(zero >= 0.5, torch.zeros_like(conditional), conditional)
        event = torch.stack([visit, first, cost], 2)
        cats = F.one_hot(b['entity_category'].clamp_min(0), self.C + 1)[:, :, :self.C].float() * valid[:, :, None]
        pv = torch.sigmoid(visit) * valid
        counts = torch.einsum('be,bec->bc', pv, cats)
        firsts = torch.einsum('be,bec->bc', pv * first, cats) / counts.clamp_min(0.001)
        costs = torch.einsum('be,bec->bc', pv * cost, cats) / counts.clamp_min(0.001)
        evidence = torch.cat([counts / 10, firsts, costs], 1)[local_obs]
        task_eh = eh[local_obs]
        task_cats = cats[local_obs]
        task_valid = valid[local_obs]
        scope = (task_cats * goal[:, None, 4:4 + self.C]).sum(2)
        inside = (task_eh * scope[:, :, None]).sum(1) / scope.sum(1).clamp_min(1)[:, None]
        outside = (task_eh * (task_valid.float() - scope)[:, :, None]).sum(1) / (task_valid.float() - scope).sum(1).clamp_min(1)[:, None]
        logit = self.goal_head(torch.cat([h[local_obs], goal, inside, outside, evidence, components], 1)).squeeze(1)
        return dict(logits=logit, events=event, zero=zero, conditional=conditional, mean=(1 - zero) * conditional)

def loss(p, b, y, local_obs):
    truth = b['event_targets']
    valid = b['entity_valid']
    positive = valid & (truth[:, :, 0] > 0.5)

    def avg(v, m):
        return v[m].mean() if m.any() else v.sum() * 0
    visit = avg(F.binary_cross_entropy_with_logits(p['events'][:, :, 0], truth[:, :, 0], reduction='none'), valid)
    trtime = torch.where(truth[:, :, 1] <= 1e-07, p['mean'], p['conditional'])
    first = avg(F.smooth_l1_loss(trtime * b['duration'][:, None] / 20, truth[:, :, 1] * b['duration'][:, None] / 20, reduction='none'), positive)
    cost = avg(F.smooth_l1_loss(p['events'][:, :, 2] * 5, truth[:, :, 2] * 5, reduction='none'), positive)
    onset = avg(F.binary_cross_entropy(p['zero'].clamp(1e-06, 1 - 1e-06), (truth[:, :, 1] <= 1e-07).float(), reduction='none'), positive)
    count = F.smooth_l1_loss((p['events'][:, :, 0].sigmoid() * valid).sum(1), (truth[:, :, 0] * valid).sum(1))
    bi = torch.arange(len(valid))[:, None]
    a = b['pair_a']
    bb = b['pair_b']
    ta = p['events'][bi, a, 1] * b['duration'][:, None]
    tb = p['events'][bi, bb, 1] * b['duration'][:, None]
    za = p['zero'][bi, a]
    zb = p['zero'][bi, bb]
    q = torch.sigmoid(tb - ta)
    ab = za * (1 - zb) + (1 - za) * (1 - zb) * q
    ba = zb * (1 - za) + (1 - za) * (1 - zb) * (1 - q)
    reverse = ((b['pair_label'] < 0.5) & (b['pair_gap'] > 1e-07)).float()
    order = avg(0.5 * (F.binary_cross_entropy(ab.clamp(1e-06, 1 - 1e-06), b['pair_label'], reduction='none') + F.binary_cross_entropy(ba.clamp(1e-06, 1 - 1e-06), reverse, reduction='none')), b['pair_valid'])
    tl = F.binary_cross_entropy_with_logits(p['logits'], y, reduction='none')
    sums = tl.new_zeros(len(valid))
    sums.scatter_add_(0, local_obs, tl)
    n = torch.bincount(local_obs, minlength=len(valid)).clamp_min(1)
    task = (sums / n).mean()
    total = task + visit + first + cost + 0.2 * onset + 0.1 * count + 0.2 * order
    return (total, dict(total=total, task=task, visit=visit, first=first, cost=cost, onset=onset, count=count, order=order))

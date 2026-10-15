"""Frozen external adapters with indexed observation/goal separation."""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass, replace
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from .config import ROOT, CATEGORIES
from . import lm_native, ce_native, gm_native, idk_native, model_api, adapter4, adapter5, native_views
from . import scoring as reasoning
from . import preprocessing as observation_prepare
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class GoalCatalog:
    contracts: tuple
    vectors: torch.Tensor
    categories: tuple
    identity_sha256: str
    structured: tuple
    indexed: object

@dataclass(frozen=True)
class ObservationState:
    h: torch.Tensor
    entity_hidden: torch.Tensor
    category_onehot: torch.Tensor
    valid: torch.Tensor
    evidence: torch.Tensor
    events: torch.Tensor
    zero: torch.Tensor
    conditional: torch.Tensor
    mean: torch.Tensor
    base: np.ndarray
    entity_ids: tuple
    entity_categories: tuple
    context_ids: tuple
    method_identity: str
    prior_evidence: tuple
    prior_index: object
    cache_keys: tuple

    def bytes(self):
        tensors = [self.h, self.entity_hidden, self.category_onehot, self.valid, self.evidence, self.events, self.zero, self.conditional, self.mean]
        index_bytes = sum((x.nbytes for x in vars(self.prior_index).values() if isinstance(x, np.ndarray)))
        metadata = json.dumps([self.entity_ids, self.entity_categories, self.context_ids, self.cache_keys], ensure_ascii=False).encode()
        return sum((x.numel() * x.element_size() for x in tensors)) + self.base.nbytes + index_bytes + len(metadata)

class ExternalRuntime:

    def __init__(self, kind, seed, device='cpu'):
        self.kind, self.seed, self.device = (kind, int(seed), torch.device(device))
        if self.device.type != 'cpu':
            raise ValueError('Frozen comparison uses CPU FP32.')
        self.categories = CATEGORIES
        self.goal_run = ROOT / 'checkpoints' / f'{kind}_{seed}'
        self.native_run = self.goal_run / 'native'
        self.modules = dict(lm=lm_native, ce=ce_native, gm=gm_native, idk=idk_native, api=model_api, prep=SimpleNamespace(old=native_views))
        self.reasoning, self.observation_prepare = (reasoning, observation_prepare)
        normalization = read(self.goal_run / 'config.json')['map_normalization']
        if kind == 'lm':
            backbone, specs = (lm_native.load_model(self.native_run), None)
        elif kind == 'ce':
            backbone, specs = ce_native.load_models(self.native_run)
            self.view_function = native_views.views
        elif kind == 'gm':
            backbone, specs = (gm_native.load_model(self.native_run), None)
        elif kind == 'idk':
            backbone, specs = (model_api.FixedKernel(), None)
            idk_native.author()
            with (self.native_run / 'model.pkl').open('rb') as stream:
                self.kernel = pickle.load(stream)['first']
        else:
            raise ValueError(kind)
        adapter = adapter4 if kind in ('lm', 'ce') else adapter5
        self.model = adapter.GoalAdapter(kind, backbone, specs, len(normalization['mean']), len(self.categories), normalization)
        ck = torch.load(self.goal_run / 'model.pt', map_location='cpu', weights_only=True)
        self.model.load_state_dict(ck['state_dict'], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        if kind in ('lm', 'gm'):
            tokenization = read(ROOT / 'config/tokenization.json')
            self.vocabulary = {tuple(cell): i + 4 for i, cell in enumerate(tokenization['vocabulary']['cells'])}
        if kind == 'ce':
            self.view_normalization = read(ROOT / 'config/ce_preprocessing.json')['normalization']
        self.identity = dict(kind=kind, seed=seed, checkpoint=sha(self.goal_run / 'model.pt'), implementation=sha(Path(__file__)))
        self.identity_sha256 = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()

    def activate(self):
        return nullcontext()

    def compile_goals(self, goals):
        """Deterministic contract-only catalog; charge cold build, reuse warm."""
        safe = tuple(({key: value for key, value in goal.items() if key in {'goal_id', 'active_components', 'target_scope_categories', 'required_categories', 'ordered_categories', 'required_count', 'accessibility_budget', 'goal_template'}} for goal in goals))
        vectors = np.asarray([self.modules['prep'].old.goal_encoding(g, self.categories)[0] for g in safe], dtype=np.float32)
        ident = hashlib.sha256(json.dumps(safe, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        indexed = self.reasoning.CompiledGoals(safe)
        structured = tuple(indexed.contracts)
        return GoalCatalog(safe, torch.from_numpy(vectors).to(self.device), self.categories, ident, structured, indexed)

    def prepare_native(self, raws, common_batch):
        """Raw trajectory -> frozen native inputs + required map summary/tensors.

        common_batch comes from the unified raw observable feature preparation.
        No archived lm_tokens, CET views, map_entity or IDK features are loaded.
        Native vocabulary/statistics/partitions are fixed training artifacts.
        """
        x = np.asarray(common_batch['x'], dtype=np.float32)
        lengths = np.asarray(common_batch['lengths'], dtype=np.int64)
        mask = np.arange(x.shape[2])[None, None, :] < lengths[:, None, None]
        average = (x * mask[:, :, :, None]).sum(2) / lengths[:, None, None]
        sd = np.sqrt(((x - average[:, :, None, :]) ** 2 * mask[:, :, :, None]).sum(2) / lengths[:, None, None])
        weights = (x[:, :, :, 9] + 0.01) * mask
        weights /= weights.sum(2, keepdims=True)
        local = (x * weights[:, :, :, None]).sum(2)
        map_entity = np.concatenate([common_batch['entity'], common_batch['base'], average, sd, local], 2).astype(np.float32)
        values = {key: common_batch[key] for key in ('x', 'base', 'entity_valid', 'entity_category', 'lengths', 'duration')}
        values['map_entity'] = map_entity
        if self.kind in ('lm', 'gm'):
            tokens = np.zeros((len(raws), 52), dtype=np.int64)
            for index, raw in enumerate(raws):
                cells = self.modules['lm']._cells(raw)
                encoded = [self.vocabulary.get(cell, 3) for cell in cells]
                tokens[index, :len(encoded) + 2] = [1] + encoded + [2]
            values['lm_tokens'] = tokens
        elif self.kind == 'ce':
            for view in ('route', 'speed', 'shape'):
                width = len(self.view_normalization[view]['mean'])
                values[view] = np.zeros((len(raws), 64, width), np.float32)
                values[view + '_mask'] = np.zeros((len(raws), 64, 1), np.float32)
            for index, raw in enumerate(raws):
                views, _ = self.view_function(raw)
                for view, sequence in views.items():
                    sequence = np.asarray(sequence, np.float64)
                    norm = self.view_normalization[view]
                    n = len(sequence)
                    values[view][index, :n] = (sequence - np.asarray(norm['mean'])) / np.asarray(norm['std'])
                    values[view + '_mask'][index, :n] = 1
        else:
            points = []
            for raw in raws:
                xy = np.asarray(raw['coordinates'], np.float64)
                points.append(np.column_stack([xy[:, 0] * 111320, xy[:, 1] * 111320 * np.cos(np.radians(40))]))
            values['kernel_embedding'] = self.modules['idk'].distribution_features(self.kernel, points)
        result = {}
        for key, value in values.items():
            a = np.asarray(value)
            if a.dtype in (np.float16, np.float64):
                a = a.astype(np.float32)
            result[key] = torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        ordered = [sorted(raw['observed_map'], key=lambda e: e['entity_id']) for raw in raws]
        result['_entity_ids'] = tuple((tuple((e['entity_id'] for e in entities)) for entities in ordered))
        result['_entity_categories'] = tuple((tuple((e['category'] for e in entities)) for entities in ordered))
        result['_context_ids'] = tuple((raw.get('context_id', '') for raw in raws))
        result['_raws'] = tuple(raws)
        return result

    @torch.inference_mode()
    def encode_observations(self, common_batch, native):
        """Exact prefix of frozen forward, computed once per observation."""
        model, b = (self.model, native)
        if model.training:
            raise RuntimeError('Observation cache cannot be built in training mode')
        with self.activate():
            if self.kind == 'lm':
                h = self.modules['lm'].embedding(model.backbone, b)
            elif self.kind == 'ce':
                h = self.modules['ce'].embedding(model.backbone, model.specs, b)
            elif self.kind == 'gm':
                h = self.modules['gm'].embedding(model.backbone, b)
            else:
                h = model.backbone(b)
        h = model.project(h)
        valid = b['entity_valid']
        _, entities = valid.shape
        ef = (b['map_entity'] - model.map_mean) / model.map_sd
        x = b['x']
        mask = torch.arange(x.shape[2], device=x.device)[None, None, :] < b['lengths'][:, None, None]
        ph = model.map_point(x)
        average = (ph * mask[:, :, :, None]).sum(2) / b['lengths'][:, None, None]
        w = (x[:, :, :, 9] + 0.01) * mask
        w = w / w.sum(2, keepdim=True).clamp_min(1e-08)
        local = (ph * w[:, :, :, None]).sum(2)
        eh = model.entity_head(torch.cat([ef, h[:, None].expand(-1, entities, -1), average, local], 2))
        residual = model.event_head(eh)
        base = b['base']
        visit = base[:, :, 0] + residual[:, :, 0]
        conditional = (base[:, :, 1] + residual[:, :, 1]).clamp(0, 1)
        cost = (base[:, :, 2] + residual[:, :, 2]).clamp_min(0)
        zprior = torch.where(base[:, :, 1] <= 1e-07, 6.0, -6.0)
        zero = torch.sigmoid(zprior + residual[:, :, 3])
        first = torch.where(zero >= 0.5, torch.zeros_like(conditional), conditional)
        events = torch.stack([visit, first, cost], 2)
        cats = F.one_hot(b['entity_category'].clamp_min(0), model.C + 1)[:, :, :model.C].float() * valid[:, :, None]
        pv = torch.sigmoid(visit) * valid
        counts = torch.einsum('be,bec->bc', pv, cats)
        firsts = torch.einsum('be,bec->bc', pv * first, cats) / counts.clamp_min(0.001)
        costs = torch.einsum('be,bec->bc', pv * cost, cats) / counts.clamp_min(0.001)
        evidence = torch.cat([counts / 10, firsts, costs], 1)
        ids = tuple((tuple(x) for x in native['_entity_ids']))
        categories = tuple((tuple(x) for x in native['_entity_categories']))
        context = tuple(native['_context_ids'])
        prior = base.cpu().numpy().copy()
        prior_index = self.reasoning.EvidenceIndex(prior, b['entity_category'].numpy(), valid.numpy())
        keys = tuple((self.observation_prepare.cache_key(raw, self.identity['checkpoint'], self.identity_sha256) for raw in native['_raws']))
        return ObservationState(h, eh, cats, valid, evidence, events, zero, conditional, (1 - zero) * conditional, prior, ids, categories, context, self.identity_sha256, (), prior_index, keys)

    def observable_components(self, state, obs_indices, goal_indices, catalog):
        """Goal-dependent deterministic prior components, never true events."""
        _, components = self.reasoning.indexed_scores(state.prior_index, catalog.indexed, obs_indices, goal_indices)
        return torch.tensor(components, dtype=torch.float32, device=self.device).reshape(-1, 4)

@dataclass(frozen=True)
class IndexedCatalog:
    original: GoalCatalog
    scope_values: torch.Tensor
    interface_identity: str

@dataclass(frozen=True)
class IndexedState:
    original: ObservationState
    category_index: torch.Tensor
    scope_valid: torch.Tensor
    interface_identity: str

    def bytes(self):
        return self.original.bytes() + sum((t.numel() * t.element_size() for t in (self.category_index, self.scope_valid)))

class IndexedRuntime:

    def __init__(self, kind, seed, device='cpu'):
        self.original = ExternalRuntime(kind, seed, device)
        self.model = self.original.model
        self.kind, self.seed, self.device = (kind, int(seed), self.original.device)
        self.goal_run, self.native_run = (self.original.goal_run, self.original.native_run)
        self.identity = self.original.identity
        self.identity_sha256 = self.original.identity_sha256

    def prepare_native(self, raws, common_batch):
        return self.original.prepare_native(raws, common_batch)

    @torch.inference_mode()
    def compile_goals(self, goals):
        catalog = self.original.compile_goals(goals)
        scope = catalog.vectors[:, 4:4 + self.model.C]
        if not bool(((scope == 0) | (scope == 1)).all()):
            raise ValueError('Frozen category scope encoding must contain only0/1')
        return IndexedCatalog(catalog, scope, self.identity_sha256)

    @torch.inference_mode()
    def encode_observations(self, common_batch, native):
        original = self.original.encode_observations(common_batch, native)
        category = native['entity_category'].clamp_min(0)
        scope_valid = original.valid * (category < self.model.C).float()
        index = torch.where(category < self.model.C, category, torch.zeros_like(category)).long()
        original = replace(original, category_onehot=torch.empty((len(index), 0, 0), device=self.device))
        return IndexedState(original, index, scope_valid, self.identity_sha256)

    def scope(self, state, observation_indices, goal_indices, catalog):
        if state.interface_identity != self.identity_sha256 or catalog.interface_identity != self.identity_sha256:
            raise ValueError('Cache or catalog belongs to another checkpoint/interface')
        oi = torch.as_tensor(observation_indices, dtype=torch.long, device=self.device)
        gi = torch.as_tensor(goal_indices, dtype=torch.long, device=self.device)
        return torch.gather(catalog.scope_values[gi], 1, state.category_index[oi]) * state.scope_valid[oi]

    @torch.inference_mode()
    def score_goals(self, state, observation_indices, goal_indices, goals, *, components=None):
        catalog = goals if isinstance(goals, IndexedCatalog) else self.compile_goals(goals)
        scope = self.scope(state, observation_indices, goal_indices, catalog)
        oi = torch.as_tensor(observation_indices, dtype=torch.long, device=self.device)
        gi = torch.as_tensor(goal_indices, dtype=torch.long, device=self.device)
        if len(oi) != len(gi):
            raise ValueError('Observation/goal query length mismatch')
        original = state.original
        eh, valid = (original.entity_hidden[oi], original.valid[oi])
        if components is None:
            components = self.original.observable_components(original, oi.tolist(), gi.tolist(), catalog.original)
        inside = (eh * scope[:, :, None]).sum(1) / scope.sum(1).clamp_min(1)[:, None]
        outside = (eh * (valid.float() - scope)[:, :, None]).sum(1) / (valid.float() - scope).sum(1).clamp_min(1)[:, None]
        inputs = torch.cat([original.h[oi], catalog.original.vectors[gi], inside, outside, original.evidence[oi], components], 1)
        logits = self.model.goal_head(inputs).squeeze(1)
        return {'logits': logits, 'observable_components': components}

def _concatenate_base(states):
    """Workload cache assembly is included in cold preparation/build timing."""
    if not states or len({s.method_identity for s in states}) != 1:
        raise ValueError('States must be nonempty and use one checkpoint/interface')
    tensor_fields = ['h', 'entity_hidden', 'category_onehot', 'valid', 'evidence', 'events', 'zero', 'conditional', 'mean']
    values = {key: torch.cat([getattr(s, key) for s in states], 0) for key in tensor_fields}
    values['base'] = np.concatenate([s.base for s in states], 0)
    for key in ['entity_ids', 'entity_categories', 'context_ids', 'prior_evidence', 'cache_keys']:
        values[key] = tuple((x for s in states for x in getattr(s, key)))
    values['method_identity'] = states[0].method_identity
    first = states[0].prior_index
    index = object.__new__(type(first))
    for key in ['valid', 'cat', 'visit', 'cost', 'count', 'first', 'present']:
        setattr(index, key, np.concatenate([getattr(s.prior_index, key) for s in states], 0))
    values['prior_index'] = index
    return ObservationState(**values)

def concatenate_states(states):
    if not states or len({s.interface_identity for s in states}) != 1:
        raise ValueError('States must be nonempty and use one checkpoint/interface')
    return IndexedState(_concatenate_base([s.original for s in states]), torch.cat([s.category_index for s in states], 0), torch.cat([s.scope_valid for s in states], 0), states[0].interface_identity)

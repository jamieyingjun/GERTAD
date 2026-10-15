"""Frozen native route/speed/shape views and supplied-goal encoding."""
import math
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .config import PARTS
CAP = 64


def cap_sequence(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        x = np.zeros((1, x.shape[1]))
    if len(x) > CAP:
        loc = np.linspace(0, len(x) - 1, CAP)
        x = np.column_stack([np.interp(loc, np.arange(len(x)), x[:, i]) for i in range(x.shape[1])])
    return x

def views(r):
    c = np.asarray(r['coordinates'], dtype=float)
    t = np.asarray(r['timestamps'], dtype=float)
    rel = c - c[0]
    rad = np.radians(c)
    dl = np.diff(rad, axis=0)
    a = np.sin(dl[:, 0] / 2) ** 2 + np.cos(rad[:-1, 0]) * np.cos(rad[1:, 0]) * np.sin(dl[:, 1] / 2) ** 2
    distance = 6371000 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    speed = np.r_[0, distance / np.maximum(np.diff(t), 1)][:, None]
    cum = np.r_[0, np.cumsum(distance)]
    nroute = max(2, int(math.ceil(cum[-1] / 50)))
    desired = np.arange(nroute) * 50 if nroute > 2 else np.array([0, cum[-1]])
    if len(desired) > CAP:
        desired = np.linspace(desired[0], desired[-1], CAP)
    uniq = np.r_[True, np.diff(cum) > 0]
    route = np.column_stack([np.interp(desired, cum[uniq], rel[uniq, j]) for j in range(2)])
    distinct = rel[np.r_[True, np.any(np.diff(rel, axis=0) != 0, axis=1)]]
    length = np.linalg.norm(np.diff(distinct, axis=0), axis=1).sum()
    if length <= 1e-12 or len(distinct) < 2:
        shape = np.zeros((2, 2))
        nshape = 2
    else:
        position = distinct / length
        times = np.arange(len(distinct)) / length
        nshape = max(2, int(math.ceil(times[-1] / 24)))
        query = np.linspace(0, times[-1], min(nshape, CAP)) if nshape > CAP or nshape == 2 else np.arange(nshape) * 24
        interpolated = np.column_stack([np.interp(query, times, position[:, j]) for j in range(2)])
        z = StandardScaler().fit_transform(interpolated)
        if np.max(np.abs(z)) < 1e-10:
            shape = np.zeros_like(z)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                shape = PCA(n_components=2).fit_transform(z)
    return ({'route': cap_sequence(route), 'speed': cap_sequence(speed), 'shape': cap_sequence(shape)}, {'route': nroute, 'speed': len(speed), 'shape': nshape})

def goal_encoding(g, categories):
    C = len(categories)
    idx = {k: i for i, k in enumerate(categories)}
    active = np.array([p in g['active_components'] for p in PARTS], np.float32)
    scope = np.array([k in g['target_scope_categories'] for k in categories], np.float32)
    required = np.array([k in g['required_categories'] for k in categories], np.float32)
    order = np.zeros(C, np.float32)
    for i, k in enumerate(g['ordered_categories']):
        order[idx[k]] = (i + 1) / max(1, len(g['ordered_categories']))
    vec = np.r_[active, scope, required, order, (g['required_count'] or 0) / 10, (g['accessibility_budget'] or 0) / 300].astype('float32')
    queries = np.tile(vec, (4, 1))
    queries[:, :4] = np.eye(4) * active[:, None]
    return (vec, queries)

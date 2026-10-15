"""Deterministic frozen physical corruptions and legal TEST workload."""
import gzip
import hashlib
import json
import numpy as np
from .config import ROOT
def _legacy_rank(value):
    return hashlib.sha256(('event-recovery-v1-20261002:' + str(value)).encode()).hexdigest()
def _v2_rank(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
def read_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def _legacy_corruption(world, condition):
    coords = np.asarray(world['coordinates'], float).copy()
    times = np.asarray(world['timestamps'], float).copy()
    entities = list(world['entities'])
    family = condition.split('_')[0]
    rng = np.random.default_rng(int(_v2_rank([world['context_id'], family])[:16], 16))
    if condition.startswith('gps_'):
        z = rng.normal(size=coords.shape)
        for i in range(1, len(z)):
            z[i] = 0.8 * z[i - 1] + np.sqrt(1 - 0.8 ** 2) * z[i]
        coords += z * float(condition.split('_')[1]) / np.array([111320.0, 111320.0 * np.cos(np.radians(coords[:, 0].mean()))])
    elif condition.startswith('drop_'):
        keep = rng.random(len(coords)) >= int(condition.split('_')[1]) / 100
        keep[[0, -1]] = True
        coords, times = (coords[keep], times[keep])
    elif condition == 'burst_20':
        count = max(1, int(round(0.2 * (len(coords) - 2))))
        start = int(rng.integers(1, len(coords) - count))
        keep = np.ones(len(coords), bool)
        keep[start:start + count] = False
        coords, times = (coords[keep], times[keep])
    elif condition.startswith('map_missing_'):
        keys = rng.random(len(entities))
        entities = [e for e, k in zip(entities, keys) if k >= int(condition.rsplit('_', 1)[1]) / 100]
    elif condition == 'timestamp_jitter_2':
        gap = np.diff(times)
        bound = 0.45 * np.minimum(gap[:-1], gap[1:])
        times[1:-1] += np.clip(rng.normal(0, 2, len(times) - 2), -bound, bound)
    elif condition != 'clean':
        raise ValueError(condition)
    assert len(coords) >= 2 and np.all(np.diff(times) > 0)
    return {'coordinates': coords, 'timestamps': times, 'observed_map': entities}

def corruption(world, condition):
    if condition in ['clean', 'gps_5', 'gps_15', 'gps_30']:
        return _legacy_corruption(world, condition)
    coords = np.asarray(world['coordinates'], float)
    times = np.asarray(world['timestamps'], float)
    keep = np.ones(len(times), bool)
    if condition.startswith('sampling_'):
        step = int(condition.rsplit('_', 1)[1])
        keep[:] = False
        keep[::step] = True
    elif condition.startswith('outage_'):
        width = float(condition.rsplit('_', 1)[1])
        duration = times[-1] - times[0]
        rng = np.random.default_rng(int(_v2_rank([world['context_id'], 'v2-outage-center'])[:16], 16))
        center = times[0] + rng.uniform() * duration
        keep = abs(times - center) > width / 2
    else:
        raise ValueError(condition)
    keep[[0, -1]] = True
    return dict(coordinates=coords[keep], timestamps=times[keep], observed_map=list(world['entities']))

def load_workload():
    observations = read_gzip(ROOT / 'data/observations.json.gz')
    rows = read_gzip(ROOT / 'data/tasks.json.gz')
    goals = json.loads((ROOT / 'data/goals.json').read_text(encoding='utf-8'))
    obsids = np.asarray([i for i, o in enumerate(observations) if o['split'] == 'test'], np.int64)
    worlds = {w['context_id']: w for w in read_gzip(ROOT / 'data/worlds.json.gz')}
    raws = []
    for i in obsids:
        o = observations[int(i)]
        raw = corruption(worlds[o['context_id']], o['condition'])
        raws.append(dict(raw, context_id=o['context_id'], condition=o['condition']))
    lookup = {int(i): j for j, i in enumerate(obsids)}
    ids = np.asarray([i for i, r in enumerate(rows) if r['split'] == 'test'], np.int64)
    oi = np.asarray([lookup[rows[int(i)]['observation_index']] for i in ids], np.int64)
    gi = np.asarray([rows[int(i)]['goal_index'] for i in ids], np.int64)
    if len(obsids) != 2981 or len(ids) != 38390 or len(goals) != 38:
        raise ValueError('Frozen TEST population differs.')
    schema = json.loads((ROOT / 'data/schema.json').read_text(encoding='utf-8'))
    return raws, goals, ids, obsids, oi, gi, dict(rows=rows, observations=observations, goals=goals, manifest=schema)

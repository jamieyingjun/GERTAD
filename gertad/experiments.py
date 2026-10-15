"""Recompute the complete frozen five-method TEST comparison."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch

from . import baselines, evaluation, quality, scoring
from .config import ROOT
from .model import build_model
from .observations import load_workload
from .pair_diagnostic import valid_pair_probs
from .preprocessing import prepare_observations

DISPLAY = {'tad': 'GERTAD', 'lm': 'LM-TAD GoalAdapter', 'ce': 'CETrajAD GoalAdapter',
           'gm': 'GM-VSAE GoalAdapter', 'idk': 'IDK GoalAdapter'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    def encode(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False, default=encode) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_files():
    manifest = read(ROOT / 'MANIFEST.json')
    for rel, expected in manifest['sha256'].items():
        target = (ROOT / rel).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file() or digest(target) != expected:
            raise RuntimeError('Missing or changed package input: ' + rel)
    return len(manifest['sha256'])


def _batches(common, size=16):
    count = len(common['x'])
    for start in range(0, count, size):
        yield start, {k: torch.from_numpy(np.ascontiguousarray(v[start:start + size])) for k, v in common.items()}


@torch.no_grad()
def predict_gertad(job, common, goals, oi, gi, obsids):
    ck = torch.load(ROOT / job['checkpoint']['file'], map_location='cpu', weights_only=True)
    net = build_model(ck['n_categories'], ck['entity_dim'], ck['architecture_config'])
    net.load_state_dict(ck['state_dict'], strict=True)
    net.eval()
    outputs = []
    for start, batch in _batches(common):
        p = net(batch)
        outputs.append({k: p[k].detach().cpu() for k in ('events', 'pmf', 'mu', 'sigma', 'zero', 'mean', 'conditional')})
    arrays = {k: np.concatenate([p[k].numpy() for p in outputs]) for k in ('events', 'zero', 'mean', 'conditional')}
    state = scoring.EvidenceIndex(arrays['events'], common['entity_category'], common['entity_valid'])
    catalog = scoring.CompiledGoals(goals)
    scores, components = [], []
    for start in range(0, len(oi), 256):
        s, c = scoring.indexed_scores(state, catalog, oi[start:start + 256], gi[start:start + 256])
        scores.append(s)
        components.append(c)
    # Diagnostics read supervision after the primary scores have been completed.
    with np.load(ROOT / 'data/order_truth.npz', allow_pickle=False) as data:
        truth = {k: data[k][obsids] for k in ('pair_a', 'pair_b', 'pair_valid', 'duration')}
    pairs = []
    offset = 0
    for p in outputs:
        n = len(p['events'])
        b = {k: torch.from_numpy(v[offset:offset + n]) for k, v in truth.items()}
        probabilities, _ = valid_pair_probs(net, p, b)
        pairs.append(probabilities.numpy())
        offset += n
    return arrays, np.concatenate(scores), np.concatenate(components), np.concatenate(pairs)


@torch.inference_mode()
def predict_external(job, raws, common, goals, oi, gi):
    runner = baselines.IndexedRuntime(job['method'], job['seed'])
    native = runner.prepare_native(raws, common)
    states = []
    for start in range(0, len(raws), 16):
        b = {k: v[start:start + 16] if isinstance(v, (np.ndarray, torch.Tensor, list, tuple)) and len(v) == len(raws) else v
             for k, v in native.items()}
        cb = {k: v[start:start + 16] for k, v in common.items()}
        states.append(runner.encode_observations(cb, b))
    state = baselines.concatenate_states(states)
    catalog = runner.compile_goals(goals)
    scores, components = [], []
    for start in range(0, len(oi), 256):
        p = runner.score_goals(state, oi[start:start + 256], gi[start:start + 256], catalog)
        scores.append(p['logits'].numpy())
        components.append(p['observable_components'].numpy())
    arrays = {k: getattr(state.original, k).numpy() for k in ('events', 'zero', 'mean', 'conditional')}
    return arrays, np.concatenate(scores), np.concatenate(components), None


def compare_predictions(prediction, reference_path):
    report = {}
    with np.load(reference_path, allow_pickle=False) as reference:
        for key in prediction:
            actual, expected = np.asarray(prediction[key]), reference[key]
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise RuntimeError('Prediction shape/dtype differs: ' + key)
            exact = bool(np.array_equal(actual, expected))
            difference = 0.0 if exact or actual.dtype == bool else float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            report[key] = dict(exact=exact, maximum_absolute_difference=difference)
            if key in ('ids', 'observation_indices', 'query_observation_indices', 'goal_indices', 'decisions', 'threshold'):
                if not exact:
                    raise RuntimeError('Frozen identity or operating decisions differ: ' + key)
            elif not np.allclose(actual, expected, rtol=1e-6, atol=1e-6):
                raise RuntimeError('Fresh predictions differ from frozen reference: ' + key)
    return report


def compare_metrics(actual, expected):
    differences = {}
    for key, value in expected.items():
        if value is None:
            if actual[key] is not None:
                raise RuntimeError('Metric eligibility differs: ' + key)
            differences[key] = None
        else:
            differences[key] = float(actual[key] - value)
            if abs(differences[key]) > 1e-12:
                raise RuntimeError('Frozen aggregate metric differs: ' + key)
    return differences


def run_job(job, output, workload, common):
    method, seed = job['method'], job['seed']
    raws, goals, ids, obsids, oi, gi, metadata = workload
    if method == 'tad':
        arrays, scores, components, order_probs = predict_gertad(job, common, goals, oi, gi, obsids)
    else:
        arrays, scores, components, order_probs = predict_external(job, raws, common, goals, oi, gi)
    threshold = quality.validate_calibration(job['calibration'])
    prediction = dict(ids=ids, observation_indices=obsids, query_observation_indices=oi, goal_indices=gi,
                      scores=scores, components=components, decisions=scores >= threshold,
                      threshold=np.asarray(threshold), **arrays)
    quality.validate_prediction(prediction)
    # Seal fresh forward output before opening saved predictions or evaluation truth.
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'predictions.npz', **prediction)
    reference = compare_predictions(prediction, ROOT / job['reference'])
    with np.load(ROOT / 'data/order_truth.npz', allow_pickle=False) as data:
        truth = {k: data[k] for k in ('pair_a', 'pair_b', 'pair_valid', 'pair_label', 'pair_gap')}
    primary = quality.evaluate_prediction(prediction, quality.apply_calibration(scores, job['calibration']),
                                          metadata, truth, evaluation, order_probs)
    common_scores, common_components = evaluation.constraint_scores(ids, arrays['events'][oi],
                                      metadata['rows'], metadata['observations'], goals)
    trace = quality.audit_trace(prediction, common_scores, common_components, goals, method, {'kind': 'raw_v3'})
    calibration = job['auxiliary_calibration']
    common_prediction = {k: v for k, v in prediction.items() if k not in ('scores', 'components', 'threshold', 'decisions')}
    common_threshold = quality.validate_calibration(calibration)
    common_prediction.update(scores=common_scores, components=common_components, threshold=np.asarray(common_threshold),
                             decisions=common_scores >= common_threshold)
    auxiliary = quality.evaluate_prediction(common_prediction, quality.apply_calibration(common_scores, calibration),
                                            metadata, truth, evaluation, order_probs)
    expected = read(ROOT / f'reference/{method}_{seed}_metrics.json')
    delta = dict(primary=compare_metrics(primary['aggregate'], expected['primary']),
                 common=compare_metrics(auxiliary['aggregate'], expected['common']))
    save(output / 'primary_metrics.json', primary)
    save(output / 'common_metrics.json', auxiliary)
    save(output / 'trace_audit.json', trace)
    result = dict(status='PASS', method=method, display_name=DISPLAY[method], seed=seed,
                  fresh_model_inference=True, training_or_calibration_refit=False, timing_benchmark=False,
                  physical_observations=len(obsids), goal_queries=len(ids),
                  checkpoint_sha256=job['checkpoint']['sha256'],
                  original_checkpoint_sha256=job['checkpoint']['original_sha256'],
                  prediction_comparison=reference, aggregate_differences=delta,
                  primary_aggregate=primary['aggregate'], common_aggregate=auxiliary['aggregate'])
    save(output / 'COMPLETED.json', result)
    return result


def write_tables(output, records):
    table = []
    for record in records:
        table.append(dict(method=record['display_name'], seed=record['seed'], **record['primary_aggregate']))
    with (output / 'comparison_by_seed.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    summary = []
    for method in DISPLAY:
        chosen = [r for r in records if r['method'] == method]
        if not chosen:
            continue
        for metric in chosen[0]['primary_aggregate']:
            values = [r['primary_aggregate'][metric] for r in chosen if r['primary_aggregate'][metric] is not None]
            summary.append(dict(method=DISPLAY[method], metric=metric, seeds=len(values),
                   mean=float(np.mean(values)) if values else None,
                   seed_sd_sample=float(np.std(values, ddof=1)) if len(values) > 1 else None))
    with (output / 'comparison_mean_sd.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    per_condition = []
    for record in records:
        directory = output / f'{record["method"]}_{record["seed"]}'
        metrics = read(directory / 'primary_metrics.json')
        for condition, values in metrics['conditions'].items():
            per_condition.append(dict(method=record['display_name'], seed=record['seed'], condition=condition,
                                      **values['classification']['macro']))
    with (output / 'classification_by_condition.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_condition[0]))
        writer.writeheader()
        writer.writerows(per_condition)
    return summary


def main(method=None, seed=None, output=None):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    print('Verifying package files...', flush=True)
    checked = verify_files()
    registry = read(ROOT / 'config/experiments.json')
    jobs = [j for j in registry['jobs'].values() if (method is None or j['method'] == method) and (seed is None or j['seed'] == seed)]
    destination = Path(output).resolve() if output else ROOT / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if destination.exists():
        raise FileExistsError('Choose a new output directory; existing results are preserved.')
    destination.mkdir(parents=True, exist_ok=False)
    save(destination / 'RUN_PLAN.json', dict(jobs=[dict(method=j['method'], seed=j['seed']) for j in jobs],
         verified_files=checked, python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
         fresh_model_inference=True, training_or_calibration_refit=False, timing_benchmark=False))
    records = []
    try:
        print('Loading frozen TEST observations and rebuilding observable features...', flush=True)
        workload = load_workload()
        common = prepare_observations(workload[0], True)
        print('Inputs ready: 2981 observations, 38390 goal queries, 11 conditions.', flush=True)
        for index, job in enumerate(jobs, 1):
            label = f'{DISPLAY[job["method"]]} / seed {job["seed"]}'
            print(f'[{index}/{len(jobs)}] Predicting and evaluating {label}...', flush=True)
            records.append(run_job(job, destination / f'{job["method"]}_{job["seed"]}', workload, common))
            print(f'[{index}/{len(jobs)}] PASS {label}', flush=True)
            gc.collect()
        summary = write_tables(destination, records)
        save(destination / 'SUMMARY.json', dict(status='PASS', jobs_completed=len(records),
             complete_five_method_comparison=len(records) == 15, fresh_model_inference=True,
             training_or_calibration_refit=False, timing_benchmark=False, summary=summary))
    except Exception:
        save(destination / 'FAILED.json', dict(status='FAIL', jobs_completed=len(records), error=traceback.format_exc()))
        print('Run failed. Details: ' + str(destination / 'FAILED.json'), flush=True)
        raise
    print('PASS: all requested predictions and metrics match the frozen references.', flush=True)
    display_path = destination.relative_to(ROOT) if destination.is_relative_to(ROOT) else destination
    print('Results: ' + display_path.as_posix(), flush=True)
    return destination

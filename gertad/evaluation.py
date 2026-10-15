"""Original task, recovery and ordering metrics with frozen thresholds."""
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, balanced_accuracy_score
from .config import PARTS
from .goals import goal_v3_from_mapping
from .contracts import TrajectoryTaskEvidence, VisitEvidence, score_v3


def evidence_score(goal, entity_ids, categories, visit, first, cost, context):
    facts = [(entity_ids[i], categories[i], float(first[i]), float(cost[i])) for i in range(len(entity_ids)) if visit[i]]
    order = {t: j for j, t in enumerate(sorted({f[2] for f in facts}))}
    ev = TrajectoryTaskEvidence(context, tuple((VisitEvidence(e, cat, order[t], max(0.0, d), raw_category=cat) for e, cat, t, d in facts)))
    return score_v3(goal_v3_from_mapping(goal), ev)

def confusion(y, p):
    tp = int(((y == 1) & p).sum())
    tn = int(((y == 0) & ~p).sum())
    fp = int(((y == 0) & p).sum())
    fn = int(((y == 1) & ~p).sum())
    return (tp, tn, fp, fn)

def _classification_base(y, s, users, threshold):
    y = np.asarray(y)
    s = np.asarray(s)
    users = np.asarray(users)
    p = s >= threshold
    tp, tn, fp, fn = confusion(y, p)

    def auc(yy, ss):
        return float(roc_auc_score(yy, ss)) if len(set(yy)) == 2 else None
    per = []
    for u in sorted(set(users)):
        mask = users == u
        yy, ss, pp = (y[mask], s[mask], p[mask])
        a, b, c, d = confusion(yy, pp)
        per.append(dict(user_id=u, n=int(mask.sum()), f1=2 * a / (2 * a + c + d) if 2 * a + c + d else 0.0, ba=(a / (a + d) + b / (b + c)) / 2 if a + d and b + c else None, auroc=auc(yy, ss)))
    macro = {k: float(np.mean([r[k] for r in per if r[k] is not None])) if any((r[k] is not None for r in per)) else None for k in ['f1', 'ba', 'auroc']}
    macro['valid_auc_users'] = sum((r['auroc'] is not None for r in per))
    macro['users'] = len(per)
    return dict(n=len(y), positive_rate=float(y.mean()), macro=macro, micro=dict(f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0, accuracy=float((p == y).mean()), auroc=auc(y, s), ap=float(average_precision_score(y, s)) if y.sum() else None, confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn)), per_user=per)

def constraint_scores(ids, events, rows, observations, goals):
    """events is task-aligned here; public API takes observation-aligned events."""
    scores, components = ([], [])
    for i, p in zip(ids, events):
        row = rows[int(i)]
        obs = observations[row['observation_index']]
        n = len(obs['entity_ids'])
        result = evidence_score(goals[row['goal_index']], obs['entity_ids'], obs['entity_categories'], p[:n, 0] >= 0, p[:n, 1], p[:n, 2] * 300, row['context_id'])
        scores.append(result['anomaly_score'])
        components.append([result['component_scores'].get(k, 0.0) for k in PARTS])
    return (np.asarray(scores), np.asarray(components))

def event_metrics(ids, events, data, rows, observations, goals):
    """v1-compatible scope metrics; true visible misses remain in MAE."""
    occurrences = Counter((rows[int(i)]['observation_index'] for i in ids))
    grouped = defaultdict(list)
    for i, p in zip(ids, events):
        row = rows[int(i)]
        oi = row['observation_index']
        obs = observations[oi]
        goal = goals[row['goal_index']]
        scope = set(goal['target_scope_categories'])
        n = len(obs['entity_ids'])
        truth = obs['truth_entities']
        visible_scope = np.array([cat in scope for cat in obs['entity_categories']], bool)
        actual = np.array([eid in truth for eid in obs['entity_ids']], bool)
        prediction = p[:n, 0] >= 0
        positive = actual & visible_scope
        selected = prediction & visible_scope
        tp = int((positive & selected).sum())
        fp = int((selected & ~positive).sum())
        fn = int((positive & ~selected).sum())
        invisible = sum((eid not in obs['entity_ids'] and e['category'] in scope for eid, e in truth.items()))
        time_errors = [abs(p[j, 1] * obs['duration'] - truth[eid]['time']) for j, eid in enumerate(obs['entity_ids']) if positive[j]]
        cost_errors = [abs(p[j, 2] * 300 - truth[eid]['accessibility_cost']) for j, eid in enumerate(obs['entity_ids']) if positive[j]]
        true_first, pred_first = ({}, {})
        for event in truth.values():
            if event['category'] in scope:
                cat = event['category']
                true_first[cat] = min(true_first.get(cat, float('inf')), event['time'])
        for j, cat in enumerate(obs['entity_categories']):
            if selected[j]:
                pred_first[cat] = min(pred_first.get(cat, float('inf')), p[j, 1] * obs['duration'])
        order_correct = []
        if 'order' in goal['active_components']:
            for a, b in zip(goal['ordered_categories'], goal['ordered_categories'][1:]):
                true_ok = a in true_first and b in true_first and (true_first[a] < true_first[b])
                pred_ok = a in pred_first and b in pred_first and (pred_first[a] < pred_first[b])
                order_correct.append(int(true_ok == pred_ok))
        grouped[row['user_id']].append({'weight': 1 / occurrences[oi], 'tp': tp, 'fp': fp, 'fn': fn, 'invisible': invisible, 'visible_count_error': abs(int(selected.sum()) - int(positive.sum())), 'full_count_error': abs(int(selected.sum()) - row['true_count']), 'first_sum': sum(time_errors), 'cost_sum': sum(cost_errors), 'positive': len(time_errors), 'order_sum': sum(order_correct), 'order_n': len(order_correct)})
    per = []
    for user, values in grouped.items():
        totals = {k: sum((v['weight'] * v[k] for v in values)) for k in values[0] if k != 'weight'}
        den = sum((v['weight'] for v in values))

        def f1(extra):
            return 2 * totals['tp'] / max(2 * totals['tp'] + totals['fp'] + totals['fn'] + extra, 1e-12)
        per.append({'user_id': user, 'visit_f1_visible': f1(0), 'visit_f1_full': f1(totals['invisible']), 'count_mae_visible': totals['visible_count_error'] / den, 'count_mae_full': totals['full_count_error'] / den, 'first_time_mae_s': totals['first_sum'] / totals['positive'] if totals['positive'] else None, 'accessibility_mae_m': totals['cost_sum'] / totals['positive'] if totals['positive'] else None, 'order_accuracy': totals['order_sum'] / totals['order_n'] if totals['order_n'] else None, 'missed_visible_events': totals['fn'], 'extra_events': totals['fp'], 'invisible_true_events': totals['invisible']})
    keys = ['visit_f1_visible', 'visit_f1_full', 'count_mae_visible', 'count_mae_full', 'first_time_mae_s', 'accessibility_mae_m', 'order_accuracy']
    macro = {k: float(np.mean([p[k] for p in per if p[k] is not None])) if any((p[k] is not None for p in per)) else None for k in keys}
    return {'macro': macro, 'per_user': per, 'weighting': 'inverse goal queries per observation, then user macro', 'regression_denominator': 'all true visited visible entities including missed predictions', 'order_accuracy_caveat': 'joint visit/order Boolean agreement; missing true categories may count correct'}

def classification(ids, scores, threshold, rows, probabilities=None):
    yy = np.array([rows[int(i)]['label'] for i in ids])
    users = np.array([rows[int(i)]['user_id'] for i in ids])
    result = _classification_base(yy, scores, users, threshold)
    result['micro']['ba'] = float(balanced_accuracy_score(yy, scores >= threshold))
    if probabilities is not None:
        probs = np.clip(probabilities, 1e-07, 1 - 1e-07)
        result['brier'] = float(brier_score_loss(yy, probs))
        ece = 0.0
        for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            mask = (probs >= a) & (probs < (b if b < 1 else 1.00001))
            if mask.any():
                ece += float(mask.mean() * abs(yy[mask].mean() - probs[mask].mean()))
        result['ece_10_equal_bins'] = ece
    return result

def _order_summary(records):
    keys = ['pair_probability_accuracy', 'pair_point_accuracy', 'end_to_end_pair_accuracy']
    grouped = defaultdict(list)
    for record in records:
        grouped[record['user_id']].append(record)
    per = []
    for user, values in sorted(grouped.items()):
        entry = {'user_id': user, 'pairs': len(values), 'observations': len({v['observation_index'] for v in values})}
        for key in keys:
            entry[key] = float(np.mean([v[key] for v in values]))
        entry['missing_predicted_pair_rate'] = float(np.mean([v['missing'] for v in values]))
        per.append(entry)
    macro = {key: float(np.mean([v[key] for v in per])) if per else None for key in keys}
    macro['missing_predicted_pair_rate'] = float(np.mean([v['missing_predicted_pair_rate'] for v in per])) if per else None
    return {'macro': macro, 'micro': {key: float(np.mean([v[key] for v in records])) if records else None for key in keys}, 'pairs': len(records), 'users': len(per), 'observations': len({v['observation_index'] for v in records}), 'per_user': per}

def order_metrics(observation_ids, events, data, observations, manifest, order_probs=None, order_probability_source=None):
    """Conditioned GT-first-entity ordering and reselected end-to-end category order.

    Probabilities refer ONLY to the GT first entities in pair_a/pair_b. They are
    not claimed as end-to-end probabilities after predicted visit/category
    selection. For end-to-end accuracy absent predicted categories are wrong,
    even when the actual strict-order label is 0.
    """
    q1, q2 = map(float, manifest['order_bins_s'])
    assert 0 <= q1 <= q2
    records = []
    seen = set()
    for oi in np.unique(observation_ids):
        oi = int(oi)
        obs = observations[oi]
        pred = events[oi]
        n = len(obs['entity_ids'])
        pred_first = {}
        for j, cat in enumerate(obs['entity_categories']):
            if pred[j, 0] >= 0:
                pred_first[cat] = min(pred_first.get(cat, float('inf')), float(pred[j, 1] * obs['duration']))
        valid_pairs = np.flatnonzero(data['pair_valid'][oi])
        metadata = obs.get('pairmeta', [])
        assert len(metadata) == len(valid_pairs), (oi, 'pair metadata alignment')
        for j, meta in zip(valid_pairs, metadata):
            a, b = (int(data['pair_a'][oi, j]), int(data['pair_b'][oi, j]))
            assert 0 <= a < n and 0 <= b < n
            label = int(data['pair_label'][oi, j])
            gap = float(data['pair_gap'][oi, j])
            assert gap >= 0
            assert label == int(meta['time_a'] < meta['time_b'])
            assert int(meta['a']) == a and int(meta['b']) == b
            ca, cb = (meta['cat_a'], meta['cat_b'])
            assert obs['entity_categories'][a] == ca and obs['entity_categories'][b] == cb
            dedup = (oi, ca, cb)
            assert dedup not in seen, (oi, ca, cb, 'duplicate category pair')
            seen.add(dedup)
            point = float(pred[a, 1]) < float(pred[b, 1])
            probability = float(order_probs[oi, j]) if order_probs is not None else float(point)
            assert np.isfinite(probability) and 0 <= probability <= 1
            missing = ca not in pred_first or cb not in pred_first
            end_correct = not missing and int(pred_first[ca] < pred_first[cb]) == label
            subset = 'exact_tie' if gap == 0 else 'very_close' if gap <= q1 else 'moderate' if gap <= q2 else 'well_separated'
            records.append({'user_id': obs['user_id'], 'observation_index': oi, 'cat_a': ca, 'cat_b': cb, 'label': label, 'gap_s': gap, 'subset': subset, 'probability': probability, 'pair_probability_accuracy': int(int(probability >= 0.5) == label), 'pair_point_accuracy': int(int(point) == label), 'end_to_end_pair_accuracy': int(end_correct), 'missing': int(missing)})
    subsets = {name: _order_summary([v for v in records if v['subset'] == name]) for name in ['very_close', 'moderate', 'well_separated', 'exact_tie']}
    subsets['nontie'] = _order_summary([v for v in records if v['gap_s'] > 0])
    subsets['subsecond'] = _order_summary([v for v in records if 0 < v['gap_s'] <= 1])
    result = _order_summary(records)
    result['subsets'] = subsets
    for name in ['very_close', 'moderate', 'well_separated', 'exact_tie', 'subsecond']:
        lookup = {v['user_id']: v for v in subsets[name]['per_user']}
        for entry in result['per_user']:
            item = lookup.get(entry['user_id'])
            entry[name + '_pairs'] = item['pairs'] if item else 0
            for key in ['pair_probability_accuracy', 'pair_point_accuracy', 'end_to_end_pair_accuracy']:
                entry[name + '_' + key] = item[key] if item else None
    result['order_bins_s'] = [q1, q2]
    result['probability_source'] = order_probability_source or ('temporal distribution' if order_probs is not None else 'strict point comparison, binary pseudo-probability')
    result['conditioned_recovery_definition'] = 'GT first visible entity pair, prediction need not declare these entities visited'
    result['end_to_end_definition'] = 'reselect category first among predicted visits; missing predicted category is incorrect'
    result['weighting'] = 'unique observation/category pair then user macro; no multiplication by goal queries'
    result['tie_definition'] = 'gap exactly zero, strict label remains 0; reported separately'
    return result

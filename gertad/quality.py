"""Pure saved-prediction metrics/trace checks; no model or benchmark loader."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

PARTS = ("count", "category", "order", "accessibility")
CONDITIONS = ("clean", "gps_5", "gps_15", "gps_30", "sampling_2", "sampling_4", "sampling_8", "outage_5", "outage_15", "outage_30", "outage_60")
SMOKE = ("clean", "gps_30", "sampling_4", "outage_30")
PRIMARY = ("f1", "ba", "auroc", "ap")


def array_hash(value):
    value = np.asarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode()
                          + np.ascontiguousarray(value).tobytes()).hexdigest()


def mean_defined(values):
    values = [float(v) for v in values if v is not None]
    if any(not np.isfinite(v) for v in values):
        raise ValueError("Nonfinite metric")
    return float(np.mean(values)) if values else None


def validate_calibration(value):
    if value.get("test_used") is not False or tuple(value["threshold_validation_conditions"]) != SMOKE:
        raise ValueError("Both calibrations require frozen validation-SMOKE-only provenance")
    for key in ("validation_task_ids_hash", "validation_labels_hash", "validation_score_hash"):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Missing validation calibration source hash: " + key)
    threshold = float(value["selection"]["threshold"])
    coef, intercept = np.asarray(value["calibration_coefficients"]), np.asarray(value["calibration_intercept"])
    if not np.isfinite(threshold) or coef.shape != (1, 1) or intercept.shape != (1,) or not np.isfinite(coef).all() or not np.isfinite(intercept).all():
        raise ValueError("Invalid serialized validation calibration")
    return threshold


def apply_calibration(scores, calibration):
    validate_calibration(calibration)
    result = expit(np.asarray(scores, np.float64) * calibration["calibration_coefficients"][0][0]
                   + calibration["calibration_intercept"][0])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite probability")
    return result


def validate_prediction(prediction):
    required = ("ids", "observation_indices", "query_observation_indices", "goal_indices", "scores", "components", "decisions", "events", "threshold", "zero", "mean", "conditional")
    if set(required) - set(prediction):
        raise ValueError("Incomplete canonical oracle")
    n, physical = len(prediction["ids"]), len(prediction["observation_indices"])
    if not n or not physical or prediction["events"].ndim != 3 or prediction["events"].shape[0] != physical or prediction["events"].shape[-1] != 3:
        raise ValueError("Invalid physical oracle shape")
    for key in ("ids", "observation_indices", "query_observation_indices", "goal_indices"):
        arr = np.asarray(prediction[key])
        if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer) or np.any(arr < 0):
            raise ValueError("Invalid canonical ID array: " + key)
    for key in ("ids", "observation_indices"):
        if not np.array_equal(prediction[key], np.unique(prediction[key])):
            raise ValueError("Canonical IDs must be sorted and unique")
    if any(len(prediction[k]) != n for k in ("query_observation_indices", "goal_indices", "scores", "components", "decisions")) or prediction["components"].shape != (n, 4):
        raise ValueError("Misaligned oracle query arrays")
    if np.any(prediction["query_observation_indices"] >= physical):
        raise ValueError("Query references a missing physical observation")
    for key in ("scores", "components", "events", "threshold", "zero", "mean", "conditional"):
        if not np.isfinite(prediction[key]).all():
            raise ValueError("Nonfinite canonical array: " + key)
    if np.asarray(prediction["threshold"]).ndim != 0:
        raise ValueError("Final production uses one frozen validation threshold")
    if np.asarray(prediction["decisions"]).dtype != np.dtype(bool) or not np.array_equal(prediction["decisions"], prediction["scores"] >= prediction["threshold"]):
        raise ValueError("Canonical decisions differ from frozen threshold")
    for key in ("zero", "mean", "conditional"):
        if prediction[key].shape != prediction["events"].shape[:2]:
            raise ValueError("Misaligned distribution summary: " + key)


def classification_from_arrays(labels, scores, users, threshold, probabilities=None):
    """Original point definitions; also usable as a synthetic metric adapter."""
    labels, scores, users = np.asarray(labels), np.asarray(scores), np.asarray(users).astype(str)
    decision = scores >= threshold
    if labels.shape != scores.shape or users.shape != scores.shape or np.any(~np.isin(labels, (0, 1))):
        raise ValueError("Invalid classification arrays")
    per = []
    for user in sorted(set(users)):
        take = users == user; y, s, d = labels[take], scores[take], decision[take]
        tp = int(np.sum((y == 1) & d)); tn = int(np.sum((y == 0) & ~d))
        fp = int(np.sum((y == 0) & d)); fn = int(np.sum((y == 1) & ~d))
        eligible = bool(tp + fn and tn + fp)
        per.append(dict(user_id=user, n=len(y), f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                        ba=(tp / (tp + fn) + tn / (tn + fp)) / 2 if eligible else None,
                        auroc=float(roc_auc_score(y, s)) if eligible else None,
                        ap=float(average_precision_score(y, s)) if eligible else None,
                        accuracy=float(np.mean(d == y))))
    macro = {k: mean_defined(v[k] for v in per) for k in PRIMARY + ("accuracy",)}
    macro.update(users=len(per), valid_auc_users=sum(v["auroc"] is not None for v in per),
                 valid_ap_users=sum(v["ap"] is not None for v in per), valid_ba_users=sum(v["ba"] is not None for v in per))
    result = dict(n=len(labels), positive_rate=float(np.mean(labels)), macro=macro, per_user=per,
                  ap_per_user={v["user_id"]: v["ap"] for v in per if v["ap"] is not None},
                  accuracy_per_user={v["user_id"]: v["accuracy"] for v in per},
                  ap_definition="average_precision_score of continuous endpoint score; equal-user macro excludes single-class users")
    if probabilities is not None:
        p = np.clip(np.asarray(probabilities), 1e-7, 1 - 1e-7)
        ece = 0.
        for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            take = (p >= a) & (p < (b if b < 1 else 1.00001))
            if take.any():
                ece += float(take.mean() * abs(labels[take].mean() - p[take].mean()))
        result.update(brier=float(np.mean((labels - p) ** 2)), ece_10_equal_bins=ece,
                      calibration_weighting="pooled task rows within this condition, not user macro")
    return result


def augment_original_classification(original, labels, scores, users, threshold, probabilities):
    calculated = classification_from_arrays(labels, scores, users, threshold, probabilities)
    saved = {v["user_id"]: v for v in original["per_user"]}
    if set(saved) != set(map(str, users)):
        raise ValueError("Original classifier user set differs")
    for row in calculated["per_user"]:
        for key in ("f1", "ba", "auroc"):
            a, b = row[key], saved[row["user_id"]][key]
            if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-12):
                raise ValueError("Original classifier definition mismatch")
        saved[row["user_id"]].update(ap=row["ap"], accuracy=row["accuracy"])
    for key in ("brier", "ece_10_equal_bins"):
        if abs(original[key] - calculated[key]) > 1e-12:
            raise ValueError("Original calibration definition mismatch")
    original["macro"].update({k: calculated["macro"][k] for k in ("ap", "accuracy", "valid_ap_users", "valid_ba_users")})
    original.update({k: calculated[k] for k in ("ap_per_user", "accuracy_per_user", "ap_definition", "calibration_weighting")})
    return original


def opposite_goal_metrics(prediction, rows):
    grouped = defaultdict(list)
    for j, task in enumerate(prediction["ids"]):
        row = rows[int(task)]
        grouped[(row["condition"], row["user_id"], row["observation_index"])].append(j)
    per = defaultdict(lambda: defaultdict(list))
    for (condition, user, observation), indices in grouped.items():
        negative = [j for j in indices if rows[int(prediction["ids"][j])]["label"] == 0]
        positive = [j for j in indices if rows[int(prediction["ids"][j])]["label"] == 1]
        if not negative or not positive:
            continue
        n, p = np.asarray(negative), np.asarray(positive)
        delta = prediction["scores"][p, None] - prediction["scores"][n][None, :]
        pd, nd = prediction["decisions"][p, None], prediction["decisions"][n][None, :]
        per[condition][user].append(dict(observation_index=int(observation), pairs=len(n) * len(p),
            both_correct=float(np.mean(pd & ~nd)), concordance=float(np.mean((delta > 0) + .5 * (delta == 0))),
            prediction_flip_rate=float(np.mean(pd != nd))))
    output = {}
    for condition in dict.fromkeys(rows[int(i)]["condition"] for i in prediction["ids"]):
        user_records = []
        for user, values in sorted(per[condition].items()):
            user_records.append(dict(user_id=user, observations=len(values), pairs=sum(v["pairs"] for v in values),
                **{k: float(np.mean([v[k] for v in values])) for k in ("both_correct", "concordance", "prediction_flip_rate")}))
        output[condition] = dict(per_user=user_records,
            **{"macro_" + k: mean_defined(v[k] for v in user_records) for k in ("both_correct", "concordance", "prediction_flip_rate")},
            pairs=sum(v["pairs"] for v in user_records), users=len(user_records),
            mixed_label_observations=sum(v["observations"] for v in user_records),
            observations=[dict(user_id=u, **v) for u, values in sorted(per[condition].items()) for v in values],
            weighting="cross-label pairs per observation; equal observations per user; equal eligible users; tie concordance=0.5")
    return output


def evaluate_prediction(prediction, probabilities, metadata, metric_data, evaluation_api, order_probs=None):
    """Truth arguments are supplied only after the complete oracle is sealed."""
    validate_prediction(prediction)
    rows, observations, goals, manifest = (metadata[k] for k in ("rows", "observations", "goals", "manifest"))
    ids = prediction["ids"]; physical = prediction["observation_indices"]
    query_obs = physical[prediction["query_observation_indices"]]
    if any(rows[int(i)]["observation_index"] != int(o) or rows[int(i)]["goal_index"] != int(g)
           for i, o, g in zip(ids, query_obs, prediction["goal_indices"])):
        raise ValueError("Task oracle differs from metadata identity")
    events = np.zeros((len(observations),) + prediction["events"].shape[1:], dtype=prediction["events"].dtype)
    events[physical] = prediction["events"]
    expanded_probs = None
    if order_probs is not None:
        if order_probs.shape != metric_data["pair_gap"][physical].shape or not np.isfinite(order_probs).all():
            raise ValueError("Invalid fixed-pair probability diagnostic")
        expanded_probs = np.zeros_like(metric_data["pair_gap"]); expanded_probs[physical] = order_probs
    pairs = opposite_goal_metrics(prediction, rows)
    records = {}
    conditions = np.asarray([rows[int(i)]["condition"] for i in ids])
    for condition in dict.fromkeys(conditions):
        take = conditions == condition; chosen = ids[take]
        labels = np.asarray([rows[int(i)]["label"] for i in chosen]); users = np.asarray([rows[int(i)]["user_id"] for i in chosen])
        base = evaluation_api.classification(chosen, prediction["scores"][take], float(prediction["threshold"]), rows, probabilities[take])
        records[str(condition)] = dict(
            classification=augment_original_classification(base, labels, prediction["scores"][take], users, float(prediction["threshold"]), probabilities[take]),
            events=evaluation_api.event_metrics(chosen, events[query_obs[take]], metric_data, rows, observations, goals),
            order=evaluation_api.order_metrics(query_obs[take], events, metric_data, observations, manifest, expanded_probs,
                "TAD temporal distribution on fixed GT first-entity pairs; untimed diagnostic" if expanded_probs is not None else
                "common adapted scalar auxiliary head: strict point comparison, binary pseudo-probability"),
            goal_pairs=dict(macro={k: pairs[str(condition)]["macro_" + source] for k, source in
                (("both_correct", "both_correct"), ("pair_concordance", "concordance"), ("prediction_flip_rate", "prediction_flip_rate"))},
                per_user={r["user_id"]: dict(both_correct=r["both_correct"], pair_concordance=r["concordance"], prediction_flip_rate=r["prediction_flip_rate"]) for r in pairs[str(condition)]["per_user"]},
                weighting=pairs[str(condition)]["weighting"]))
    aggregate = {k: mean_defined(r["classification"]["macro"][k] for r in records.values()) for k in PRIMARY + ("accuracy",)}
    aggregate.update({k: mean_defined(r["classification"][source] for r in records.values()) for k, source in (("brier", "brier"), ("ece", "ece_10_equal_bins"))})
    for key, path in {"pair": ("goal_pairs", "both_correct"), "concordance": ("goal_pairs", "pair_concordance"),
                      "visit": ("events", "visit_f1_visible"), "first_time_mae_s": ("events", "first_time_mae_s"),
                      "accessibility_mae_m": ("events", "accessibility_mae_m"), "order": ("order", "end_to_end_pair_accuracy")}.items():
        aggregate[key] = mean_defined(r[path[0]]["macro"][path[1]] for r in records.values())
    for key in ("f1", "ba"):
        if {"gps_30", "sampling_8"} <= set(records):
            aggregate["severe_" + key] = mean_defined(records[c]["classification"]["macro"][key] for c in ("gps_30", "sampling_8"))
    aggregate["primary_four_metric_mean"] = mean_defined(aggregate[k] for k in PRIMARY)
    return dict(conditions=records, aggregate=aggregate, opposite_goal=pairs,
                metric_definitions=dict(ap="continuous-score average precision; two-class-user macro",
                    brier_ece="pooled task rows within condition; equal conditions for aggregate",
                    goals="pairs within observation, observations within user, equal users",
                    recovery="original scope-weighted per-user functions; true visible visits include predicted misses",
                    ordering="original unique observation/category pairs then user macro"))


def audit_trace(prediction, common_scores, common_components, goals, method, scorer):
    validate_prediction(prediction)
    common_scores, common_components = np.asarray(common_scores), np.asarray(common_components)
    if common_scores.shape != prediction["scores"].shape or common_components.shape != prediction["components"].shape or not np.isfinite(common_scores).all() or not np.isfinite(common_components).all():
        raise ValueError("Invalid common-score diagnostic")
    out = dict(status="PASS", method=method, primary_score_hash=array_hash(prediction["scores"]),
               primary_component_hash=array_hash(prediction["components"]), common_score_hash=array_hash(common_scores),
               common_component_hash=array_hash(common_components), decisions_from_frozen_threshold=True)
    if method != "tad":
        if any(k in prediction for k in ("raw_v3_scores", "refined_violation_contributions")):
            raise ValueError("External primary endpoint cannot acquire TAD refinement")
        out.update(primary_semantics="unchanged learned GoalAdapter logits with observable-prior trace",
                   common_semantics="explicit score_v3 from adapted predicted events; separately calibrated diagnostic",
                   raw_component_equality_required=False)
        return out
    mask = np.asarray([[p in goals[int(g)]["active_components"] for p in PARTS] for g in prediction["goal_indices"]], bool)
    if np.any(mask.sum(-1) == 0) or np.any((common_components < 0) | (common_components > 1)):
        raise ValueError("Invalid raw executable components")
    raw = prediction.get("raw_v3_scores", prediction["scores"])
    out["max_raw_score_difference"] = float(np.max(abs(raw - common_scores)))
    out["max_raw_component_difference"] = float(np.max(abs(prediction["components"] - common_components)))
    if not np.allclose(raw, common_scores, atol=1e-12, rtol=0) or not np.allclose(prediction["components"], common_components, atol=1e-12, rtol=0):
        raise ValueError("TAD captured raw V3 score/trace differs from canonical score_v3")
    if scorer.get("kind", "raw_v3") == "raw_v3":
        if not np.array_equal(prediction["scores"] >= prediction["threshold"], common_scores >= prediction["threshold"]):
            raise ValueError("Raw canonical scorer produces threshold flips")
        out.update(primary_semantics="raw executable V3", refinement=False, raw_threshold_flips=0)
        return out
    required = ("raw_v3_scores", "refined_violation_contributions", "normalized_component_weights", "optional_soft_components")
    if set(required) - set(prediction):
        raise ValueError("Refined primary lacks auditable saved trace")
    weight = np.asarray(scorer["global_weights"], np.float64)
    if weight.shape != (4,) or np.any(weight <= 0) or not np.isfinite(weight).all():
        raise ValueError("Final refinement requires one positive global four-vector")
    expected_weights = mask * weight; expected_weights /= expected_weights.sum(-1, keepdims=True)
    normalized, soft, contributions = (prediction[k] for k in required[2:] + required[1:2])
    if any(v.shape != common_components.shape for v in (normalized, soft, contributions)) or not all(np.isfinite(v).all() for v in (normalized, soft, contributions)):
        raise ValueError("Invalid refinement trace shape/value")
    if np.any((soft < 0) | (soft > 1)) or not np.array_equal(normalized, expected_weights):
        raise ValueError("Refinement weights/proxy bounds disagree with frozen scorer")
    mixing = float(scorer.get("mixing", 0))
    expected = normalized * (1 - ((1 - mixing) * common_components + mixing * soft))
    if not 0 <= mixing <= 1 or not np.allclose(contributions, expected, atol=1e-12, rtol=0) or not np.array_equal(prediction["scores"], contributions.sum(-1)):
        raise ValueError("Refinement additive trace disagrees with primary score")
    out.update(primary_semantics="positive component refinement of raw V3/proxy satisfaction", refinement=True,
               exact_additive_primary_trace=True, max_contribution_formula_difference=float(np.max(abs(contributions - expected))),
               changed_decisions_vs_raw_at_primary_threshold=int(np.sum(prediction["decisions"] != (raw >= prediction["threshold"]))))
    return out

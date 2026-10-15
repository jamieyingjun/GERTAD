"""Observed visit evidence and explicit executable goal formulas."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from typing import Mapping, Sequence
import unicodedata
from .goals import StructuredGoalV3
COUNT_SEMANTICS = 'DISTINCT_TARGET_ENTITIES'
GEOMETRY_COST_SEMANTICS = 'frozen_insertion_extra_distance_d2_m_not_road_travel_time'
EQUAL_WEIGHT_AGGREGATION = 'EQUAL_WEIGHT_ARITHMETIC_MEAN'


def _finite_nonnegative(value: float, field: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f'{field} must be finite and nonnegative')
    return number

def normalize_category(value: str) -> str:
    """Return the frozen lexical form used by goals and observed evidence."""
    normalized = ' '.join(unicodedata.normalize('NFKC', str(value)).split())
    if not normalized:
        raise ValueError('category cannot be empty after normalization')
    return normalized

@dataclass(frozen=True, slots=True)
class VisitEvidence:
    entity_id: str
    category: str
    first_visit_index: int
    geometry_cost: float | None
    raw_category: str | None = None

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError('entity_id is required')
        raw_category = self.category if self.raw_category is None else self.raw_category
        normalized = normalize_category(raw_category)
        object.__setattr__(self, 'raw_category', str(raw_category))
        object.__setattr__(self, 'category', normalized)
        if isinstance(self.first_visit_index, bool) or int(self.first_visit_index) < 0:
            raise ValueError('first_visit_index must be a nonnegative integer')
        if self.geometry_cost is not None:
            object.__setattr__(self, 'geometry_cost', _finite_nonnegative(self.geometry_cost, 'geometry_cost'))

@dataclass(frozen=True, slots=True)
class TrajectoryTaskEvidence:
    context_id: str
    visits: tuple[VisitEvidence, ...]
    evidence_source: str = 'observed_trajectory_and_frozen_map_semantics'
    geometry_cost_semantics: str = GEOMETRY_COST_SEMANTICS
    bound_goal_id: str | None = None
    bound_context_id: str | None = None
    pairing_source: str | None = None

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ValueError('context_id is required')
        by_id: dict[str, VisitEvidence] = {}
        for visit in self.visits:
            old = by_id.get(visit.entity_id)
            if old is None:
                by_id[visit.entity_id] = visit
                continue
            if old.category != visit.category:
                raise ValueError('duplicate entity_id has inconsistent categories')
            if old.geometry_cost is None or visit.geometry_cost is None:
                geometry_equal = old.geometry_cost is visit.geometry_cost
            else:
                geometry_equal = isclose(old.geometry_cost, visit.geometry_cost, rel_tol=0.0, abs_tol=1e-09)
            if not geometry_equal:
                raise ValueError('duplicate entity_id has inconsistent geometry costs')

    def unique_visits(self) -> tuple[VisitEvidence, ...]:
        """Return one deterministic record per entity, keeping its first visit."""
        by_id: dict[str, VisitEvidence] = {}
        for visit in self.visits:
            old = by_id.get(visit.entity_id)
            if old is None or (visit.first_visit_index, visit.entity_id) < (old.first_visit_index, old.entity_id):
                by_id[visit.entity_id] = visit
        return tuple(sorted(by_id.values(), key=lambda item: (item.first_visit_index, item.entity_id)))

def _clip(value: float) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError('scores must be finite')
    return min(1.0, max(0.0, number))

def count_satisfaction(visited_target_count: int, required_count: int) -> float:
    if required_count < 1:
        raise ValueError('required_count must be positive')
    return _clip(visited_target_count / required_count)

def order_satisfaction(first_visit: Mapping[str, int], ordered_categories: tuple[str, ...]) -> float:
    if len(ordered_categories) < 2:
        raise ValueError('order requires at least two categories')
    satisfied = 0
    for left, right in zip(ordered_categories, ordered_categories[1:]):
        if left in first_visit and right in first_visit and (first_visit[left] < first_visit[right]):
            satisfied += 1
    return _clip(satisfied / (len(ordered_categories) - 1))

def accessibility_satisfaction(geometry_costs: tuple[float, ...], budget: float) -> float:
    if not isfinite(float(budget)) or budget <= 0.0:
        raise ValueError('budget must be finite and positive')
    if any((not isfinite(float(value)) or value < 0.0 for value in geometry_costs)):
        raise ValueError('geometry costs must be finite and nonnegative')
    if not geometry_costs:
        return 0.0
    excess = sum((max(float(value) - budget, 0.0) for value in geometry_costs))
    return _clip(1.0 - excess / (max(len(geometry_costs), 1) * budget))

def score_v3(goal: StructuredGoalV3, evidence: TrajectoryTaskEvidence) -> dict[str, object]:
    """Apply the frozen four component formulas under the clarified V3 scope."""
    scope = set(goal.target_scope_categories)
    visits = tuple((item for item in evidence.unique_visits() if item.category in scope))
    categories = {item.category for item in visits}
    first: dict[str, int] = {}
    for visit in visits:
        first[visit.category] = min(first.get(visit.category, visit.first_visit_index), visit.first_visit_index)
    scores: dict[str, float] = {}
    if 'count' in goal.active_components:
        scores['count'] = count_satisfaction(len(visits), int(goal.required_count))
    if 'category' in goal.active_components:
        required = set(goal.required_categories)
        scores['category'] = len(categories & required) / len(required)
    if 'order' in goal.active_components:
        scores['order'] = order_satisfaction(first, goal.ordered_categories)
    if 'accessibility' in goal.active_components:
        scores['accessibility'] = accessibility_satisfaction(tuple((float(item.geometry_cost) for item in visits)), float(goal.accessibility_budget))
    satisfaction = sum(scores.values()) / len(scores)
    return {'goal_id': goal.goal_id, 'context_id': evidence.context_id, 'active_components': list(goal.active_components), 'component_scores': scores, 'composite_satisfaction': satisfaction, 'anomaly_score': 1.0 - satisfaction, 'aggregation': EQUAL_WEIGHT_AGGREGATION, 'count_semantics': 'DISTINCT_TARGET_ENTITIES', 'target_scope_categories': list(goal.target_scope_categories), 'required_categories': list(goal.required_categories), 'target_entity_count': len(visits)}

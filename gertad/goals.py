"""Structured executable goal contracts."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Mapping
import math
import unicodedata
COMPONENT_ORDER = ('count', 'category', 'order', 'accessibility')
TEMPLATES = ('COUNT_ONLY', 'CATEGORY_ONLY', 'ORDER_ONLY', 'ACCESSIBILITY_ONLY', 'COMPOSITE')

def normalize_category(value: str) -> str:
    """Return the frozen lexical form used by goals and observed evidence."""

    normalized = " ".join(unicodedata.normalize("NFKC", str(value)).split())
    if not normalized:
        raise ValueError("category cannot be empty after normalization")
    return normalized

@dataclass(frozen=True, slots=True)
class StructuredGoalV3:
    goal_id: str
    active_components: tuple[str, ...]
    target_scope_categories: tuple[str, ...]
    required_categories: tuple[str, ...] = ()
    required_count: int | None = None
    ordered_categories: tuple[str, ...] = ()
    accessibility_budget: float | None = None
    goal_template: str = "COMPOSITE"

    def __post_init__(self) -> None:
        if not self.goal_id:
            raise ValueError("goal_id is required")
        active = tuple(self.active_components)
        expected = tuple(item for item in COMPONENT_ORDER if item in active)
        if active != expected or len(active) != len(set(active)) or not active:
            raise ValueError("active_components must be nonempty, unique, and in canonical order")
        scope = tuple(normalize_category(item) for item in self.target_scope_categories)
        required = tuple(normalize_category(item) for item in self.required_categories)
        ordered = tuple(normalize_category(item) for item in self.ordered_categories)
        object.__setattr__(self, "target_scope_categories", scope)
        object.__setattr__(self, "required_categories", required)
        object.__setattr__(self, "ordered_categories", ordered)
        if not scope or len(scope) != len(set(scope)):
            raise ValueError("target_scope_categories must be nonempty and unique")
        if len(required) != len(set(required)) or not set(required).issubset(scope):
            raise ValueError("required_categories must be unique and a subset of target_scope_categories")
        if len(ordered) != len(set(ordered)):
            raise ValueError("ordered_categories cannot repeat under first-visit order semantics")
        if not set(ordered).issubset(scope):
            raise ValueError("ordered_categories must be a subset of target_scope_categories")
        if "count" in active:
            if isinstance(self.required_count, bool) or not isinstance(self.required_count, int) or self.required_count < 1:
                raise ValueError("active count requires required_count as a positive integer")
        elif self.required_count is not None:
            raise ValueError("inactive count requires required_count=null")
        if "category" in active:
            if not required:
                raise ValueError("active category requires nonempty required_categories")
        elif required:
            raise ValueError("inactive category requires required_categories=[]")
        if "order" in active:
            if len(ordered) < 2:
                raise ValueError("active order requires at least two distinct ordered_categories")
        elif ordered:
            raise ValueError("inactive order requires ordered_categories=[]")
        if "accessibility" in active:
            if self.accessibility_budget is None or not math.isfinite(float(self.accessibility_budget)) or float(self.accessibility_budget) <= 0:
                raise ValueError("active accessibility requires a positive finite accessibility_budget")
        elif self.accessibility_budget is not None:
            raise ValueError("inactive accessibility requires accessibility_budget=null")
        if self.goal_template not in TEMPLATES:
            raise ValueError("goal_template is not in the frozen template vocabulary")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["active_components"] = list(self.active_components)
        value["target_scope_categories"] = list(self.target_scope_categories)
        value["required_categories"] = list(self.required_categories)
        value["ordered_categories"] = list(self.ordered_categories)
        return value

def goal_v3_from_mapping(value: Mapping[str, object]) -> StructuredGoalV3:
    required_fields = {
        "goal_id",
        "active_components",
        "target_scope_categories",
        "required_categories",
        "required_count",
        "ordered_categories",
        "accessibility_budget",
        "goal_template",
    }
    if set(value) != required_fields:
        raise ValueError("StructuredGoalV3 fields must match the frozen contract exactly")
    return StructuredGoalV3(
        goal_id=str(value["goal_id"]),
        active_components=tuple(str(item) for item in value["active_components"]),
        target_scope_categories=tuple(str(item) for item in value["target_scope_categories"]),
        required_categories=tuple(str(item) for item in value["required_categories"]),
        required_count=None if value["required_count"] is None else value["required_count"],
        ordered_categories=tuple(str(item) for item in value["ordered_categories"]),
        accessibility_budget=None if value["accessibility_budget"] is None else float(value["accessibility_budget"]),
        goal_template=str(value["goal_template"]),
    )

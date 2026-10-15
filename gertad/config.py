"""Frozen physical input schema and ontology."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'config/categories.json').read_text(encoding='utf-8'))
CATEGORIES = tuple(CONFIG['category_ids_in_embedding_order'])
CI = {value: index for index, value in enumerate(CATEGORIES)}
E_MAX = CONFIG['max_entities']
PAD = CONFIG['max_points']
PARTS = ('count', 'category', 'order', 'accessibility')

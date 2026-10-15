"""Frozen LM-TAD backbone loading and native embedding."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from .config import ROOT
VENDOR = ROOT / 'third_party/LMTAD'
AUTHOR_MODEL = VENDOR / 'code/models/LMTAD.py'
AUTHOR_MODEL_SHA256 = '0aabecb6b22c83d9a9e3b6b7123f4be3210aae58c93ded67eb76e854b682b68c'
AUTHOR_MODULE = '_external_v4_official_lmtad'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def author_module():
    """Import unchanged author source without retaining a generic utils binding."""
    if _sha(AUTHOR_MODEL) != AUTHOR_MODEL_SHA256:
        raise RuntimeError('Audited LM-TAD author source changed')
    if AUTHOR_MODULE in sys.modules:
        return sys.modules[AUTHOR_MODULE]
    missing = object()
    previous_utils = sys.modules.get('utils', missing)
    previous_path = list(sys.path)
    try:
        spec_utils = importlib.util.spec_from_file_location('_external_v4_lmtad_utils', VENDOR / 'code/utils.py')
        utils = importlib.util.module_from_spec(spec_utils)
        spec_utils.loader.exec_module(utils)
        sys.modules['utils'] = utils
        sys.path.insert(0, str(VENDOR / 'code'))
        spec = importlib.util.spec_from_file_location(AUTHOR_MODULE, AUTHOR_MODEL)
        module = importlib.util.module_from_spec(spec)
        sys.modules[AUTHOR_MODULE] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(AUTHOR_MODULE, None)
        raise
    finally:
        sys.path[:] = previous_path
        if previous_utils is missing:
            sys.modules.pop('utils', None)
        else:
            sys.modules['utils'] = previous_utils

def _cells(row):
    coordinates = np.asarray(row['coordinates'], dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError('Native LM-TAD requires geographic [latitude, longitude] coordinates')
    if not np.isfinite(coordinates).all():
        raise ValueError('Nonfinite native LM-TAD coordinates')
    if len(coordinates) > 50:
        raise ValueError('Current frozen v2 protocol has at most 50 observation points')
    projected = np.column_stack([coordinates[:, 0] * 111320, coordinates[:, 1] * 111320 * np.cos(np.radians(40))])
    return [tuple(cell) for cell in np.floor(projected / 100).astype(np.int64)]

def embedding(model, batchdict):
    """Differentiable 128-dimensional pooled author representation, no new encoder."""
    tokens = batchdict['lm_tokens'] if isinstance(batchdict, dict) else batchdict
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens, dtype=torch.long)
    tokens = tokens.to(device=next(model.parameters()).device, dtype=torch.long)
    if tokens.ndim != 2:
        raise ValueError('LM-TAD token input must be B x T')
    inputs = tokens[:, :-1] if tokens.shape[1] == 52 else tokens
    if inputs.shape[1] > model.config.block_size:
        raise ValueError('LM-TAD hidden input exceeds audited native context length')
    position = torch.arange(inputs.shape[1], device=inputs.device, dtype=torch.long)
    x = model.transformer.drop(model.transformer.wte(inputs) + model.transformer.wpe(position))
    for block in model.transformer.h:
        x = block(x)
    x = model.transformer.ln_f(x)
    valid = (inputs != model.config.pad_token).to(x.dtype).unsqueeze(-1)
    return (x * valid).sum(1) / valid.sum(1).clamp_min(1)

def load_model(outdir):
    checkpoint = torch.load(Path(outdir) / 'model.pt', map_location='cpu', weights_only=False)
    official = author_module()
    model = official.LMTAD(official.LMTADConfig(**checkpoint['config']))
    model.load_state_dict(checkpoint['state_dict'])
    return model

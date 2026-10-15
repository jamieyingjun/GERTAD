"""Frozen CETrajAD ensemble loading and native embedding."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from .config import ROOT
AUTHOR_MODEL = ROOT / 'third_party/CETrajAD/LSTM_model.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _import(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous_path = sys.path.copy()
    threads = torch.get_num_threads()
    try:
        sys.path.insert(0, str(Path(path).parent))
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.path[:] = previous_path
        torch.set_num_threads(threads)
    return module

def _author():
    return _import('external_v4_cetrajad_author_readonly', AUTHOR_MODEL)

def load_models(outdir):
    """Load only this new output directory's completed native checkpoints."""
    outdir = Path(outdir)
    manifest = json.loads((outdir / 'native_manifest.json').read_text(encoding='utf-8'))
    if manifest['author_encoder_sha256'] != sha(AUTHOR_MODEL):
        raise ValueError('Author encoder source changed since native fit')
    author = _author()
    models = nn.ModuleList()
    specs = manifest['specs']
    for spec in specs:
        saved = torch.load(outdir / spec['checkpoint'], map_location='cpu', weights_only=False)
        if saved.get('data_source_fingerprint') != manifest['data_source_fingerprint'] or not saved.get('from_scratch'):
            raise ValueError('Checkpoint provenance does not match new native training')
        model = author.LSTMAutoencoder(saved['config'])
        model.load_state_dict(saved['state_dict'])
        model.eval()
        models.append(model)
    return (models, specs)

def embedding(models, specs, batchdict):
    """Differentiable 624-dimension concatenation of author final LSTM hiddens.

    Padding is deliberately not changed to packed-sequence pooling.  The caller
    may fine-tune all encoder parameters under the declared goal adaptation.
    """
    if len(models) != len(specs):
        raise ValueError('CETrajAD model/spec mismatch')
    representations = []
    for model, spec in zip(models, specs):
        x = batchdict[spec['view']]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        representations.append(model.encoder(x))
    return torch.cat(representations, dim=-1)

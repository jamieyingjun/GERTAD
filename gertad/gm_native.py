"""Frozen GM-VSAE research-port backbone and native embedding."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from .config import ROOT
AUTHOR_MODEL = ROOT / 'third_party/LMTAD/code/models/GMSVAE.py'
AUTHOR_MODEL_SHA256 = '4430243868d2382745865c7ff286f1861efba358d10c1d8432051447769f2b31'

AUTHOR_COMMIT = '80bb89a8ea108db8f13cb9959826424e9c45f41c'
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 7
POLICY = {'implementation': "unchanged LM-TAD researchers' PyTorch GM-VSAE reimplementation", 'source_repo': 'https://github.com/jonathankabala/LMTAD', 'original_repo': 'https://github.com/liuyiding1993/ICDE2020_GMVSAE', 'source_commit': AUTHOR_COMMIT, 'source_sha256': AUTHOR_MODEL_SHA256, 'tokens': 'existing train-only 100m vocabulary; cells only; BOS/EOS/PAD stripped; UNK retained; ID0 decoder start', 'dim_emb': 32, 'dim_h': 128, 'components': 10, 'optimizer': 'Adam', 'lr': 0.0001, 'betas': [0.5, 0.999], 'batch_size': BATCH_SIZE, 'gradient_clip': 1, 'max_epochs': MAX_EPOCHS, 'patience': PATIENCE, 'native_objective': 'port reconstruction mean + Gaussian mean + uniform loss; all coefficients 1', 'selection': 'validation total native port loss; 10 fixed-RNG Monte Carlo samples', 'score': '10 fixed-RNG Monte Carlo mean masked per-sequence reconstruction NLL; higher anomalous', 'score_order': 'requested observation IDs sorted, chunked in fixed batches of32, output order restored', 'random_policy': 'torch fork_rng; manual_seed(100000+run_seed); no training RNG advance by validation/scoring', 'representation': 'deterministic final GRU encoder hidden128 after Gaussian-mixture native training', 'tf_differences': ['learnable prior log variance', 'full softmax CE', 'all-one native loss weights', 'random mixture initialization without KMeans pretraining', 'recNLL rather than sigmoid native score'], 'test_selection': False}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def policy_hash():
    return hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def author_module():
    if sha(AUTHOR_MODEL) != AUTHOR_MODEL_SHA256:
        raise RuntimeError('Audited research-port source changed')
    name = '_external_v5_gmvsae_research_port'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, AUTHOR_MODEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _tokens(arrays):
    """Strip boundary tokens without fitting or using goal/label fields."""
    tokens = arrays['lm_tokens']
    tokens = tokens.to(dtype=torch.long) if isinstance(tokens, torch.Tensor) else torch.as_tensor(np.asarray(tokens), dtype=torch.long)
    if tokens.ndim != 2 or tokens.shape[1] != 52:
        raise ValueError('Frozen tokenizer must provide Nx52 sequences')
    cells = tokens[:, 1:51].clone()
    cells[cells < 3] = 0
    if not torch.all((cells != 0).sum(1) > 0):
        raise ValueError('Every native observation needs at least one observed cell')
    return cells

def embedding(model, batchdict):
    """Deterministic differentiable GRU hidden, preserving the research encoder."""
    values = _tokens(batchdict).to(device=next(model.parameters()).device)
    lengths = (values != 0).sum(1).cpu()
    values = values[:, :int(lengths.max())].contiguous()
    return model.encoder(values, lengths)[-1]

def _new_model(config):
    source = author_module()
    model = source.GMSVAE(source.GMSVAEConfig(dim_emb=32, dim_h=128, components=10, vocab_size=int(config['vocab_size']), pad_token=0, lr=0.0001))
    model._external_seed = int(config['seed'])
    return model

def load_model(outdir):
    saved = torch.load(Path(outdir) / 'model.pt', map_location='cpu', weights_only=False)
    if saved['policy_sha256'] != policy_hash():
        raise RuntimeError('Saved native GM policy differs')
    model = _new_model(saved['config'])
    model.load_state_dict(saved['state_dict'])
    return model

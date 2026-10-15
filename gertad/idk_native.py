"""Frozen IDK distribution representation; no fitting during evaluation."""
import hashlib
import importlib.machinery
import importlib.util
import sys
import numpy as np
from .config import ROOT
SOURCE = ROOT / 'third_party/IDK/iNN_IK.py'
EXPECTED = '9021e395221fee74727321a543bda1a714969f29bef4ddbaa038458e7e945a91'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def author():
    if sha(SOURCE) != EXPECTED:
        raise AssertionError('Author IDK source hash mismatch')
    import importlib.machinery, importlib.util, sys
    name = 'external_v5_idk_author'
    if name in sys.modules:
        return sys.modules[name]
    loader = importlib.machinery.SourceFileLoader(name, str(SOURCE))
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    loader.exec_module(m)
    return m

def distribution_features(kernel, distributions):
    features = []
    for start in range(0, len(distributions), 96):
        chunk = distributions[start:start + 96]
        flat = np.concatenate(chunk)
        mapped = kernel.transform(flat)
        offset = 0
        for points in chunk:
            features.append(np.asarray(mapped[offset:offset + len(points)].mean(0)).ravel())
            offset += len(points)
    return np.asarray(features, np.float32)

"""Fixed ground-truth pair diagnostic after primary prediction sealing."""
import torch
from .model import ordering


def valid_pair_probs(model, p, b):
    mask = b['pair_valid']
    bi, pi = torch.where(mask)
    prob = p['events'].new_zeros(mask.shape)
    reverse = p['events'].new_zeros(mask.shape)
    if not len(bi):
        return (prob, reverse)
    a, bb = (b['pair_a'][bi, pi], b['pair_b'][bi, pi])
    if not model.distribution:
        ta = p['events'][bi, a, 1] * b['duration'][bi]
        tb = p['events'][bi, bb, 1] * b['duration'][bi]
        q = torch.sigmoid(tb - ta)
        za, zb = (p['zero'][bi, a], p['zero'][bi, bb])
        ab = za * (1 - zb) + (1 - za) * (1 - zb) * q
        ba = zb * (1 - za) + (1 - za) * (1 - zb) * (1 - q)
    else:
        args = [p[k][bi, a] for k in ['pmf', 'mu', 'sigma']] + [p['zero'][bi, a]]
        argsb = [p[k][bi, bb] for k in ['pmf', 'mu', 'sigma']] + [p['zero'][bi, bb]]
        ab = ordering(*args, *argsb)
        ba = ordering(*argsb, *args)
    prob[bi, pi], reverse[bi, pi] = (ab, ba)
    return (prob, reverse)

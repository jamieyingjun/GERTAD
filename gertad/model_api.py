"""Native representation dispatch for the two added adapters."""
from torch import nn
from . import gm_native as gmvsae_native


class FixedKernel(nn.Module):

    def forward(self, b):
        return b['kernel_embedding']

def embedding(kind, model, b):
    if kind == 'gm':
        return gmvsae_native.embedding(model, b)
    return model(b)

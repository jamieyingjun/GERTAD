"""Original seven-term physical training objective."""
import math
import torch
from torch.nn import functional as F

def loss_function(model,p,b):
    truth=b['event_targets'];valid=b['entity_valid'];positive=valid&(truth[:,:,0]>.5)
    def avg(v,mask):return v[mask].mean() if mask.any() else v.sum()*0
    visit=avg(F.binary_cross_entropy_with_logits(p['events'][:,:,0],truth[:,:,0],reduction='none'),valid)
    train_mean=torch.where(truth[:,:,1]<=1e-7,p['mean'],p['conditional'])
    first=avg(F.smooth_l1_loss(train_mean*b['duration'][:,None]/20,truth[:,:,1]*b['duration'][:,None]/20,reduction='none'),positive)
    cost=avg(F.smooth_l1_loss(p['events'][:,:,2]*5,truth[:,:,2]*5,reduction='none'),positive)
    zero_target=(truth[:,:,1]<=1e-7).float()
    onset=avg(F.binary_cross_entropy(p['zero'].clamp(1e-6,1-1e-6),zero_target,reduction='none'),positive)
    nonzero=positive&(truth[:,:,1]>1e-7)
    z=torch.logit(truth[:,:,1].clamp(1e-6,1-1e-6))[:,:,None]
    logdensity=-.5*((z-p['mu'])/p['sigma']).square()-p['sigma'].log()-.5*math.log(2*math.pi)
    logweight=p['pmf'].clamp_min(1e-30).log().masked_fill(p['pmf']==0,-torch.inf)
    # Select supported rows BEFORE logsumexp: all-minus-inf padded mixtures have undefined gradients.
    nll=-torch.logsumexp((logweight+logdensity)[nonzero],1).mean() if model.distribution and nonzero.any() else first*0
    prob,reverse=model.pair_probs(p,b)
    reverse_label=((b['pair_label']<.5)&(b['pair_gap']>1e-7)).float()
    order=avg(.5*(F.binary_cross_entropy(prob.clamp(1e-6,1-1e-6),b['pair_label'],reduction='none')+
        F.binary_cross_entropy(reverse.clamp(1e-6,1-1e-6),reverse_label,reduction='none')),b['pair_valid'])
    count=F.smooth_l1_loss((p['events'][:,:,0].sigmoid()*valid).sum(1),(truth[:,:,0]*valid).sum(1))
    total=visit+first+cost+.1*nll+.2*onset+.1*count+(.2*order if model.order_loss else 0)
    return total,dict(total=total,visit=visit,first=first,cost=cost,nll=nll,onset=onset,count=count,order=order)

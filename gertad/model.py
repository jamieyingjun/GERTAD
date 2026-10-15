"""Shared latent-token physical recovery network."""
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DYNAMICS_FIELDS = (4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17)
ENTITY_RELATIVE_FIELDS = (0, 1, 2, 3, 9)
FROZEN_CONFIG = dict(width=24, layers=1, heads=4, ffn=48,
                     head_width=32, dropout=0.1, latents=8)


def ordering(pa,ma,sa,za,pb,mb,sb,zb):
    # Logistic transform is monotone. Independent marginal approximation, not calibrated joint posterior.
    z=(mb[:,None,:]-ma[:,:,None])/torch.sqrt(sa[:,:,None].square()+sb[:,None,:].square())
    q=(pa[:,:,None]*pb[:,None,:]*(.5+.5*torch.erf(z/math.sqrt(2)))).sum((1,2))
    return za*(1-zb)+(1-za)*(1-zb)*q

class LinearAttentionBlock(nn.Module):
    """Noncausal positive-kernel attention with true linear sequence work.

    phi(q)=ELU(q)+1, normalized q(K^T V)/(q sum(K)+epsilon).
    Intermediates are H*d_head*d_head and T*H*d_head; no T*T array.
    """
    def __init__(self,width,heads,ffn,dropout):
        super().__init__()
        assert width%heads==0
        self.width,self.heads,self.head_dim=width,heads,width//heads
        self.norm1=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width)
        self.output=nn.Linear(width,width)
        self.norm2=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,ffn),nn.GELU(),nn.Dropout(dropout),nn.Linear(ffn,width))
        self.dropout=nn.Dropout(dropout)
        self.attention_disabled=False
    def forward(self,h,mask):
        if not self.attention_disabled:
            B,T,D=h.shape
            q,k,v=self.qkv(self.norm1(h)).reshape(B,T,3,self.heads,self.head_dim).unbind(2)
            q=(F.elu(q)+1).transpose(1,2)
            k=(F.elu(k)+1).transpose(1,2)*mask[:,None,:,None]
            v=v.transpose(1,2)*mask[:,None,:,None]
            kv=torch.einsum('bhtd,bhte->bhde',k,v)
            normalizer=torch.einsum('bhtd,bhd->bht',q,k.sum(2)).clamp_min(1e-6)
            attended=(torch.einsum('bhtd,bhde->bhte',q,kv)/normalizer[:,:,:,None]).transpose(1,2).reshape(B,T,D)
            h=h+self.dropout(self.output(attended))
        h=h+self.dropout(self.ffn(self.norm2(h)))
        return h.masked_fill(~mask[:,:,None],0)

class SharedEncoder(nn.Module):
    def __init__(self,n_categories,entity_dim,kind,width,layers,heads,ffn,dropout,latents=8):
        super().__init__()
        self.kind,self.width,self.heads=kind,width,heads
        self.dynamics_project=nn.Sequential(nn.Linear(13,width),nn.LayerNorm(width),nn.GELU())
        self.relative_project=nn.Sequential(nn.Linear(5,width),nn.LayerNorm(width),nn.GELU())
        self.category=nn.Embedding(n_categories+1,width)
        self.transformer_disabled=False
        if kind=='shared_linear_gate':
            self.temporal=nn.ModuleList(LinearAttentionBlock(width,heads,ffn,dropout) for _ in range(layers))
        else:
            layer=nn.TransformerEncoderLayer(width,heads,ffn,dropout,batch_first=True,norm_first=True,activation='gelu')
            self.temporal=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        if kind=='shared_latent':
            self.latent_queries=nn.Parameter(torch.randn(latents,width)*0.02)
            self.to_latents=nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
            self.to_time=nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
            self.latent_norm=nn.LayerNorm(width)
        if kind=='shared_cross':
            self.cross_q=nn.Linear(entity_dim,width)
            self.cross_k=nn.Linear(width,width)
            self.cross_v=nn.Linear(width,width)
            self.cross_o=nn.Linear(width,width)
        if kind in ('shared_gate','shared_linear_gate','shared_latent'):
            self.gate=nn.Linear(width,width)
            self.fusion_norm=nn.LayerNorm(width)
        else:
            self.fusion=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.LayerNorm(width))
        if kind=='shared_local_module':
            self.entity_norm=nn.LayerNorm(width)
            self.entity_local=nn.Conv1d(width,width,3,padding=1,groups=width)
            self.entity_mix=nn.Linear(width,width)
        self.last_shared_batch_shape=None
        self.last_valid_entities=None
    def forward(self,b,idx):
        x=b['x'];B,E,T,_=x.shape
        first=b['entity_valid'].long().argmax(1)
        dynamic=x[torch.arange(B,device=x.device),first][:,:,DYNAMICS_FIELDS]
        mask=torch.arange(T,device=x.device)[None]<b['lengths'][:,None]
        shared=self.dynamics_project(dynamic).masked_fill(~mask[:,:,None],0)
        self.last_shared_batch_shape=(B,T,self.width)
        self.last_valid_entities=int(idx.numel())
        if not self.transformer_disabled:
            if self.kind=='shared_linear_gate':
                for block in self.temporal:
                    shared=block(shared,mask)
            elif self.kind=='shared_latent':
                queries=self.latent_queries[None].expand(B,-1,-1)
                latent=self.to_latents(queries,shared,shared,key_padding_mask=~mask,need_weights=False)[0]
                latent=self.temporal(self.latent_norm(latent))
                shared=shared+self.to_time(shared,latent,latent,need_weights=False)[0]
            else:
                shared=self.temporal(shared,src_key_padding_mask=~mask)
        shared=shared.masked_fill(~mask[:,:,None],0)
        bi=torch.div(idx,E,rounding_mode='floor')
        relative=x.reshape(B*E,T,18)[idx][:,:,ENTITY_RELATIVE_FIELDS]
        cats=b['entity_category'].reshape(-1)[idx].clamp_min(0)
        r=self.relative_project(relative)+self.category(cats)[:,None]
        h=shared[bi]
        if self.kind=='shared_cross':
            # Shared K/V projection once per observation, one query per entity.
            dh=self.width//self.heads
            q=self.cross_q(b['entity'].reshape(B*E,-1)[idx]).reshape(-1,self.heads,dh)
            k=self.cross_k(shared).reshape(B,T,self.heads,dh)[bi]
            v=self.cross_v(shared).reshape(B,T,self.heads,dh)[bi]
            logits=torch.einsum('nhd,nthd->nht',q,k)/math.sqrt(dh)
            a=torch.softmax(logits.masked_fill(~mask[bi,None,:],-1e9),-1)
            context=self.cross_o(torch.einsum('nht,nthd->nhd',a,v).reshape(-1,self.width))
            h=h+context[:,None]
        if hasattr(self,'gate'):
            h=self.fusion_norm(torch.sigmoid(self.gate(r))*h+r)
        else:
            h=self.fusion(torch.cat([h,r],-1))
        h=h.masked_fill(~mask[bi,:,None],0)
        if self.kind=='shared_local_module':
            z=self.entity_norm(h).masked_fill(~mask[bi,:,None],0)
            z=self.entity_local(z.transpose(1,2)).transpose(1,2)
            h=(h+self.entity_mix(F.gelu(z))).masked_fill(~mask[bi,:,None],0)
        return h

class Phase3Recovery(nn.Module):
    def __init__(self,n_categories,entity_dim,architecture,width=32,layers=1,heads=4,ffn=64,head_width=32,dropout=0.1,latents=8):
        nn.Module.__init__(self)
        self.kind=self.architecture=architecture
        self.distribution=self.order_loss=True
        self.width=width
        self.encoder=SharedEncoder(n_categories,entity_dim,architecture,width,layers,heads,ffn,dropout,latents)
        self.pool_head=nn.Sequential(nn.Linear(2*width+entity_dim,head_width),nn.GELU(),nn.Linear(head_width,4))
        self.time_head=nn.Sequential(nn.Linear(2*width,head_width),nn.GELU(),nn.Linear(head_width,3))
        for head in (self.pool_head,self.time_head):
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        gx,gw=np.polynomial.hermite.hermgauss(8)
        self.register_buffer('gh_x',torch.tensor(gx,dtype=torch.float32))
        self.register_buffer('gh_w',torch.tensor(gw/np.sqrt(np.pi),dtype=torch.float32))
    def forward(self,b):
        x=b['x'];B,E,T,_=x.shape
        idx=torch.where(b['entity_valid'].reshape(-1))[0]
        events=x.new_zeros(B*E,3);pmf=x.new_zeros(B*E,T)
        mu=x.new_zeros(B*E,T);sigma=x.new_ones(B*E,T)
        zero=x.new_zeros(B*E);mean=x.new_zeros(B*E);conditional=x.new_zeros(B*E)
        if len(idx):
            xx=x.reshape(B*E,T,18)[idx]
            lengths=b['lengths'][:,None].expand(B,E).reshape(-1)[idx]
            base=b['base'].reshape(B*E,3)[idx]
            ef=b['entity'].reshape(B*E,-1)[idx]
            anchors=b['times'][:,None].expand(B,E,T).reshape(B*E,T)[idx]
            mask=torch.arange(T,device=x.device)[None]<lengths[:,None]
            h=self.encoder(b,idx)
            average=h.sum(1)/lengths[:,None]
            weights=(xx[:,:,9]+0.01)*mask
            weights=weights/weights.sum(1,keepdim=True)
            local=(h*weights[:,:,None]).sum(1)
            pool=torch.cat([average,local],1)
            r=self.pool_head(torch.cat([pool,ef],1))
            visit=base[:,0]+r[:,0]
            cost=(base[:,2]+r[:,1]).clamp_min(0)
            zprior=torch.where(base[:,1]<=1e-7,6.0,-6.0)
            pzero=torch.sigmoid(zprior+r[:,2])
            raw=self.time_head(torch.cat([h,pool[:,:self.width,None].transpose(1,2).expand(-1,T,-1)],2))
            nearest=(anchors-base[:,1,None]).abs().masked_fill(~mask,1e6).argmin(1)
            initial=anchors.clone()
            initial[torch.arange(len(idx),device=x.device),nearest]=base[:,1].clamp(1e-6,1-1e-6)
            latent=torch.logit(initial.clamp(0.001,0.999))+raw[:,:,1]
            std=(F.softplus(raw[:,:,2]-2.3)+0.005).clamp_max(5)
            dur=b['duration'][:,None].expand(B,E).reshape(-1)[idx]
            bandwidth=torch.maximum(2/dur,torch.tensor(0.002,device=x.device))
            prior=-((anchors-base[:,1,None])/bandwidth[:,None]).square().clamp_max(20)
            pp=torch.softmax((prior+raw[:,:,0]).masked_fill(~mask,-1e9),1)
            continuous=(torch.sigmoid(latent[:,:,None]+math.sqrt(2)*std[:,:,None]*self.gh_x)*self.gh_w).sum(-1)
            cm=(pp*continuous).sum(1)
            first=torch.where(pzero>=0.5,torch.zeros_like(cm),cm)
            events[idx]=torch.stack([visit,first,cost],1)
            pmf[idx]=pp;mu[idx]=latent;sigma[idx]=std;zero[idx]=pzero
            mean[idx]=(1-pzero)*cm;conditional[idx]=cm
        return dict(events=events.reshape(B,E,3),pmf=pmf.reshape(B,E,T),mu=mu.reshape(B,E,T),sigma=sigma.reshape(B,E,T),zero=zero.reshape(B,E),mean=mean.reshape(B,E),conditional=conditional.reshape(B,E))

    def pair_probs(self,p,b):
            B,P=b['pair_a'].shape;bi=torch.arange(B,device=p['events'].device)[:,None].expand(B,P);a=b['pair_a'];bb=b['pair_b']
            if not self.distribution:
                ta=p['events'][bi,a,1]*b['duration'][:,None];tb=p['events'][bi,bb,1]*b['duration'][:,None]
                q=torch.sigmoid((tb-ta)/1.)
                za=p['zero'][bi,a];zb=p['zero'][bi,bb]
                return za*(1-zb)+(1-za)*(1-zb)*q,zb*(1-za)+(1-za)*(1-zb)*(1-q)
            args=[p[k][bi,a].reshape(B*P,-1) for k in ['pmf','mu','sigma']]+[p['zero'][bi,a].reshape(-1)]
            argsb=[p[k][bi,bb].reshape(B*P,-1) for k in ['pmf','mu','sigma']]+[p['zero'][bi,bb].reshape(-1)]
            ab=ordering(*args,*argsb).reshape(B,P);ba=ordering(*argsb,*args).reshape(B,P)
            return ab,ba

def build_model(n_categories=91, entity_dim=102, config=None):
    options = dict(FROZEN_CONFIG if config is None else config)
    if options != FROZEN_CONFIG or n_categories != 91 or entity_dim != 102:
        raise ValueError('This release requires the frozen shared-latent configuration.')
    model = Phase3Recovery(n_categories, entity_dim, 'shared_latent', **options)
    model.architecture_config = options
    return model

"""Reusable evidence indexes and raw executable constraint scores."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import CATEGORIES, CI, PARTS
from .goals import goal_v3_from_mapping
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class CompiledGoals:
    def __init__(self,goals):
        self.contracts=[goal_v3_from_mapping(g) for g in goals]
        self.identity=hashlib.sha256(json.dumps([g.to_dict() for g in self.contracts],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        self.implementation_hash=sha(Path(__file__))
        C=len(CATEGORIES);G=len(goals)
        self.active=np.asarray([[p in g.active_components for p in PARTS] for g in self.contracts],bool)
        self.scope=np.zeros((G,C),bool);self.required=np.zeros((G,C),bool)
        P=max(1,max(len(g.ordered_categories)-1 for g in self.contracts))
        self.left=np.zeros((G,P),np.int64);self.right=self.left.copy();self.pair_valid=np.zeros((G,P),bool)
        self.count=np.asarray([g.required_count or 1 for g in self.contracts],np.float64)
        self.budget=np.asarray([g.accessibility_budget or 1 for g in self.contracts],np.float64)
        for i,g in enumerate(self.contracts):
            self.scope[i,[CI[k] for k in g.target_scope_categories]]=True
            self.required[i,[CI[k] for k in g.required_categories]]=True
            for j,(a,b) in enumerate(zip(g.ordered_categories,g.ordered_categories[1:])):
                self.left[i,j]=CI[a];self.right[i,j]=CI[b];self.pair_valid[i,j]=True
        self.active_count=self.active.sum(1);self.required_count=self.required.sum(1)
        self.order_count=self.pair_valid.sum(1)

class EvidenceIndex:
    def __init__(self,events,categories,valid):
        events=np.asarray(events);self.valid=np.asarray(valid,bool)
        self.cat=np.asarray(categories,np.int64).clip(0)
        self.visit=(events[:,:,0]>=0)&self.valid
        # Frozen scorer multiplies a float32 vector BEFORE float conversion.
        self.cost=(events[:,:,2]*300).astype(np.float64)
        B,E=self.cat.shape;C=len(CATEGORIES)
        self.count=np.zeros((B,C),np.int64);self.first=np.full((B,C),np.inf,np.float64)
        bi=np.broadcast_to(np.arange(B)[:,None],(B,E))
        np.add.at(self.count,(bi[self.visit],self.cat[self.visit]),1)
        np.minimum.at(self.first,(bi[self.visit],self.cat[self.visit]),events[:,:,1][self.visit].astype(np.float64))
        self.present=self.count>0

def indexed_scores(state,compiled,observation_indices,goal_indices):
    oi=np.asarray(observation_indices,np.int64);gi=np.asarray(goal_indices,np.int64)
    scope=compiled.scope[gi];selected=scope[np.arange(len(oi))[:,None],state.cat[oi]]&state.visit[oi]
    count=selected.sum(1);components=np.zeros((len(oi),4),np.float64)
    components[:,0]=np.minimum(count/compiled.count[gi],1)
    components[:,1]=(state.present[oi]&compiled.required[gi]).sum(1)/np.maximum(compiled.required_count[gi],1)
    first=state.first[oi];a=np.take_along_axis(first,compiled.left[gi],1);b=np.take_along_axis(first,compiled.right[gi],1)
    components[:,2]=((np.isfinite(a)&np.isfinite(b)&(a<b))&compiled.pair_valid[gi]).sum(1)/np.maximum(compiled.order_count[gi],1)
    budget=compiled.budget[gi]
    excess=(np.maximum(state.cost[oi]-budget[:,None],0)*selected).sum(1)
    components[:,3]=np.where(count>0,np.clip(1-excess/(np.maximum(count,1)*budget),0,1),0)
    components*=compiled.active[gi]
    scores=1-components.sum(1)/compiled.active_count[gi]
    return scores,components

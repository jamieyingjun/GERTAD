"""Goal-independent physical observation preparation."""
import math
import numpy as np
from . import features
from . import geometry as c
from .config import CATEGORIES, CI, E_MAX, PAD
EARTH = 6371008.8

def _geometry(coords,times,entities,vectorized):
    elapsed=times-times[0]
    if not vectorized:
        return [(c.event_intervals(coords,elapsed,e,dwell=0.),c.insertion_cost(coords,e)) for e in entities]
    centers=np.asarray([[e['latitude'],e['longitude']] for e in entities],dtype=np.float64)
    if not len(centers):return []
    scale=np.column_stack([np.full(len(centers),math.pi*EARTH/180),math.pi*EARTH/180*np.cos(np.radians(centers[:,0]))])
    xy=(coords[None]-centers[:,None])*scale[:,None]
    a=xy[:,:-1];v=np.diff(xy,axis=1)
    aa=v[...,0]*v[...,0]+v[...,1]*v[...,1]
    cc=a[...,0]*a[...,0]+a[...,1]*a[...,1]-150.**2
    bb=2*(a[...,0]*v[...,0]+a[...,1]*v[...,1]);disc=bb*bb-4*aa*cc
    stationary=aa<1e-20
    root=np.sqrt(np.maximum(0,disc));den=np.where(stationary,1.,2*aa)
    lo=np.maximum(0,(-bb-root)/den);hi=np.minimum(1,(-bb+root)/den)
    lo=np.where(stationary,0.,lo);hi=np.where(stationary,1.,hi)
    valid=np.where(stationary,cc<=0,(disc>=0)&(hi>lo))
    starts=elapsed[:-1]+lo*np.diff(times);ends=elapsed[:-1]+hi*np.diff(times)
    rad=np.radians(coords);cr=np.radians(centers)
    def dist(a,b):
        d=b-a
        h=np.sin(d[...,0]/2)**2+np.cos(a[...,0])*np.cos(b[...,0])*np.sin(d[...,1]/2)**2
        return 2*EARTH*np.arcsin(np.sqrt(np.clip(h,0,1)))
    # Respect original argument directions for endpoint spherical arithmetic.
    left=dist(rad[None,:-1],cr[:,None]);right=dist(cr[:,None],rad[None,1:])
    segments=dist(rad[:-1],rad[1:]);cost=np.maximum(0,(left+right-segments).min(1))
    result=[]
    for j in range(len(entities)):
        merged=[]
        for k in np.flatnonzero(valid[j]):
            start=float(starts[j,k]);end=float(ends[j,k])
            if merged and start<=merged[-1][1]+1e-7:merged[-1][1]=max(merged[-1][1],end)
            else:merged.append([start,end])
        result.append(([(a,b) for a,b in merged if b-a>=-1e-7],float(cost[j])))
    return result

def prepare_observation(raw,optimized=True):
    coords=np.asarray(raw['coordinates'],np.float64);times=np.asarray(raw['timestamps'],np.float64)
    if len(coords)<2 or len(coords)>PAD or not np.all(np.diff(times)>0):raise ValueError('invalid physical observation')
    entities=sorted(raw['observed_map'],key=lambda e:e['entity_id']);T=len(times);elapsed=times-times[0];duration=elapsed[-1]
    if len(entities)>E_MAX:raise ValueError('candidate bound exceeded; no pruning is permitted')
    scale=np.array([111320.,111320.*np.cos(np.radians(coords[:,0].mean()))]);xy=(coords-coords[0])*scale
    dt=np.r_[0,np.diff(times)]
    ef=np.zeros((E_MAX,11+len(CATEGORIES)),np.float32);base=np.zeros((E_MAX,3),np.float32)
    valid=np.zeros(E_MAX,bool);cats=np.full(E_MAX,-1,np.int64);xx=np.zeros((E_MAX,PAD,18),np.float16)
    anchors=np.asarray([[e['latitude'],e['longitude']] for e in entities],np.float64).reshape(-1,2)
    relative=(coords[None]-anchors[:,None])*scale if entities else np.empty((0,T,2))
    distances=np.linalg.norm(relative,axis=-1)
    geom=_geometry(coords,times,entities,optimized)
    # Immutable observation dynamics are computed once, while entity-relative fields remain per candidate.
    if optimized and entities:
        shared=features.make_features(xy,times,np.zeros(2))
        allx=np.broadcast_to(shared,(len(entities),T,18)).copy()
        feature_relative=xy[None]-(anchors-coords[0])[:,None]*scale
        feature_distance=np.linalg.norm(feature_relative,axis=-1)
        allx[:,:,:2]=feature_relative/150;allx[:,:,2]=feature_distance/150
        allx[:,:,3]=np.column_stack([np.zeros(len(entities)),np.diff(feature_distance,axis=1)])/150
        allx[:,:,9]=1/(1+np.exp(np.clip((feature_distance-150)/30,-40,40)))
        # Float32 before float16 mirrors make_features and frozen storage.
        xx[:len(entities),:T]=np.clip(allx,-100,100).astype(np.float32).astype(np.float16)
    for j,(e,(intervals,cost)) in enumerate(zip(entities,geom)):
        longest=max([b-a for a,b in intervals],default=0.)
        qualified=[a for a,b in intervals if b-a>=60-1e-7]
        first=min(qualified) if qualified else min([a for a,b in intervals],default=duration)
        d=distances[j];center=(anchors[j]-coords[0])*scale;cat=CI[e['category']]
        ef[j,:11]=[center[0]/150,center[1]/150,min(d.min()/150,10),min(longest/60,10),first/duration,min(cost/300,10),min(sum(b-a for a,b in intervals)/60,20),float(np.mean(d<=150)),T/50,min(dt.max()/60,20),float(bool(qualified))]
        ef[j,11+cat]=1;cats[j]=cat;valid[j]=True;base[j]=[np.clip((longest-60)/10,-8,8),first/duration,cost/300]
        if not optimized:xx[j,:T]=features.make_features(xy,times,(anchors[j]-coords[0])*scale)
    timepad=np.zeros(PAD,np.float32);timepad[:T]=elapsed/duration
    return dict(x=xx.astype(np.float32),times=timepad,entity=ef,base=base,entity_valid=valid,entity_category=cats,lengths=np.int64(T),duration=np.float32(duration))

def prepare_observations(raws,optimized=True):
    records=[prepare_observation(r,optimized) for r in raws]
    return {k:np.stack([r[k] for r in records]) for k in records[0]}


def cache_key(raw, checkpoint_hash, method_version):
    import hashlib
    import json
    payload = {key: raw[key] for key in ('coordinates', 'timestamps', 'observed_map', 'context_id', 'condition')}
    payload.update(checkpoint=checkpoint_hash, implementation=method_version)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True,
                                    default=lambda value: value.tolist()).encode()).hexdigest()

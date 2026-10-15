"""Observable 18-field physical features."""
import numpy as np

def make_features(xy_m,timestamps_s,anchor_xy_m):
    xy=np.asarray(xy_m,float);times=np.asarray(timestamps_s,float);elapsed=times-times[0]
    rel=xy-np.asarray(anchor_xy_m);d=np.linalg.norm(rel,axis=1)
    delta=np.vstack([np.zeros(2),np.diff(xy,axis=0)]);dt=np.r_[0,np.diff(times)]
    step=np.linalg.norm(delta,axis=1);speed=step/np.maximum(dt,.05)
    heading=delta/np.maximum(step[:,None],1e-6);angle=np.arctan2(heading[:,1],heading[:,0]);change=np.r_[0,np.abs(np.angle(np.exp(1j*np.diff(angle))))/np.pi]
    positive_dt=dt[dt>.05];median=max(float(np.median(positive_dt)) if len(positive_dt) else 1.,.05)
    duration=max(elapsed[-1],1e-6)
    result=np.c_[rel/150,d/150,np.r_[0,np.diff(d)]/150,np.log1p(dt)/5,np.clip(speed,0,60)/20,
        heading,change,1/(1+np.exp(np.clip((d-150)/30,-40,40))),elapsed/duration,
        np.minimum(dt/median,10)/10,(dt/median>1.5).astype(float),(speed>.5).astype(float),
        np.ones(len(d)),np.zeros(len(d)),delta/150]
    assert result.shape==(len(times),18)
    return np.clip(result,-100,100).astype('float32')

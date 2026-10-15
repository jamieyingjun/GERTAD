"""Physical circle intersections and insertion-extra distance."""
import math
import numpy as np
EARTH = 6371008.8
RADIUS = 150.0
DWELL = 60.0

def spherical_distance(a, b):
    a,b=np.radians(np.asarray(a,dtype=float)),np.radians(np.asarray(b,dtype=float))
    d=b-a
    h=np.sin(d[...,0]/2)**2+np.cos(a[...,0])*np.cos(b[...,0])*np.sin(d[...,1]/2)**2
    return 2*EARTH*np.arcsin(np.sqrt(np.clip(h,0,1)))

def insertion_cost(coords, entity):
    a=np.asarray(coords[:-1]);b=np.asarray(coords[1:]);p=np.array([entity['latitude'],entity['longitude']])
    extra=spherical_distance(a,p)+spherical_distance(p,b)-spherical_distance(a,b)
    return float(max(0,np.min(extra)))

def circle_interval(a, b, radius=RADIUS):
    """Closed segment fraction interval inside circle, including stationary dwell."""
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float);v=b-a
    aa=float(v@v);cc=float(a@a-radius**2)
    if aa<1e-20:return (0.,1.) if cc<=0 else None
    bb=float(2*a@v);disc=bb*bb-4*aa*cc
    if disc<0:return None
    root=math.sqrt(max(0,disc));lo=max(0.,(-bb-root)/(2*aa));hi=min(1.,(-bb+root)/(2*aa))
    return (lo,hi) if hi>lo else None

def event_intervals(coords, times, entity, radius=RADIUS, dwell=DWELL):
    coords=np.asarray(coords,dtype=float);times=np.asarray(times,dtype=float)
    if len(coords)<2 or len(times)!=len(coords) or not np.all(np.diff(times)>0):raise ValueError('Invalid world controls')
    center=np.array([entity['latitude'],entity['longitude']])
    scale=np.array([math.pi*EARTH/180,math.pi*EARTH/180*math.cos(math.radians(center[0]))])
    xy=(coords-center)*scale
    merged=[]
    for i in range(len(xy)-1):
        interval=circle_interval(xy[i],xy[i+1],radius)
        if interval is None:continue
        lo,hi=interval;start=float(times[i]+lo*(times[i+1]-times[i]));end=float(times[i]+hi*(times[i+1]-times[i]))
        if merged and start<=merged[-1][1]+1e-7:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([start,end])
    return [(start,end) for start,end in merged if end-start>=dwell-1e-7]

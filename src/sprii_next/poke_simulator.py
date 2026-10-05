"""Fixed-action PokeWorld replay; no resampling of initial state or actions."""
import numpy as np


def replay(initial,actions,theta,cfg):
    """Return state at every step using the native Hertz-contact equations."""
    x=np.asarray(initial,np.float64).copy();a=np.asarray(actions,np.float64)
    theta=np.asarray(theta,np.float64)
    if x.shape!=(len(a),8) or a.shape[2]!=2 or theta.shape!=(len(a),3):raise ValueError('replay dimensions differ')
    if np.any(theta<=0):raise ValueError('positive physical parameters required')
    m,g,k=theta.T;fp=x[:,:2].copy();fv=x[:,2:4].copy();op=x[:,4:6].copy();ov=x[:,6:8].copy()
    dt=cfg['dt']/cfg['substeps'];fm=cfg['finger_mass'];out=[];contacts=[]
    def reflect(p,v,limit):
        for axis in (0,1):
            high=p[:,axis]>limit;low=p[:,axis]<-limit
            p[high,axis]=limit;p[low,axis]=-limit
            v[high&(v[:,axis]>0),axis]*=-cfg['wall_restitution']
            v[low&(v[:,axis]<0),axis]*=-cfg['wall_restitution']
    for action in a.transpose(1,0,2):
        touched=np.zeros(len(a),bool)
        for _ in range(cfg['substeps']):
            sep=op-fp;distance=np.linalg.norm(sep,axis=1).clip(1e-8);normal=sep/distance[:,None]
            overlap=cfg['finger_radius']+cfg['object_radius']-distance
            local=1.5*k*np.sqrt(np.maximum(overlap,0.))
            effective=m/(m+fm)
            damping=2*cfg['damping_ratio']*np.sqrt(local*effective)
            magnitude=np.maximum(0.,k*np.maximum(overlap,0.)**1.5-damping*np.sum((ov-fv)*normal,axis=1))
            magnitude*=overlap>0;touched|=overlap>0
            force=magnitude[:,None]*normal
            fv+=(cfg['force_max']*action-force)/fm*dt
            ov+=(force-(g*m)[:,None]*ov)/m[:,None]*dt
            fp+=fv*dt;op+=ov*dt
            reflect(fp,fv,cfg['arena_half_extent']-cfg['finger_radius'])
            reflect(op,ov,cfg['arena_half_extent']-cfg['object_radius'])
        out.append(np.concatenate((fp,fv,op,ov),axis=1));contacts.append(touched)
    return np.stack(out,axis=1),np.stack(contacts,axis=1)


def sensitivities(initial,actions,theta,cfg,target_scale,parameter_scale,*,fraction=.01):
    indices=np.array([0,1,3,7,15]);values=[];agreement=[]
    for j in range(3):
        delta=np.asarray(theta[:,j])*fraction
        derivatives=[]
        for multiplier in (1.,.5):
            plus=theta.copy();minus=theta.copy()
            plus[:,j]+=delta*multiplier;minus[:,j]-=delta*multiplier
            yp,_=replay(initial,actions,plus,cfg);ym,_=replay(initial,actions,minus,cfg)
            derivative=(yp[:,indices]-ym[:,indices])/(2*delta[:,None,None]*multiplier)
            derivatives.append(derivative/target_scale[None,:,:])
        scaled=np.linalg.norm(derivatives[0],axis=2)*parameter_scale[j]
        discrepancy=np.linalg.norm(derivatives[0]-derivatives[1],axis=2)/(np.linalg.norm(derivatives[1],axis=2)+1e-8)
        values.append(scaled);agreement.append(discrepancy)
    return np.stack(values,axis=2),np.stack(agreement,axis=2)

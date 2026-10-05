"""Shared validation of the declared public image/action/time support."""
import numpy as np


def image_history(experience,config):
    experience.validate()
    obs=np.asarray(experience.observations)
    actions=np.asarray(experience.actions,float);mask=np.asarray(experience.masks,bool)
    if obs.ndim!=4 or len(obs)<1 or obs.shape[1:]!=(2,config.resolution,config.resolution):raise ValueError('history image shape')
    if not np.isfinite(obs).all() or np.any(obs[:,0]<0) or np.any(obs[:,0]>1):raise ValueError('invalid history image values')
    if actions.shape!=(len(obs),2) or mask.shape!=(len(obs),) or mask[0] or not mask[1:].all():raise ValueError('history action support')
    if not np.isfinite(actions).all() or np.any(np.linalg.norm(actions,axis=1)>1+1e-7):raise ValueError('invalid past actions')
    times=np.asarray(experience.timestamps)
    if times.shape!=(len(obs),) or not np.allclose(times,np.arange(len(obs))*config.control_dt,atol=1e-8,rtol=0):raise ValueError('history time support')
    if np.any(obs[0,1]) or np.any(actions[0]):raise ValueError('hidden predecessor or action')
    images=np.rint(obs[:,0]*255).astype(np.uint8)
    if not np.allclose(obs[:,0],images/255,atol=1e-7,rtol=0):raise ValueError('expected normalized uint8 frame content')
    expected=np.zeros_like(obs[:,1]);expected[1:]=(images[1:].astype(float)-images[:-1].astype(float))/255
    if not np.allclose(obs[:,1],expected,atol=1e-7,rtol=0):raise ValueError('inconsistent within-support image difference')
    return images,actions[1:]



def public_query_images(query,config,*,allowed_targets=('joint_state_delta_8d','cold_rest_joint_state_delta_8d')):
    query.validate();payload=query.observations
    allowed={'observations','relative_times','horizon_seconds','target_spec'}
    if not isinstance(payload,dict) or set(payload)-allowed:raise ValueError('query provides fields outside no-past-actions profile')
    if payload.get('target_spec') not in allowed_targets:raise ValueError('unregistered target specification')
    obs=np.asarray(payload['observations'],np.float32)
    if obs.ndim!=4 or len(obs)<1 or obs.shape[1:]!=(2,config.resolution,config.resolution) or not np.isfinite(obs).all():raise ValueError('invalid query images')
    if np.any(obs[:,0]<0) or np.any(obs[:,0]>1) or np.any(obs[0,1]):raise ValueError('invalid image values or hidden predecessor')
    frames=np.rint(obs[:,0]*255).astype(np.uint8)
    if not np.allclose(obs[:,0],frames.astype(np.float32)/255,atol=1e-7,rtol=0):raise ValueError('image content must be normalized uint8')
    diff=np.zeros_like(obs[:,1]);diff[1:]=obs[1:,0]-obs[:-1,0]
    if not np.allclose(obs[:,1],diff,atol=1e-7,rtol=0):raise ValueError('query difference crosses declared support')
    times=np.asarray(payload['relative_times'])
    if times.shape!=(len(obs),) or not np.allclose(times,np.arange(len(obs))*config.control_dt,atol=1e-8,rtol=0):raise ValueError('query time support')
    actions=np.asarray(query.actions,np.float32)
    if actions.ndim!=2 or actions.shape[1]!=2 or len(actions)<1 or not np.isfinite(actions).all() or np.any(np.linalg.norm(actions,axis=1)>1+1e-7):raise ValueError('invalid future actions')
    if not np.isclose(payload['horizon_seconds'],len(actions)*config.control_dt,atol=1e-8,rtol=0):raise ValueError('future duration mismatch')
    if not np.array_equal(np.asarray(query.horizons),[len(actions)]):raise ValueError('future horizon mismatch')
    return obs.copy(),actions.copy()


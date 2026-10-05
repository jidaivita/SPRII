"""Declared full-history visual certificate, separate from compressed learners.

Only pixel/action arrays reach the unknown-parameter estimator. Multiple
donors use a joint trajectory fit with separate initial states. Conditional
query inference reuses the calibrated passive-prefix approximation. This is
neither a global identifiability theorem nor an exact pixel Bayes posterior.
"""
import hashlib
import numpy as np
from .schema import Config

CONFIGURATION=dict(method='full_visual_evidence_certificate_v1',starts=3,single_max_nfev=100,joint_max_nfev=150,
    parameter_bounds=[[.5,.25,4.],[2.,1.5,25.]],noise_floor_m=.0006,posterior_samples=256,posterior_burn=96,
    seed=71973,passive_grid=[192,3072],passive_refinement=[192,1024],interval_quantiles=[.1,.9],
    tracker='msaa_coverage_centroid_v1',prior='uniform_in_physical_parameters',
    constant_unforced_history='zero likelihood factor by estimator convention; identical pixels alone are not a physics equivalence proof',
    failure_policy='retain support/tracking/fit failures; fit failure uses declared prior fallback; no successful-case-only aggregate')
PARAMETERS=('m','gamma','k','k_over_m')


def validate_public_inputs(histories,query,resolution):
    if resolution not in (64,128):raise ValueError('unregistered visual resolution')
    if not isinstance(histories,list) or len(histories)>2:raise ValueError('at most two registered raw histories')
    def images(value):
        a=np.asarray(value)
        if a.dtype!=np.uint8 or a.ndim!=3 or len(a)<1 or a.shape[1:]!=(resolution,resolution):raise ValueError('expected original uint8 images')
        return a
    def actions(value,count):
        a=np.asarray(value)
        if a.shape!=(count,2) or not np.isfinite(a).all() or np.any(np.linalg.norm(a,axis=1)>1+1e-7):raise ValueError('action support differs from observed transitions')
    for h in histories:
        if set(h)!={'images','actions'}:raise PermissionError('raw estimator accepts only public history pixels/actions')
        a=images(h['images']);actions(h['actions'],len(a)-1)
    if query is not None:
        if set(query)!={'images','kind','budget','horizon','future_actions'}:raise PermissionError('private or unregistered query fields')
        a=images(query['images']);kind=query['kind'];budget=query['budget']
        if kind not in ('cold','moving') or budget not in ((0,1) if kind=='cold' else (0,1,3,7,15,31,63,95)):raise ValueError('unregistered query support')
        if len(a)!=budget+1 or query['horizon'] not in (1,4,16,32):raise ValueError('query observation or forecast budget differs')
        if kind=='cold' and not np.array_equal(a,np.broadcast_to(a[:1],a.shape)):raise ValueError('cold query is not observed rest')
        if query['future_actions'] is not None:actions(query['future_actions'],query['horizon'])


def array_digest(*values):
    result=hashlib.sha256()
    for value in values:
        a=np.ascontiguousarray(value);result.update(str((a.shape,str(a.dtype))).encode());result.update(a.tobytes())
    return result.hexdigest()


def parameter_summary(samples):
    a=np.asarray(samples,float)
    if a.ndim!=2 or a.shape[1]!=3 or len(a)<1 or not np.isfinite(a).all() or np.any(a<=0):raise ValueError('invalid physical parameter samples')
    physical=np.c_[a,a[:,2]/a[:,0]]
    return dict(mean=physical.mean(0).tolist(),log_mean=np.log(physical).mean(0).tolist(),
        interval80=np.quantile(physical,[.1,.9],axis=0).tolist(),samples=len(a),parameters=list(PARAMETERS),
        interpretation='approximate posterior under the declared bounded continuous prior, not a point-identification certificate')


def public_query_batch(query):
    from .schema import Episode,query_packet
    from .adapters import z_query
    # Pad only unobserved future images to use the existing codec; the codec
    # exposes the actual observed prefix and supplied actions, never this pad.
    images=query['images'];actions=query['future_actions']
    if actions is None:return None
    n=len(images);h=len(actions)
    padded=np.concatenate((images,np.repeat(images[-1:],h,axis=0)))
    ep=Episode(padded,np.concatenate((np.zeros((n-1,2),np.float32),actions)),np.arange(n+h)*.05,np.zeros((n+h,8)),{})
    packet=query_packet(ep,n-1,n-1,h);packet['target_spec']='cold_rest_joint_state_delta_8d' if query['kind']=='cold' else 'passive_prefix95_joint_state_delta_8d'
    return z_query(packet)


def estimate(histories,query,*,resolution,configuration):
    """No bank, identity, true parameter or state argument exists on this path."""
    validate_public_inputs(histories,query,resolution)
    if configuration!=CONFIGURATION:raise ValueError('raw certificate inference recipe differs')
    from .calibration import track
    from .joint_identification import fit_histories,information_factor
    from .persistent_reference import posterior_samples
    cfg=Config(resolution=resolution);tracked=[]
    for h in histories:
        try:positions=track(h['images'],cfg)
        except (ValueError,FloatingPointError) as exc:return dict(status='TRACKING_FAILED',error=str(exc),parameter_estimate=None,prediction=None,tracked=tracked)
        tracked.append(dict(positions=positions.tolist(),fingerprint=array_digest(h['images'],h['actions'])))
    unique={x['fingerprint']:i for i,x in enumerate(tracked)}
    informative=[dict(positions=np.asarray(value['positions']),actions=histories[i]['actions'],fingerprint=value['fingerprint']) for i,value in enumerate(tracked)
        if len(histories[i]['images'])>=3 and (np.any(histories[i]['actions']) or not np.array_equal(histories[i]['images'],np.broadcast_to(histories[i]['images'][:1],histories[i]['images'].shape)))]
    fit=None;fit_error=None;H=np.zeros((3,3));b=np.zeros(3)
    if informative:
        try:
            fit=fit_histories(informative,cfg,starts=configuration['starts'],max_nfev=configuration['joint_max_nfev'],parameter_bounds=configuration['parameter_bounds'])
            H,b=information_factor(fit,cfg,configuration['noise_floor_m'])
            if not fit['optimizer_success']:fit_error='optimizer budget did not converge; declared prior fallback'
        except (ValueError,FloatingPointError) as exc:fit_error=str(exc)
    samples=posterior_samples(H,b,samples=configuration['posterior_samples'],burn=configuration['posterior_burn'],seed=configuration['seed'])
    diagnostic={};prediction=None;query_positions=None
    if query is not None:
        try:query_positions=track(query['images'],cfg)
        except (ValueError,FloatingPointError) as exc:return dict(status='TRACKING_FAILED',error=str(exc),parameter_estimate=None,prediction=None,tracked=tracked,fit=fit)
        if query['kind']=='moving':
            from .passive_query_reference import conditional_parameters,predict_passive
            times=(95-query['budget']+np.arange(len(query_positions)))*.05
            if query['future_actions'] is not None:
                prediction,diagnostic=predict_passive(public_query_batch(query),H,b,cfg,shape=tuple(configuration['passive_grid']),samples=configuration['posterior_samples'],seed=configuration['seed'],return_parameter_samples=True)
                samples=np.asarray(diagnostic.pop('parameter_samples'))
            else:
                samples,_,_,_,diagnostic=conditional_parameters(query_positions,times,H,b,shape=tuple(configuration['passive_grid']),samples=configuration['posterior_samples'],noise_m=configuration['noise_floor_m'],seed=configuration['seed'])
        elif query['future_actions'] is not None:
            from .cold_opportunity import visual_rest_state
            from .batch_prediction import reference_rollout_batch
            initial=visual_rest_state(query_positions[-1],cfg)
            path=reference_rollout_batch(samples,np.broadcast_to(initial,(len(samples),8)),query['future_actions'],cfg)
            prediction=np.mean(path[:,-1]-initial,axis=0)
    return dict(status='FIT_FAILED_PRIOR_FALLBACK' if fit_error else 'EXECUTED',fit=fit,fit_error=fit_error,
        parameter_estimate=parameter_summary(samples),prediction=None if prediction is None else np.asarray(prediction).tolist(),
        tracked=tracked,query_positions=None if query_positions is None else query_positions.tolist(),conditional_diagnostics=diagnostic,
        history_information_H=H.tolist(),history_information_b=b.tolist(),
        processed_histories=len(histories),unique_histories=len(unique),processed_frames=sum(len(h['images']) for h in histories),
        unique_frames=sum(len(histories[i]['images']) for i in unique.values()),
        zero_force_scale_ambiguity=not any(np.any(h['actions']) for h in histories),
        no_observed_history_excitation=not informative,raw_history_retention='full history permitted for this calibration reference; not a compressed-lifecycle claim')


def privileged_references(query,theta,initial_state,*,resolution):
    """Separate evaluator-only references, called after unknown predictions seal."""
    validate_public_inputs([],query,resolution)
    if query['future_actions'] is None:return dict(status='FUTURE_SUPPORT_UNAVAILABLE')
    from .identification import reference_rollout
    from .calibration import track
    from .cold_opportunity import visual_rest_state
    cfg=Config(resolution=resolution);theta=np.asarray(theta,float);initial_state=np.asarray(initial_state,float)
    state_prediction=reference_rollout(theta,initial_state,query['future_actions'],cfg)[-1]-initial_state
    visual_error=None;visual_prediction=None
    try:
        if query['kind']=='cold':
            initial=visual_rest_state(track(query['images'],cfg)[-1],cfg)
            visual_prediction=reference_rollout(theta,initial,query['future_actions'],cfg)[-1]-initial
        else:
            from .passive_query_reference import predict_passive
            visual_prediction,_=predict_passive(public_query_batch(query),np.zeros((3,3)),np.zeros(3),cfg,theta=theta)
    except (ValueError,FloatingPointError) as exc:visual_error=str(exc)
    return dict(status='EXECUTED',visual_theta=None if visual_prediction is None else np.asarray(visual_prediction).tolist(),
        visual_theta_error=visual_error,state_theta=state_prediction.tolist(),
        interpretation='visual_theta retains visual state uncertainty; state_theta uses evaluator truth and independent DOP853; neither is the unknown-parameter method')

"""Pure observed-trajectory scoring with explicit time-integration semantics."""
import numpy as np

CENTER_INTEGRAL_RULE='trapezoidal_over_observed_horizon'


def control_metrics(states,actions,goal,config):
    states=np.asarray(states);actions=np.asarray(actions)
    center=(states[:,:2]+states[:,2:4])/2
    center_velocity=(states[:,4:6]+states[:,6:8])/2
    relative_velocity=states[:,6:8]-states[:,4:6]
    length=np.linalg.norm(states[:,2:4]-states[:,:2],axis=1)
    errors=dict(center_error_m=np.linalg.norm(center-goal,axis=1),
        center_speed_m_s=np.linalg.norm(center_velocity,axis=1),
        relative_speed_m_s=np.linalg.norm(relative_velocity,axis=1),length_error_m=np.abs(length-config.ell0))
    tolerances=dict(center_error_m=.02,center_speed_m_s=.02,relative_speed_m_s=.03,length_error_m=.02)
    rows=[]
    for seconds in (8.,12.,16.):
        end=round(seconds/config.control_dt)
        if end>=len(states):
            rows.append(dict(horizon_s=seconds,status='INCOMPLETE',success=False));continue
        intervals=round(.5/config.control_dt)
        if abs(intervals*config.control_dt-.5)>1e-12:raise ValueError('stability window must align with observation intervals')
        observed={k:float(v[max(0,end-intervals):end+1].max()) for k,v in errors.items()}
        squared=errors['center_error_m'][:end+1]**2
        integral=float((.5*squared[0]+squared[1:-1].sum()+.5*squared[-1])*config.control_dt)
        rows.append(dict(horizon_s=seconds,status='EXECUTED',success=all(observed[k]<=t for k,t in tolerances.items()),
            sustained_maxima=observed,tolerances=tolerances,sustained_observation_span_s=.5,sustained_frames=intervals+1,
            integrated_center_error_m2_s=integral,center_error_integral_rule=CENTER_INTEGRAL_RULE,
            effort_n2_s=float(np.sum(actions[:end]**2)*config.force_max**2*config.control_dt)))
    return rows

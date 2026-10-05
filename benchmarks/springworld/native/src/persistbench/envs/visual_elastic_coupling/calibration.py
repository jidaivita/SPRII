"""Pixel-only integrated identification diagnostics; no true state enters fitting."""
import numpy as np
from .schema import Config

def track_interior(images,config=Config()):
    from .rendering import pixel_to_world
    coordinates=[]
    for frame in images:
        objects=[]
        for gray in (96,224):
            # Known parameter-independent role appearance is a public model prior.
            y,x=np.nonzero(frame==gray)
            if len(x)<2:
                raise ValueError("role tracking failed: insufficient visible interior pixels")
            objects.extend(pixel_to_world(float(x.mean()),float(y.mean()),config))
        coordinates.append(objects)
    return np.asarray(coordinates,dtype=np.float64)


def track(images, config=Config()):
    """Coverage-weighted role centroids from the public uint8 image alone.

    The fixed renderer uses four MSAA samples. Decode mixtures of the known
    role intensity, neutral connector and background in a local disk region.
    This retains partial boundary pixels; the old interior-only estimator is
    retained as ``track_interior`` for paired measurement diagnostics. No
    segmentation buffer, simulator state, parameter or episode ID is read.
    """
    from .rendering import pixel_to_world
    frames = np.asarray(images)
    if frames.ndim != 3 or frames.shape[1:] != (config.resolution, config.resolution):
        raise ValueError("tracker image/config shape mismatch")
    yy, xx = np.indices(frames.shape[1:])
    radius = config.radius / config.field_width * config.resolution
    coordinates = []
    for frame in frames:
        centers = []
        for gray in (96, 224):
            y, x = np.nonzero(frame == gray)
            if len(x) < 2:
                raise ValueError("role tracking failed: insufficient interior pixels")
            centers.append(np.array([x.mean(), y.mean()]))
        distances = [np.hypot(xx-c[0], yy-c[1]) for c in centers]
        pair = []
        for role, gray in enumerate((96, 224)):
            weights = np.zeros_like(frame, dtype=float)
            # Different roles can share gray levels; locality resolves identity.
            region = (distances[role] < radius+1.0) & (distances[role] < distances[1-role])
            for disk_samples in range(1, 5):
                for line_samples in range(5-disk_samples):
                    level = round((gray*disk_samples+40*line_samples)/4)
                    weights[region & (frame == level)] = disk_samples/4
            total = weights.sum()
            if total < 1:
                raise ValueError("role coverage tracking failed")
            pair.extend(pixel_to_world((weights*xx).sum()/total,
                                       (weights*yy).sum()/total, config))
        coordinates.append(pair)
    return np.asarray(coordinates, dtype=np.float64)

def _integral(a,dt):
    return np.concatenate([np.zeros_like(a[:1]),np.cumsum((a[1:]+a[:-1])*.5*dt,axis=0)])

def identify(images,actions,config=Config(),tracker=track_interior):
    return identify_positions(tracker(images,config),actions,config)


def identify_positions(positions,actions,config=Config()):
    """Fit integrated equations to measured positions; diagnose rank explicitly.

    Unknowns beta=(1/m,gamma,k/m) plus four initial velocities. Integrating
    damping by parts avoids noisy finite-difference acceleration estimates.
    Force integration is exact for the recorded zero-order hold actions.
    Spring integration is trapezoidal on observed positions, an approximation.
    """
    x=np.asarray(positions,float); t=np.arange(len(x))*config.control_dt
    if np.asarray(actions).shape != (len(x)-1,2):
        raise ValueError("ID requires exactly the observed transition actions")
    r=x[:,2:4]-x[:,:2]; lengths=np.linalg.norm(r,axis=1)
    if np.any(lengths<=0): raise ValueError("tracking degeneracy")
    spring=(lengths-config.ell0)[:,None]*r/lengths[:,None]
    spring=np.concatenate([spring,-spring],axis=1)
    spring_int=_integral(_integral(spring,config.control_dt),config.control_dt)
    drag_int=-_integral(x-x[0],config.control_dt)
    force_int=np.zeros_like(x)
    impulse=np.zeros(2); displacement=np.zeros(2)
    for i,u in enumerate(np.asarray(actions)*config.force_max):
        displacement+=impulse*config.control_dt+.5*u*config.control_dt**2
        impulse+=u*config.control_dt
        force_int[i+1,:2]=displacement
    design=np.zeros((len(x)-1,4,7))
    design[:,:,0]=force_int[1:]; design[:,:,1]=drag_int[1:]; design[:,:,2]=spring_int[1:]
    for d in range(4): design[:,d,3+d]=t[1:]
    matrix=design.reshape(-1,7); y=(x[1:]-x[0]).reshape(-1)
    solution,_,rank,sv=np.linalg.lstsq(matrix,y,rcond=1e-8)
    beta=solution[:3]
    return dict(beta=beta.tolist(),rank=int(rank),singular_values=sv.tolist(),
        full_rank=bool(rank==7),condition_number=float(sv[0]/sv[-1]) if sv[-1]>1e-14 else None,
        integrated_residual_rmse=float(np.sqrt(np.mean((np.sum(matrix*solution[None,:],axis=1)-y)**2))),
        m=float(1/beta[0]) if rank==7 and beta[0]>0 else None,
        gamma=float(beta[1]),k_over_m=float(beta[2]),
        k=float(beta[2]/beta[0]) if rank==7 and beta[0]>0 else None,
        initial_velocity=solution[3:].tolist(),
        status="diagnostic_only",uses_private_state=False,
        approximation="trapezoidal spring integration from pixel centroids; unconstrained least squares")

def component_errors(prediction,target):
    error=np.asarray(prediction)-np.asarray(target)
    pos=error[...,:4]; vel=error[...,4:]
    return dict(position_mse_m2=float(np.mean(pos**2)),velocity_mse_m2_s2=float(np.mean(vel**2)),
        center_position_mse_m2=float(np.mean(((pos[...,:2]+pos[...,2:])/2)**2)),
        relative_position_mse_m2=float(np.mean((pos[...,2:]-pos[...,:2])**2)),
        center_velocity_mse_m2_s2=float(np.mean(((vel[...,:2]+vel[...,2:])/2)**2)),
        relative_velocity_mse_m2_s2=float(np.mean((vel[...,2:]-vel[...,:2])**2)))

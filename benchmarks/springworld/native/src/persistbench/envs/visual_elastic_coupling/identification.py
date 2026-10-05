"""Constrained trajectory fitting from visible histories, with rank diagnostics.

The forward equations are a declared physics prior, independent of MuJoCo's
episode generator. Initial position and velocity are nuisance variables fitted
from the same history. No finite-difference acceleration or true state is used.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .schema import Config
from .optimization import select_solution
from .calibration import track,identify_positions


def reference_rollout(parameters, initial_state, actions, config=Config(), *, rtol=2e-8):
    """DOP853 with exact action-change boundaries, for the explicit reference.

    Candidate fits may leave the camera view; they are not new data episodes.
    The spring singularity is rejected. Accepted experimental trajectories
    still come exclusively from the stage-checked MuJoCo generator.
    """
    m, gamma, k = np.asarray(parameters, float)
    if not np.isfinite([m, gamma, k]).all() or m <= 0 or gamma < 0 or k <= 0:
        raise ValueError("invalid reference parameters")
    actions = np.asarray(actions, float)
    if actions.ndim != 2 or actions.shape[1] != 2 or not np.isfinite(actions).all():
        raise ValueError("invalid actions")
    state = np.asarray(initial_state, float).copy()
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("invalid initial state")
    result = [state.copy()]
    start = 0
    while start < len(actions):
        stop = start+1
        while stop < len(actions) and np.array_equal(actions[stop], actions[start]):
            stop += 1
        force = actions[start]*config.force_max
        def rhs(t, s):
            rx, ry = s[2]-s[0], s[3]-s[1]
            length = np.hypot(rx, ry)
            if length < 1e-7:
                raise ValueError("reference spring singularity")
            factor = (k/m)*(1-config.ell0/length)
            ax, ay = factor*rx, factor*ry
            return [s[4], s[5], s[6], s[7],
                    force[0]/m-gamma*s[4]+ax, force[1]/m-gamma*s[5]+ay,
                    -gamma*s[6]-ax, -gamma*s[7]-ay]
        times = np.arange(1, stop-start+1)*config.control_dt
        solution = solve_ivp(rhs, (0., times[-1]), state, t_eval=times,
                             method="DOP853", rtol=rtol, atol=rtol*.01)
        if not solution.success or solution.y.shape[1] != len(times):
            raise ValueError("reference integration failed")
        result.extend(solution.y.T)
        state = solution.y[:, -1]
        start = stop
    return np.asarray(result)


def fit_positions(positions, actions, config=Config(), *, max_nfev=100,
                  parameter_bounds=((.5, .25, 4.), (2., 1.5, 25.)),
                  starts=3):
    """Fit measured positions; callers label the observation source explicitly.

    Bounds are a declared shared development prior, never per-system labels.
    With zero force the fitted variables are gamma and k/m. Mass and stiffness
    are returned as None because their common scale is structurally invisible.
    The projected Jacobian removes initial-state nuisance directions before
    assessing local parameter sensitivity. This is not a global certificate.
    """
    x = np.asarray(positions, float)
    actions = np.asarray(actions, float)
    if x.ndim != 2 or x.shape[1] != 4 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("expected at least three measured 4D positions")
    if actions.shape != (len(x)-1, 2):
        raise ValueError("fit requires exactly the observed transition actions")
    low, high = np.asarray(parameter_bounds, float)
    if low.shape != (3,) or np.any(low <= 0) or np.any(high <= low):
        raise ValueError("invalid shared prior")
    forced = bool(np.any(np.abs(actions) > 1e-12))
    plow = low if forced else np.array([low[1], low[2]/high[0]])
    phigh = high if forced else np.array([high[1], high[2]/low[0]])
    ntheta = len(plow)
    pixel = config.field_width/config.resolution
    lower = np.r_[np.log(plow), x[0]-2*pixel, [-2.]*4]
    upper = np.r_[np.log(phigh), x[0]+2*pixel, [2.]*4]
    initial_velocity = (x[min(3, len(x)-1)]-x[0])/(min(3, len(x)-1)*config.control_dt)
    def decode(z):
        p = np.exp(z[:ntheta])
        return p if forced else np.array([1., p[0], p[1]])
    def residual(z):
        predicted = reference_rollout(decode(z), z[ntheta:], actions, config)
        return ((predicted[:, :4]-x)/pixel).ravel()
    # A diagonal interpolation of all three physical parameters barely varies
    # k/m. Use the visible integrated-equation estimate plus starts covering the
    # full admissible ratio range, including the fast radial oscillations.
    initial_diagnostic=identify_positions(x,actions,config)
    beta=np.asarray(initial_diagnostic["beta"])
    ratio=np.clip(beta[2],low[2]/high[0],high[2]/low[0])
    mass=1/beta[0] if beta[0]>1e-10 else np.sqrt(low[0]*high[0])
    mass=np.clip(mass,max(low[0],low[2]/ratio),min(high[0],high[2]/ratio))
    warm=np.array([mass,np.clip(beta[1],low[1],high[1]),mass*ratio]) if forced else np.array([np.clip(beta[1],low[1],high[1]),ratio])
    seed_parameters=[warm]
    for fraction in np.linspace(0.,1.,starts-1) if starts>1 else []:
        if forced:
            candidate=np.array([high[0]*(low[0]/high[0])**fraction,
                np.sqrt(low[1]*high[1]),low[2]*(high[2]/low[2])**fraction])
        else:
            candidate=np.array([np.sqrt(low[1]*high[1]),plow[1]*(phigh[1]/plow[1])**fraction])
        seed_parameters.append(candidate)
    attempts = []
    solutions = []
    for index,seed in enumerate(seed_parameters):
        velocity=np.asarray(initial_diagnostic["initial_velocity"]) if index==0 else initial_velocity
        z0 = np.r_[np.log(seed),x[0],np.clip(velocity, -1.9, 1.9)]
        try:
            fit = least_squares(residual, z0, bounds=(lower, upper),
                                x_scale=np.r_[np.ones(ntheta), [pixel]*4, [.2]*4],
                                max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-7)
            attempts.append(dict(success=bool(fit.success), cost=float(fit.cost),
                                 nfev=int(fit.nfev), message=fit.message,initial_parameters=seed.tolist()))
            solutions.append(fit)
        except (ValueError, FloatingPointError) as exc:
            attempts.append(dict(success=False, error=str(exc)))
    if not solutions:
        raise ValueError("all trajectory-fit starts failed: "+str(attempts))
    best = select_solution(solutions)
    j = best.jac
    nuisance = j[:, ntheta:]
    coefficients = np.linalg.lstsq(nuisance, j[:, :ntheta], rcond=1e-9)[0]
    # Avoid platform BLAS small-matrix warnings; the explicit contraction also
    # makes the nuisance projection independent of that optimized code path.
    projected = j[:, :ntheta]-np.einsum("ij,jk->ik", nuisance, coefficients)
    sv = np.linalg.svd(projected, compute_uv=False)
    rank = int(np.count_nonzero(sv > max(1e-6, sv[0]*1e-6)))
    fitted = decode(best.x)
    identified = rank == ntheta and bool(best.success)
    return dict(m=float(fitted[0]) if forced and identified else None,
                gamma=float(fitted[1]), k=float(fitted[2]) if forced and identified else None,
                k_over_m=float(fitted[2]/fitted[0]),
                initial_state=best.x[ntheta:].tolist(),
                position_residual_rmse_m=float(np.sqrt(np.mean(best.fun**2))*pixel),
                local_parameter_rank=rank, parameter_dimension=ntheta,
                parameter_log_singular_values_per_pixel=sv.tolist(),
                parameter_log_information_per_pixel=(projected.T@projected).tolist(),
                parameter_log_mode=best.x[:ntheta].tolist(),
                parameter_log_coordinates=["m","gamma","k"] if forced else ["gamma","k_over_m"],
                structural_mass_stiffness_ambiguity=not forced,
                active_bounds=best.active_mask[:ntheta].tolist(),
                optimizer_success=bool(best.success),
                status="locally_identified" if identified else "weak_or_unidentified",
                parameter_bounds=[low.tolist(), high.tolist()], attempts=attempts,
                initialization="visible integrated equations plus full k/m range starts v2",
                prior="known dynamics, camera/role appearance, common parameter bounds; fitted initial state",
                uncertainty="local sensitivity only; held-out recovery/prediction required")


def identify_visible(images, actions, config=Config(), **kwargs):
    result = fit_positions(track(images, config), actions, config, **kwargs)
    result.update(uses_private_state=False, observation_source="uint8_grayscale",
                  tracker="msaa_coverage_centroid_v1")
    return result

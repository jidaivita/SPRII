"""Full-support development bank with preserved failures and paired cameras.

Counterfactual common seeds across theta improve coverage comparisons. Episodes
within each system use distinct initial/action streams. This development grid
is explicitly not a sealed split or the final pretraining population.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
from .schema import Config, Parameters
from .data import action_library


def worker_init():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str((multiprocessing.current_process()._identity[0]-1) % 4)


def generate_one(job):
    from .physics import trajectory, InvalidTrajectory
    from .rendering import render_states
    from .calibration import track, track_interior
    index, theta, kind, replicate, seed, transitions, output, profiles, motion = job
    root = Path(output)/"episodes"/f"{index:05d}"
    root.mkdir(parents=True)
    cfg = Config()
    stream = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(stream[0])
    center = rng.uniform(-.08, .08, 2)
    angle = rng.uniform(0, 2*np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    deformation = float(rng.choice([-.08, -.04, .04, .08])) if kind in ("free", "forced") else 0.
    length = cfg.ell0+deformation
    speed = float(rng.uniform(.1, motion["glide_speed_max"])) if kind == "glide" else 0.
    velocity = speed*np.array([np.cos(angle+.5), np.sin(angle+.5)])
    initial = np.r_[center-length*direction/2, center+length*direction/2, velocity, velocity]
    actions = action_library(np.random.default_rng(stream[1]), transitions, kind)*motion["force_gain"]
    diagnostics = {}
    began = time.monotonic()
    failure = None
    try:
        states = trajectory(Parameters(*theta), initial, actions, cfg, diagnostics=diagnostics)
    except InvalidTrajectory as exc:
        failure = dict(reason=exc.reason, time_s=exc.time)
        states = diagnostics.get("accepted_observed_states", initial[None])
    envelopes = diagnostics.get("transition_position_envelope", np.empty((0, 4)))
    np.savez_compressed(root/"private.npz", state=states, initial_state=initial,
                        transition_position_envelope=envelopes)
    np.savez_compressed(root/"actions.npz", actions=actions,
                        timestamps=np.arange(transitions+1)*cfg.control_dt)
    row = dict(index=index, theta=dict(zip(("m", "gamma", "k"), theta)), kind=kind,
               replicate=replicate, seed=seed, split="development",
               episode_key=f"vec_dev_grid_{index:05d}", requested_transitions=transitions,
               completed_transitions=len(states)-1, full_physics_valid=failure is None,
               failure=failure, deformation_m=deformation, speed_m_s=speed,
               config=asdict(cfg), motion_profile=motion, profiles=[])
    for field, resolution in profiles:
        c = replace(cfg, field_width=field, resolution=resolution)
        boundary = field/2-c.margin-c.radius
        invalid = np.flatnonzero(np.max(envelopes, axis=1)>boundary)
        frames = min(len(states), int(invalid[0])+1 if len(invalid) else len(states))
        if np.max(np.abs(initial[:4])) > boundary:
            frames = 0
        p = dict(field_width=field, resolution=resolution, valid_prefix_frames=frames,
                 full_episode_valid=frames==transitions+1,
                 window_prefix_visibility={str(n): frames>=n for n in (8, 16, 24, 48, 96, 193)})
        if frames:
            images = render_states(states[:frames], c)
            name = f"visible_f{field:g}_r{resolution}.npz"
            np.savez_compressed(root/name, images=images, actions=actions[:frames-1],
                                timestamps=np.arange(frames)*c.control_dt)
            p["asset"] = name
            try:
                for label, tracker in (("coverage", track), ("interior", track_interior)):
                    measured = tracker(images, c)
                    p[f"{label}_position_rmse_m"] = float(np.sqrt(np.mean((measured-states[:frames, :4])**2)))
                    np.savez_compressed(root/f"{label}_f{field:g}_r{resolution}.npz", positions=measured)
                p["tracking_status"] = "EXECUTED"
            except ValueError as exc:
                p["tracking_status"] = "FAILED"; p["tracking_error"] = str(exc)
        row["profiles"].append(p)
    row["elapsed_seconds"] = time.monotonic()-began
    (root/"private.json").write_text(json.dumps(row, indent=2)+"\n")
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--design", choices=("grid", "continuous"), default="grid")
    parser.add_argument("--systems", type=int, default=64)
    parser.add_argument("--glide-speed-max", type=float, default=.3)
    parser.add_argument("--force-gain", type=float, default=1.)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=930000)
    parser.add_argument("--independent-seeds", action="store_true")
    parser.add_argument("--physics-only", action="store_true")
    parser.add_argument("--kinds",default="static,glide,free,forced")
    args=parser.parse_args()
    if not .1 <= args.glide_speed_max <= .3 or not 0 < args.force_gain <= 1:
        raise ValueError("motion candidate outside registered development range")
    kinds=args.kinds.split(',')
    if not 1<=args.replicates<100 or not kinds or any(k not in ('static','glide','free','forced') for k in kinds) or len(set(kinds))!=len(kinds):
        raise ValueError('invalid replicate or action-kind design')
    if args.output.exists():
        raise ValueError("attempt directory exists; use a new version")
    args.output.mkdir(parents=True)
    grid=list(itertools.product([.5, 1., 1.5, 2.], [.25, .6, 1., 1.5], [4., 8., 16., 25.]))
    if args.design == "continuous":
        from scipy.stats import qmc
        if args.systems < 1 or args.systems & (args.systems-1):
            raise ValueError("continuous development size must be a power of two")
        grid=qmc.scale(qmc.Sobol(3,scramble=True,seed=940731).random_base2(args.systems.bit_length()-1),
                       [.5,.25,4.],[2.,1.5,25.]).tolist()
    profiles=[(2., 64), (2., 128), (1.5, 64), (1.5, 128), (1., 64), (1., 128)]
    if args.physics_only: profiles=[]
    motion=dict(glide_speed_max=args.glide_speed_max,force_gain=args.force_gain)
    jobs=[]
    for si,theta in enumerate(grid):
        for kind in kinds:
            ki=("static", "glide", "free", "forced").index(kind)
            for replicate in range(args.replicates):
                seed=args.seed_base+ki*100+replicate+(si*10000 if args.independent_seeds else 0)
                jobs.append((len(jobs), theta, kind, replicate, seed, 192, str(args.output), profiles,motion))
    if args.smoke:
        jobs=[jobs[i] for i in sorted(set((min(9,len(jobs)-1),len(jobs)*3//4,len(jobs)-1)))]
    manifest=dict(schema="vec.development-bank.v1.1", split="development", test_read=False,
                  parameter_grid=grid, parameter_design=args.design, jobs=len(jobs), profiles=profiles,
                  common_seeds_across_theta=not args.independent_seeds, independent_seeds_within_theta=True,
                  seed_base=args.seed_base,replicates=args.replicates,kinds=kinds,rendering_performed=not args.physics_only,
                  initial_state_distribution=f"center U[-.08,.08]^2; orientation uniform; free/forced strain +/- .04/.08; glide speed U[.1,{args.glide_speed_max}]",
                  motion_profile=motion,
                  history_windows=[8, 16, 24, 48, 96], physics_transitions=192,
                  camera_policy="same states; full disks and margin; all RK4 stages covered; retained valid prefixes labeled",
                  formal_profile_frozen=False, smoke=args.smoke)
    (args.output/"BANK_CONFIG.json").write_text(json.dumps(manifest, indent=2)+"\n")
    package=Path(__file__).parent
    sources=[dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(package.glob("*.py"))]
    (args.output/"SOURCE_MANIFEST.json").write_text(json.dumps(sources, indent=2)+"\n")
    rows=[]; errors=[]; began=time.monotonic()
    with (args.output/"progress.jsonl").open("w") as f:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=worker_init) as pool:
            pending={pool.submit(generate_one, job): job[0] for job in jobs}
            for future in as_completed(pending):
                try:
                    row=future.result(); rows.append(row)
                    event=dict(index=row["index"], kind=row["kind"], theta=row["theta"],
                               valid=row["full_physics_valid"], completed=len(rows), total=len(jobs),
                               seconds=row["elapsed_seconds"])
                except Exception as exc:
                    event=dict(index=pending[future], error=repr(exc)); errors.append(event)
                f.write(json.dumps(event)+"\n"); f.flush(); print(json.dumps(event), flush=True)
    summary=dict(config=manifest, episodes=sorted(rows, key=lambda x:x["index"]), errors=errors,
                 elapsed_seconds=time.monotonic()-began, status="COMPLETE" if not errors else "EXECUTION_FAILURES_RETAINED")
    (args.output/"BANK_REPORT.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

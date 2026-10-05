from dataclasses import dataclass

import mujoco
import numpy as np


PARAMETER_NAMES = ("mass_0", "mass_1", "mass_2", "damping_1", "damping_2")


def log_uniform_std(bounds) -> float:
    a, b = (float(value) for value in bounds)
    mean = ((b * np.log(b) - b) - (a * np.log(a) - a)) / (b - a)
    second = (
        b * (np.log(b) ** 2 - 2 * np.log(b) + 2)
        - a * (np.log(a) ** 2 - 2 * np.log(a) + 2)
    ) / (b - a)
    return float(np.sqrt(max(second - mean ** 2, 0.0)))


def prior_log_scale(prior: dict) -> np.ndarray:
    return np.asarray([log_uniform_std(prior["mass_scale"])] * 3 + [log_uniform_std(prior["damping_scale"])] * 2)


def _xml(cfg: dict) -> str:
    lengths = cfg["link_length_m"]
    radius = cfg["link_radius_m"]
    density = cfg["base_density"]
    damp = cfg["base_joint_damping"]
    timestep = cfg["timestep_s"]
    ctrl = cfg["control_limit"]
    return f"""
<mujoco model="paper_c_swimmer">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{timestep}" integrator="{cfg['integrator']}" gravity="0 0 0"
          density="{cfg['fluid_density']}" viscosity="{cfg['fluid_viscosity']}"/>
  <default>
    <geom type="capsule" size="{radius}" density="{density}" rgba="0.25 0.45 0.65 1"/>
    <joint type="hinge" axis="0 0 1" limited="true" range="-1.75 1.75"/>
    <motor ctrllimited="true" ctrlrange="-{ctrl} {ctrl}" gear="1"/>
  </default>
  <worldbody>
    <body name="link0">
      <joint name="root_x" type="slide" axis="1 0 0" limited="false" damping="0"/>
      <joint name="root_y" type="slide" axis="0 1 0" limited="false" damping="0"/>
      <joint name="root_angle" limited="false" damping="0.02"/>
      <geom name="geom0" fromto="0 0 0 {lengths[0]} 0 0"/>
      <body name="link1" pos="{lengths[0]} 0 0">
        <joint name="joint1" damping="{damp[0]}"/>
        <geom name="geom1" fromto="0 0 0 {lengths[1]} 0 0"/>
        <body name="link2" pos="{lengths[1]} 0 0">
          <joint name="joint2" damping="{damp[1]}"/>
          <geom name="geom2" fromto="0 0 0 {lengths[2]} 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1"/>
    <motor name="motor2" joint="joint2"/>
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class InitialState:
    qpos: np.ndarray
    qvel: np.ndarray


class SwimmerModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_string(_xml(cfg))
        self.base_mass = self.model.body_mass.copy()
        self.base_inertia = self.model.body_inertia.copy()
        self.base_damping = self.model.dof_damping.copy()

    def apply_log_scales(self, log_scales: np.ndarray) -> None:
        scales = np.exp(np.asarray(log_scales, dtype=float))
        for body_index, scale in zip((1, 2, 3), scales[:3]):
            self.model.body_mass[body_index] = self.base_mass[body_index] * scale
            self.model.body_inertia[body_index] = self.base_inertia[body_index] * scale
        self.model.dof_damping[3] = self.base_damping[3] * scales[3]
        self.model.dof_damping[4] = self.base_damping[4] * scales[4]

    def sample_initial_state(self, rng: np.random.Generator, initial_cfg: dict) -> InitialState:
        qpos = np.zeros(self.model.nq, dtype=float)
        qvel = np.zeros(self.model.nv, dtype=float)
        qpos[2] = rng.uniform(*initial_cfg["root_angle_rad"])
        qpos[3:] = rng.uniform(*initial_cfg["joint_angle_rad"], size=2)
        qvel[:] = rng.uniform(*initial_cfg["velocity"], size=self.model.nv)
        return InitialState(qpos=qpos, qvel=qvel)

    @staticmethod
    def observation(data: mujoco.MjData) -> np.ndarray:
        return np.concatenate([data.qpos[2:].copy(), data.qvel.copy()])

    def rollout(self, log_scales: np.ndarray, initial: InitialState, actions: np.ndarray, landmark_indices: np.ndarray) -> np.ndarray:
        self.apply_log_scales(log_scales)
        data = mujoco.MjData(self.model)
        data.qpos[:] = initial.qpos
        data.qvel[:] = initial.qvel
        mujoco.mj_forward(self.model, data)
        outputs = []
        landmark_set = set(int(value) for value in landmark_indices)
        for step, action in enumerate(actions, start=1):
            data.ctrl[:] = action
            mujoco.mj_step(self.model, data)
            if step in landmark_set:
                outputs.append(self.observation(data))
        return np.concatenate(outputs)

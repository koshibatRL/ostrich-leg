"""
Custom Gymnasium environment for the 3 bipedal leg models.

Works with MuJoCo + Stable-Baselines3.
Implements the shared reward function from base_config.py.
Supports Model A (12 actuators), Model B/C (10 actuators).
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class BipedalWalkEnv(gym.Env):
    """
    Bipedal walking environment for MuJoCo models.

    Observation space: proprioceptive only
      - base angular velocity (3)
      - projected gravity (3)
      - command velocity (3)
      - joint positions relative to default (nq_joints)
      - joint velocities (nv_joints)
      - previous actions (nu)

    Action space: normalized torques [-1, 1] for each actuator
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_xml="models/model_a_forward_knee.xml",
        phase=1,
        target_vel=1.0,
        target_height=0.85,
        episode_length_s=20.0,
        render_mode=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.target_vel = target_vel
        self.target_height = target_height
        self.phase = phase

        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(model_xml)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.dt = self.mj_model.opt.timestep
        self.decimation = 4  # policy at 50Hz if timestep=0.005; adapt for 0.002
        self.policy_dt = self.dt * self.decimation

        self.nu = self.mj_model.nu  # number of actuators
        self.nq_free = 7  # freejoint: 3 pos + 4 quat
        self.nv_free = 6  # freejoint: 3 vel + 3 angvel

        self.nq_joints = self.mj_model.nq - self.nq_free
        self.nv_joints = self.mj_model.nv - self.nv_free

        self.max_episode_steps = int(episode_length_s / self.policy_dt)
        self.step_count = 0

        # Default joint positions (from qpos0)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[2] = target_height
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.default_qpos = self.mj_data.qpos.copy()
        self.default_joint_pos = self.default_qpos[self.nq_free:].copy()

        # Observation: angvel(3) + proj_gravity(3) + cmd(3) + joint_pos(nq) + joint_vel(nv) + prev_act(nu)
        obs_dim = 3 + 3 + 3 + self.nq_joints + self.nv_joints + self.nu
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: normalized torques
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )

        self.prev_actions = np.zeros(self.nu, dtype=np.float32)

        # Reward weights by phase
        if phase == 1:
            self.w = {
                "forward_vel": 1.0,
                "alive": 0.5,
                "torso_upright": 0.3,
                "energy": 0.0,
                "joint_acc": 0.0,
                "torso_height_var": -0.5,
                "action_rate": -0.005,
                "foot_slip": 0.0,
                "joint_limit": -5.0,
            }
        else:
            self.w = {
                "forward_vel": 1.0,
                "alive": 0.5,
                "torso_upright": 0.5,
                "energy": -0.001,
                "joint_acc": -2.5e-7,
                "torso_height_var": -1.0,
                "action_rate": -0.01,
                "foot_slip": -0.5,
                "joint_limit": -10.0,
            }

        # Renderer (lazy init)
        self._renderer = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Set initial pose
        self.mj_data.qpos[:] = self.default_qpos.copy()

        # Small noise
        noise_pos = self.np_random.uniform(-0.02, 0.02, size=self.nq_joints)
        self.mj_data.qpos[self.nq_free:] += noise_pos
        self.mj_data.qpos[2] += self.np_random.uniform(-0.01, 0.01)

        noise_vel = self.np_random.uniform(-0.1, 0.1, size=self.nv_joints)
        self.mj_data.qvel[self.nv_free:] += noise_vel

        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.step_count = 0
        self.prev_actions = np.zeros(self.nu, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Apply action for decimation steps
        self.mj_data.ctrl[:] = action
        for _ in range(self.decimation):
            mujoco.mj_step(self.mj_model, self.mj_data)

        self.step_count += 1

        obs = self._get_obs()
        reward, info = self._compute_reward(action)

        # Termination conditions
        torso_z = self.mj_data.qpos[2]
        quat = self.mj_data.qpos[3:7]
        # tilt angle from upright
        up_vec = self._quat_to_up(quat)
        tilt = np.arccos(np.clip(up_vec[2], -1, 1))

        terminated = bool(torso_z < 0.3 or tilt > 1.2)
        truncated = bool(self.step_count >= self.max_episode_steps)

        info["torso_z"] = torso_z
        info["tilt_rad"] = tilt
        info["episode_step"] = self.step_count

        self.prev_actions = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.distance = 3.0
        camera.elevation = -20.0
        camera.azimuth = 90.0
        camera.lookat[:] = [self.mj_data.qpos[0], 0, 0.6]
        self._renderer.update_scene(self.mj_data, camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_obs(self):
        quat = self.mj_data.qpos[3:7]

        # Base angular velocity (in body frame)
        angvel = self.mj_data.qvel[3:6].copy()

        # Projected gravity in body frame
        proj_gravity = self._quat_rotate_inverse(quat, np.array([0, 0, -1.0]))

        # Command
        cmd = np.array([self.target_vel, 0.0, 0.0], dtype=np.float64)

        # Joint positions relative to default
        joint_pos = self.mj_data.qpos[self.nq_free:] - self.default_joint_pos

        # Joint velocities
        joint_vel = self.mj_data.qvel[self.nv_free:]

        obs = np.concatenate([
            angvel,
            proj_gravity,
            cmd,
            joint_pos,
            joint_vel,
            self.prev_actions,
        ]).astype(np.float32)

        return obs

    def _compute_reward(self, action):
        components = {}

        # Forward velocity tracking
        vel_x = self.mj_data.qvel[0]
        vel_error = vel_x - self.target_vel
        components["forward_vel"] = float(np.exp(-4.0 * vel_error ** 2))

        # Alive bonus
        components["alive"] = 1.0

        # Torso upright
        quat = self.mj_data.qpos[3:7]
        up_vec = self._quat_to_up(quat)
        components["torso_upright"] = float((up_vec[2] + 1.0) / 2.0)

        # Energy penalty (sum |torque * joint_vel| over actuated joints)
        torques = self.mj_data.actuator_force.copy()
        joint_vels = self.mj_data.qvel[self.nv_free:]
        energy = float(np.sum(np.abs(torques * joint_vels[:self.nu])))
        components["energy"] = energy

        # Joint acceleration
        joint_acc = self.mj_data.qacc[self.nv_free:]
        components["joint_acc"] = float(np.sum(joint_acc ** 2))

        # Torso height variation
        height_error = self.mj_data.qpos[2] - self.target_height
        components["torso_height_var"] = float(height_error ** 2)

        # Action rate
        action_diff = action - self.prev_actions
        components["action_rate"] = float(np.sum(action_diff ** 2))

        # Foot slip (use sensor data if available, else approximate)
        components["foot_slip"] = self._compute_foot_slip()

        # Joint limit penalty
        jnt_pos = self.mj_data.qpos[self.nq_free:]
        margin = 0.1
        below = 0.0
        above = 0.0
        for i in range(self.mj_model.njnt):
            if self.mj_model.jnt_type[i] != 3:  # skip non-hinge (freejoint)
                continue
            qpos_idx = self.mj_model.jnt_qposadr[i]
            if qpos_idx < self.nq_free:
                continue
            lo = self.mj_model.jnt_range[i, 0]
            hi = self.mj_model.jnt_range[i, 1]
            q = self.mj_data.qpos[qpos_idx]
            below += max(0, lo + margin - q) ** 2
            above += max(0, q - (hi - margin)) ** 2
        components["joint_limit"] = float(below + above)

        # Weighted sum
        total = 0.0
        for key, value in components.items():
            w = self.w.get(key, 0.0)
            total += w * value

        return float(total), components

    def _compute_foot_slip(self):
        """Approximate foot slip from contact data."""
        # Simple approximation: check if foot bodies have contacts and velocity
        slip = 0.0
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            # Check if one geom is the floor (geom id 0 typically)
            floor_geom = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
            )
            foot_geom = None
            if geom1 == floor_geom:
                foot_geom = geom2
            elif geom2 == floor_geom:
                foot_geom = geom1
            else:
                continue

            # Get the body of the foot geom
            body_id = self.mj_model.geom_bodyid[foot_geom]
            # Get body velocity (linear)
            body_vel = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.mj_model, self.mj_data,
                mujoco.mjtObj.mjOBJ_BODY, body_id, body_vel, 0
            )
            # Horizontal velocity magnitude
            slip += np.sqrt(body_vel[3] ** 2 + body_vel[4] ** 2)

        return float(slip)

    @staticmethod
    def _quat_to_up(quat):
        """Get the up vector (z-axis) of the body in world frame from quaternion."""
        w, x, y, z = quat
        # Rotation matrix 3rd column (up direction)
        up = np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y),
        ])
        return up

    @staticmethod
    def _quat_rotate_inverse(quat, vec):
        """Rotate vector by inverse of quaternion."""
        w, x, y, z = quat
        # Conjugate quaternion
        q_conj = np.array([w, -x, -y, -z])
        # Quaternion multiply: q_conj * vec_quat * q
        vec_quat = np.array([0, vec[0], vec[1], vec[2]])

        def qmul(a, b):
            return np.array([
                a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
                a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
                a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
                a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
            ])

        result = qmul(qmul(q_conj, vec_quat), quat)
        return result[1:4]

"""
MJX-based (GPU-accelerated) bipedal walking environment.

All operations are JAX-compatible for jit compilation.
Runs thousands of parallel environments on a single GPU.
"""

import jax
import jax.numpy as jnp
from jax import random
import mujoco
from mujoco import mjx
from functools import partial
from typing import NamedTuple


class EnvState(NamedTuple):
    """Full state of one environment instance (batched over N envs)."""
    mjx_data: mjx.Data
    step_count: jnp.ndarray      # (N,)
    prev_actions: jnp.ndarray    # (N, nu)
    rng: jnp.ndarray             # (N, 2)
    done: jnp.ndarray            # (N,)


class BipedalMJXEnv:
    """
    GPU-parallel bipedal walking environment using MJX.

    Usage:
        env = BipedalMJXEnv("models/model_a_forward_knee.xml", num_envs=4096)
        state, obs = env.reset(jax.random.PRNGKey(0))
        state, obs, reward, done, info = env.step(state, actions)
    """

    def __init__(
        self,
        model_xml: str,
        num_envs: int = 4096,
        phase: int = 1,
        target_vel: float = 1.0,
        target_height: float = 0.85,
        episode_length_s: float = 20.0,
    ):
        # Load MuJoCo model and convert to MJX
        self.mj_model = mujoco.MjModel.from_xml_path(model_xml)
        self.mjx_model = mjx.put_model(self.mj_model)

        self.num_envs = num_envs
        self.target_vel = target_vel
        self.target_height = target_height
        self.phase = phase

        self.dt = self.mj_model.opt.timestep
        self.decimation = 4
        self.policy_dt = self.dt * self.decimation

        self.nu = self.mj_model.nu
        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv
        self.nq_free = 7
        self.nv_free = 6
        self.nq_joints = self.nq - self.nq_free
        self.nv_joints = self.nv - self.nv_free

        self.max_episode_steps = int(episode_length_s / self.policy_dt)

        # Observation dimension
        self.obs_dim = 3 + 3 + 3 + self.nq_joints + self.nv_joints + self.nu

        # Get default qpos
        mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, mj_data)
        mj_data.qpos[2] = target_height
        mujoco.mj_forward(self.mj_model, mj_data)
        self.default_qpos = jnp.array(mj_data.qpos)
        self.default_qvel = jnp.zeros(self.nv)
        self.default_joint_pos = self.default_qpos[self.nq_free:]

        # Store a single MJX data template for resetting
        self._mjx_data_template = mjx.put_data(self.mj_model, mj_data)

        # Joint ranges for limit penalty
        self.jnt_range = jnp.array(self.mj_model.jnt_range[1:])  # skip freejoint

        # Reward weights
        if phase == 1:
            self.w_forward_vel = 1.0
            self.w_alive = 0.5
            self.w_torso_upright = 0.3
            self.w_energy = 0.0
            self.w_joint_acc = 0.0
            self.w_torso_height_var = -0.5
            self.w_action_rate = -0.005
            self.w_joint_limit = -5.0
        else:
            self.w_forward_vel = 1.0
            self.w_alive = 0.5
            self.w_torso_upright = 0.5
            self.w_energy = -0.001
            self.w_joint_acc = -2.5e-7
            self.w_torso_height_var = -1.0
            self.w_action_rate = -0.01
            self.w_joint_limit = -10.0

    def _reset_single(self, rng):
        """Reset a single environment. Returns (mjx_data, rng)."""
        rng, k1, k2, k3 = random.split(rng, 4)

        qpos = self.default_qpos.copy()
        qvel = self.default_qvel.copy()

        # Add noise
        joint_noise = random.uniform(k1, shape=(self.nq_joints,), minval=-0.02, maxval=0.02)
        height_noise = random.uniform(k2, shape=(), minval=-0.01, maxval=0.01)
        vel_noise = random.uniform(k3, shape=(self.nv_joints,), minval=-0.1, maxval=0.1)

        qpos = qpos.at[self.nq_free:].add(joint_noise)
        qpos = qpos.at[2].add(height_noise)
        qvel = qvel.at[self.nv_free:].add(vel_noise)

        data = self._mjx_data_template.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)

        return data, rng

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng):
        """Reset all environments. Returns (state, obs)."""
        rngs = random.split(rng, self.num_envs)

        mjx_datas, new_rngs = jax.vmap(self._reset_single)(rngs)

        state = EnvState(
            mjx_data=mjx_datas,
            step_count=jnp.zeros(self.num_envs, dtype=jnp.int32),
            prev_actions=jnp.zeros((self.num_envs, self.nu)),
            rng=new_rngs,
            done=jnp.zeros(self.num_envs, dtype=jnp.bool_),
        )

        obs = self._get_obs(state)
        return state, obs

    def _step_single(self, mjx_model, data, action):
        """Simulate decimation steps for a single env."""
        data = data.replace(ctrl=action)

        def body_fn(data, _):
            return mjx.step(mjx_model, data), None

        data, _ = jax.lax.scan(body_fn, data, None, length=self.decimation)
        return data

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, actions):
        """Step all environments. Returns (new_state, obs, reward, done, info)."""
        actions = jnp.clip(actions, -1.0, 1.0)

        # Step physics (vmapped over envs)
        mjx_datas = jax.vmap(
            lambda d, a: self._step_single(self.mjx_model, d, a)
        )(state.mjx_data, actions)

        new_step_count = state.step_count + 1

        # Compute reward
        reward = self._compute_reward_batched(mjx_datas, actions, state.prev_actions)

        # Termination
        torso_z = mjx_datas.qpos[:, 2]
        quat = mjx_datas.qpos[:, 3:7]
        up_vec = jax.vmap(self._quat_to_up)(quat)
        tilt = jnp.arccos(jnp.clip(up_vec[:, 2], -1.0, 1.0))

        terminated = (torso_z < 0.3) | (tilt > 1.2)
        truncated = new_step_count >= self.max_episode_steps
        done = terminated | truncated

        # Auto-reset done environments
        new_rngs = jax.vmap(lambda k: random.split(k)[0])(state.rng)
        reset_rngs = jax.vmap(lambda k: random.split(k)[1])(state.rng)

        reset_datas, reset_rngs_out = jax.vmap(self._reset_single)(reset_rngs)

        # Where done, use reset data; else keep stepped data
        new_mjx_datas = jax.tree.map(
            lambda reset, stepped: jnp.where(
                done.reshape(-1, *([1] * (reset.ndim - 1))),
                reset, stepped
            ),
            reset_datas, mjx_datas
        )

        new_step_count = jnp.where(done, 0, new_step_count)
        new_prev_actions = jnp.where(done[:, None], jnp.zeros_like(actions), actions)
        new_rngs = jnp.where(done[:, None], reset_rngs_out, new_rngs)

        new_state = EnvState(
            mjx_data=new_mjx_datas,
            step_count=new_step_count,
            prev_actions=new_prev_actions,
            rng=new_rngs,
            done=done,
        )

        obs = self._get_obs(new_state)

        info = {
            "torso_z": torso_z,
            "tilt": tilt,
            "terminated": terminated,
            "truncated": truncated,
        }

        return new_state, obs, reward, done, info

    def _get_obs(self, state):
        """Extract observation from state. (num_envs, obs_dim)"""
        data = state.mjx_data

        angvel = data.qvel[:, 3:6]

        quat = data.qpos[:, 3:7]
        gravity_world = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (self.num_envs, 1))
        proj_gravity = jax.vmap(self._quat_rotate_inverse)(quat, gravity_world)

        cmd = jnp.tile(
            jnp.array([self.target_vel, 0.0, 0.0]),
            (self.num_envs, 1)
        )

        joint_pos = data.qpos[:, self.nq_free:] - self.default_joint_pos
        joint_vel = data.qvel[:, self.nv_free:]

        obs = jnp.concatenate([
            angvel, proj_gravity, cmd, joint_pos, joint_vel, state.prev_actions
        ], axis=-1)

        return obs

    def _compute_reward_batched(self, data, actions, prev_actions):
        """Vectorized reward computation. Returns (num_envs,)."""
        # Forward velocity
        vel_x = data.qvel[:, 0]
        vel_error = vel_x - self.target_vel
        r_forward_vel = jnp.exp(-4.0 * vel_error ** 2)

        # Alive
        r_alive = jnp.ones(self.num_envs)

        # Torso upright
        quat = data.qpos[:, 3:7]
        up_vec = jax.vmap(self._quat_to_up)(quat)
        r_torso_upright = (up_vec[:, 2] + 1.0) / 2.0

        # Energy
        torques = data.actuator_force
        joint_vels = data.qvel[:, self.nv_free:self.nv_free + self.nu]
        p_energy = jnp.sum(jnp.abs(torques * joint_vels), axis=-1)

        # Joint acceleration
        joint_acc = data.qacc[:, self.nv_free:]
        p_joint_acc = jnp.sum(joint_acc ** 2, axis=-1)

        # Torso height variation
        height_error = data.qpos[:, 2] - self.target_height
        p_torso_height_var = height_error ** 2

        # Action rate
        action_diff = actions - prev_actions
        p_action_rate = jnp.sum(action_diff ** 2, axis=-1)

        # Joint limit penalty
        jnt_pos = data.qpos[:, self.nq_free:]
        margin = 0.1
        lo = self.jnt_range[:, 0]
        hi = self.jnt_range[:, 1]
        below = jnp.sum(jnp.clip(lo + margin - jnt_pos, 0, None) ** 2, axis=-1)
        above = jnp.sum(jnp.clip(jnt_pos - (hi - margin), 0, None) ** 2, axis=-1)
        p_joint_limit = below + above

        # Weighted sum
        reward = (
            self.w_forward_vel * r_forward_vel
            + self.w_alive * r_alive
            + self.w_torso_upright * r_torso_upright
            + self.w_energy * p_energy
            + self.w_joint_acc * p_joint_acc
            + self.w_torso_height_var * p_torso_height_var
            + self.w_action_rate * p_action_rate
            + self.w_joint_limit * p_joint_limit
        )

        return reward

    @staticmethod
    def _quat_to_up(quat):
        w, x, y, z = quat
        return jnp.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y),
        ])

    @staticmethod
    def _quat_rotate_inverse(quat, vec):
        w, x, y, z = quat
        q_conj = jnp.array([w, -x, -y, -z])
        vec_quat = jnp.array([0.0, vec[0], vec[1], vec[2]])

        def qmul(a, b):
            return jnp.array([
                a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
                a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
                a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
                a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
            ])

        result = qmul(qmul(q_conj, vec_quat), quat)
        return result[1:4]

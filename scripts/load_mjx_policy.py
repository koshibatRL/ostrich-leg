"""
Utility: load a trained MJX policy (JAX params) and wrap it for CPU evaluation.

Usage:
    from load_mjx_policy import load_mjx_policy
    policy = load_mjx_policy("results/checkpoints/forward_knee_phase1_.../ckpt_final.npz",
                             obs_dim=45, action_dim=12)
    action = policy(obs_numpy)  # obs: (obs_dim,) numpy -> action: (action_dim,) numpy
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


class ActorCritic(nn.Module):
    """Must match the architecture in ppo_jax.py."""
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256, 128)

    @nn.compact
    def __call__(self, x):
        # Actor
        a = x
        for dim in self.hidden_dims:
            a = nn.Dense(dim)(a)
            a = nn.elu(a)
        mean = nn.Dense(self.action_dim)(a)
        log_std = self.param(
            "log_std", nn.initializers.zeros, (self.action_dim,)
        )

        # Critic
        v = x
        for dim in self.hidden_dims:
            v = nn.Dense(dim)(v)
            v = nn.elu(v)
        value = nn.Dense(1)(v)

        return mean, log_std, value.squeeze(-1)


def load_mjx_policy(checkpoint_path, obs_dim, action_dim):
    """
    Load MJX checkpoint (.npz) and return a callable policy function.

    Args:
        checkpoint_path: path to .npz checkpoint
        obs_dim: observation dimension
        action_dim: action dimension

    Returns:
        policy: callable (numpy obs -> numpy action), deterministic (uses mean)
    """
    # Initialize network to get param structure
    network = ActorCritic(action_dim=action_dim)
    dummy_obs = jnp.zeros((1, obs_dim))
    rng = jax.random.PRNGKey(0)
    params_template = network.init(rng, dummy_obs)

    # Load saved params
    data = np.load(checkpoint_path, allow_pickle=True)
    leaves_template = jax.tree.leaves(params_template)
    tree_def = jax.tree.structure(params_template)

    # Reconstruct params from flat arrays
    loaded_leaves = []
    for i in range(len(leaves_template)):
        loaded_leaves.append(jnp.array(data[str(i)]))

    params = jax.tree.unflatten(tree_def, loaded_leaves)

    # Create deterministic policy function
    @jax.jit
    def _forward(params, obs):
        mean, log_std, value = network.apply(params, obs)
        return mean

    def policy(obs_numpy):
        """obs_numpy: (obs_dim,) or (batch, obs_dim) numpy array -> action numpy array"""
        if obs_numpy.ndim == 1:
            obs = jnp.array(obs_numpy[None])
            action = np.array(_forward(params, obs)[0])
        else:
            obs = jnp.array(obs_numpy)
            action = np.array(_forward(params, obs))
        return np.clip(action, -1.0, 1.0)

    return policy


def get_obs_from_mujoco(model, data, default_joint_pos, nu):
    """
    Build observation vector from MuJoCo data matching BipedalWalkEnv/BipedalMJXEnv format.

    Obs = [angvel(3), proj_gravity(3), cmd(3), joint_pos(nq_joints), joint_vel(nv_joints), prev_actions(nu)]
    """
    nq_free = 7
    nv_free = 6

    # Angular velocity
    angvel = data.qvel[3:6].copy()

    # Projected gravity via inverse quaternion rotation
    quat = data.qpos[3:7]
    w, x, y, z = quat
    q_conj = np.array([w, -x, -y, -z])
    gvec = np.array([0, 0, -1.0])

    def qmul(a, b):
        return np.array([
            a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
            a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
            a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0],
        ])

    gq = np.array([0, gvec[0], gvec[1], gvec[2]])
    proj_g = qmul(qmul(q_conj, gq), quat)[1:4]

    cmd = np.array([1.0, 0.0, 0.0])
    joint_pos = data.qpos[nq_free:] - default_joint_pos
    joint_vel = data.qvel[nv_free:]

    obs = np.concatenate([
        angvel, proj_g, cmd, joint_pos, joint_vel, np.zeros(nu)
    ]).astype(np.float32)

    return obs

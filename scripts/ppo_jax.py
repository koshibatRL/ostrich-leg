"""
Minimal PPO implementation in JAX for MJX environments.

Fully jit-compiled training loop running entirely on GPU.
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple, Sequence
import distrax


# ==============================================================
# Actor-Critic Network
# ==============================================================
class ActorCritic(nn.Module):
    """Shared trunk with separate actor/critic heads."""
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


# ==============================================================
# PPO Rollout Buffer
# ==============================================================
class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray


# ==============================================================
# PPO Training
# ==============================================================
class PPOConfig(NamedTuple):
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    n_epochs: int = 5
    n_minibatches: int = 4
    n_steps: int = 64       # rollout length per env
    normalize_advantages: bool = True


def create_train_state(rng, obs_dim, action_dim, config: PPOConfig):
    """Initialize network and optimizer."""
    network = ActorCritic(action_dim=action_dim)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = network.init(rng, dummy_obs)

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.lr),
    )

    return TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )


def get_action_and_value(train_state, obs, rng):
    """Sample action from policy, return action, log_prob, value."""
    mean, log_std, value = train_state.apply_fn(train_state.params, obs)
    std = jnp.exp(log_std)
    dist = distrax.MultivariateNormalDiag(mean, std)
    action = dist.sample(seed=rng)
    log_prob = dist.log_prob(action)
    return action, log_prob, value


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Compute GAE advantages. All inputs: (n_steps, num_envs)."""

    def _scan_fn(carry, t):
        next_value, gae = carry
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        return (values[t], gae), gae

    _, advantages = jax.lax.scan(
        _scan_fn,
        (last_value, jnp.zeros_like(last_value)),
        jnp.arange(rewards.shape[0] - 1, -1, -1),
    )
    advantages = advantages[::-1]  # reverse back to chronological order
    returns = advantages + values
    return advantages, returns


def ppo_update(train_state, batch, config: PPOConfig, rng):
    """Single PPO update over the batch."""

    obs, actions, old_log_probs, advantages, returns = batch

    if config.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_size = obs.shape[0]
    batch_size = total_size // config.n_minibatches

    def _epoch_step(carry, _):
        train_state, rng = carry
        rng, perm_rng = random.split(rng)
        perm = random.permutation(perm_rng, total_size)

        def _minibatch_step(train_state, start_idx):
            idx = jax.lax.dynamic_slice(perm, (start_idx,), (batch_size,))
            mb_obs = obs[idx]
            mb_actions = actions[idx]
            mb_old_log_probs = old_log_probs[idx]
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]

            def loss_fn(params):
                mean, log_std, values = train_state.apply_fn(params, mb_obs)
                std = jnp.exp(log_std)
                dist = distrax.MultivariateNormalDiag(mean, std)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Policy loss (clipped)
                ratio = jnp.exp(log_probs - mb_old_log_probs)
                pg_loss1 = ratio * mb_advantages
                pg_loss2 = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * mb_advantages
                pg_loss = -jnp.minimum(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((values - mb_returns) ** 2).mean()

                total_loss = pg_loss + config.value_coef * v_loss - config.entropy_coef * entropy

                return total_loss, {
                    "pg_loss": pg_loss,
                    "v_loss": v_loss,
                    "entropy": entropy,
                    "approx_kl": ((ratio - 1) - jnp.log(ratio)).mean(),
                }

            grads, info = jax.grad(loss_fn, has_aux=True)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, info

        start_indices = jnp.arange(config.n_minibatches) * batch_size
        train_state, infos = jax.lax.scan(_minibatch_step, train_state, start_indices)

        return (train_state, rng), infos

    (train_state, _), epoch_infos = jax.lax.scan(
        _epoch_step, (train_state, rng), None, length=config.n_epochs
    )

    # Average metrics over epochs and minibatches
    metrics = jax.tree.map(lambda x: x.mean(), epoch_infos)
    return train_state, metrics

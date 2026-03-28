"""
GPU-accelerated training using MJX + JAX PPO.

Usage:
    python scripts/train_mjx.py --model A --phase 1 --num_envs 4096 --max_iterations 1000
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from mjx_env import BipedalMJXEnv
from ppo_jax import (
    PPOConfig, create_train_state, compute_gae, ppo_update
)

import distrax


MODEL_CONFIGS = {
    "A": ("models/model_a_forward_knee.xml", "forward_knee"),
    "B": ("models/model_b_reverse_knee.xml", "reverse_knee"),
    "C": ("models/model_c_bidirectional_knee.xml", "bidirectional_knee"),
}


@jax.jit
def get_action_and_value(train_state, obs, rng):
    mean, log_std, value = train_state.apply_fn(train_state.params, obs)
    std = jnp.exp(log_std)
    dist = distrax.MultivariateNormalDiag(mean, std)
    action = dist.sample(seed=rng)
    log_prob = dist.log_prob(action)
    return action, log_prob, value


def train(args):
    model_xml, model_name = MODEL_CONFIGS[args.model]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_phase{args.phase}_{timestamp}"
    log_dir = os.path.join("results", "checkpoints", run_name)
    os.makedirs(log_dir, exist_ok=True)

    run_config = vars(args)
    run_config["backend"] = "mjx"
    run_config["timestamp"] = timestamp
    with open(os.path.join(log_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"{'=' * 60}")
    print(f"  BIPEDAL LEG PLATFORM - MJX GPU Training")
    print(f"  Model:      {args.model} ({model_name})")
    print(f"  Phase:      {args.phase}")
    print(f"  Envs:       {args.num_envs}")
    print(f"  Iterations: {args.max_iterations}")
    print(f"  n_steps:    {args.n_steps}")
    print(f"  decimation: {args.decimation}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  Log dir:    {log_dir}")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    env = BipedalMJXEnv(
        model_xml=model_xml,
        num_envs=args.num_envs,
        phase=args.phase,
        target_vel=1.0,
        episode_length_s=20.0,
        decimation=args.decimation,
    )

    ppo_config = PPOConfig(
        n_steps=args.n_steps,
        n_minibatches=args.n_minibatches,
    )

    rng = random.PRNGKey(args.seed)
    rng, init_rng, env_rng = random.split(rng, 3)

    train_state = create_train_state(
        init_rng, env.obs_dim, env.nu, ppo_config
    )

    print(f"\n[INFO] Obs dim: {env.obs_dim}, Action dim: {env.nu}")
    print(f"[INFO] Total params: {sum(p.size for p in jax.tree.leaves(train_state.params)):,}")
    sys.stdout.flush()

    # ============================================================
    # JIT Compilation (warmup)
    # ============================================================
    print("[INFO] JIT compiling env.reset...")
    sys.stdout.flush()
    t0 = time.time()
    env_state, obs = env.reset(env_rng)
    jax.block_until_ready(obs)
    print(f"[INFO] env.reset compiled in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    print("[INFO] JIT compiling env.step...")
    sys.stdout.flush()
    t0 = time.time()
    dummy_actions = jnp.zeros((env.num_envs, env.nu))
    env_state, obs, reward, done, info = env.step(env_state, dummy_actions)
    jax.block_until_ready(reward)
    print(f"[INFO] env.step compiled in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    print("[INFO] JIT compiling policy forward...")
    sys.stdout.flush()
    t0 = time.time()
    rng, action_rng = random.split(rng)
    action, log_prob, value = get_action_and_value(train_state, obs, action_rng)
    jax.block_until_ready(action)
    print(f"[INFO] policy compiled in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    # JIT compile PPO update
    print("[INFO] JIT compiling PPO update...")
    sys.stdout.flush()
    t0 = time.time()
    batch_size = ppo_config.n_steps * env.num_envs
    dummy_batch = (
        jnp.zeros((batch_size, env.obs_dim)),
        jnp.zeros((batch_size, env.nu)),
        jnp.zeros(batch_size),
        jnp.zeros(batch_size),
        jnp.zeros(batch_size),
    )
    rng, update_rng = random.split(rng)
    train_state, metrics = ppo_update(train_state, dummy_batch, ppo_config, update_rng)
    jax.block_until_ready(metrics["pg_loss"])
    print(f"[INFO] PPO update compiled in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    # Re-reset after warmup
    rng, env_rng = random.split(rng)
    env_state, obs = env.reset(env_rng)

    print("[INFO] Starting training loop...")
    sys.stdout.flush()

    # ============================================================
    # Training loop
    # ============================================================
    total_steps = 0
    start_time = time.time()
    best_reward = -float("inf")
    reward_history = []

    for iteration in range(args.max_iterations):
        # --- Collect rollout ---
        rollout_obs = []
        rollout_actions = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []
        rollout_log_probs = []

        for step in range(ppo_config.n_steps):
            rng, action_rng = random.split(rng)
            action, log_prob, value = get_action_and_value(
                train_state, obs, action_rng
            )

            rollout_obs.append(obs)
            rollout_actions.append(action)
            rollout_values.append(value)
            rollout_log_probs.append(log_prob)

            env_state, obs, reward, done, info = env.step(env_state, action)

            rollout_rewards.append(reward)
            rollout_dones.append(done.astype(jnp.float32))

        # Stack: (n_steps, num_envs, ...)
        t_obs = jnp.stack(rollout_obs)
        t_actions = jnp.stack(rollout_actions)
        t_rewards = jnp.stack(rollout_rewards)
        t_dones = jnp.stack(rollout_dones)
        t_values = jnp.stack(rollout_values)
        t_log_probs = jnp.stack(rollout_log_probs)

        # Last value for GAE
        _, _, last_value = get_action_and_value(train_state, obs, rng)

        advantages, returns = compute_gae(
            t_rewards, t_values, t_dones, last_value,
            ppo_config.gamma, ppo_config.gae_lambda
        )

        # Flatten
        batch = (
            t_obs.reshape(batch_size, -1),
            t_actions.reshape(batch_size, -1),
            t_log_probs.reshape(batch_size),
            advantages.reshape(batch_size),
            returns.reshape(batch_size),
        )

        rng, update_rng = random.split(rng)
        train_state, metrics = ppo_update(
            train_state, batch, ppo_config, update_rng
        )

        total_steps += ppo_config.n_steps * env.num_envs
        mean_reward = float(t_rewards.mean())
        reward_history.append(mean_reward)

        if iteration % args.log_interval == 0 or iteration == args.max_iterations - 1:
            jax.block_until_ready(metrics["pg_loss"])
            elapsed = time.time() - start_time
            fps = total_steps / max(elapsed, 1)
            recent = reward_history[-100:]

            print(
                f"  iter {iteration:>5d} | "
                f"reward={mean_reward:>7.3f} (avg100={np.mean(recent):>7.3f}) | "
                f"pg={float(metrics['pg_loss']):>8.4f} v={float(metrics['v_loss']):>8.3f} "
                f"ent={float(metrics['entropy']):>6.3f} | "
                f"done={float(t_dones.mean()):.2f} | "
                f"fps={fps:,.0f} | "
                f"steps={total_steps:,}"
            )
            sys.stdout.flush()

        if (iteration + 1) % args.save_interval == 0:
            save_checkpoint(train_state, log_dir, iteration + 1)

        if mean_reward > best_reward:
            best_reward = mean_reward

    # ============================================================
    # Final save
    # ============================================================
    jax.block_until_ready(train_state.params)
    elapsed = time.time() - start_time
    total_fps = total_steps / max(elapsed, 1)

    save_checkpoint(train_state, log_dir, "final")

    summary = {
        "model": args.model,
        "model_name": model_name,
        "phase": args.phase,
        "decimation": args.decimation,
        "total_iterations": args.max_iterations,
        "total_steps": total_steps,
        "elapsed_s": elapsed,
        "avg_fps": total_fps,
        "best_reward": best_reward,
        "final_reward_avg100": float(np.mean(reward_history[-100:])),
    }
    with open(os.path.join(log_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(log_dir, "reward_history.json"), "w") as f:
        json.dump(reward_history, f)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total steps:    {total_steps:,}")
    print(f"  Elapsed:        {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Avg FPS:        {total_fps:,.0f}")
    print(f"  Best reward:    {best_reward:.3f}")
    print(f"  Final avg(100): {np.mean(reward_history[-100:]):.3f}")
    print(f"  Saved to:       {log_dir}")
    print(f"{'=' * 60}")


def save_checkpoint(train_state, log_dir, label):
    """Save params as .npz with tree structure info."""
    ckpt_path = os.path.join(log_dir, f"ckpt_{label}.npz")

    # Flatten params to numpy arrays
    leaves = jax.tree.leaves(train_state.params)
    params_dict = {str(i): np.array(v) for i, v in enumerate(leaves)}

    # Save tree structure for reconstruction
    tree_def = jax.tree.structure(train_state.params)
    params_dict["_tree_def_str"] = np.array(str(tree_def))

    np.savez(ckpt_path, **params_dict)
    print(f"  [SAVED] {ckpt_path}")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="MJX GPU Training")
    parser.add_argument("--model", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=64)
    parser.add_argument("--n_minibatches", type=int, default=4)
    parser.add_argument("--decimation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

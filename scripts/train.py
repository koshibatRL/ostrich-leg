"""
Bipedal Leg Platform — Training Script

Usage:
    python scripts/train.py --model A --phase 1 --num_envs 64 --max_iterations 200
    python scripts/train.py --model B --phase 2 --num_envs 64 --max_iterations 15000
"""

import argparse
import os
import json
import time
from datetime import datetime

import numpy as np


MODEL_CONFIGS = {
    "A": ("models/model_a_forward_knee.xml", "forward_knee"),
    "B": ("models/model_b_reverse_knee.xml", "reverse_knee"),
    "C": ("models/model_c_bidirectional_knee.xml", "bidirectional_knee"),
}


def make_env(model_xml, phase, target_vel=1.0, rank=0, seed=42):
    """Create a single environment instance (for SubprocVecEnv)."""
    def _init():
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from bipedal_env import BipedalWalkEnv
        env = BipedalWalkEnv(
            model_xml=model_xml,
            phase=phase,
            target_vel=target_vel,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train bipedal leg models")
    parser.add_argument("--model", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_iterations", type=int, default=200,
                        help="Number of PPO updates (total_timesteps = max_iterations * num_envs * n_steps)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_steps", type=int, default=64,
                        help="Steps per env per PPO update")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    model_xml, model_name = MODEL_CONFIGS[args.model]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_phase{args.phase}_{timestamp}"
    log_dir = os.path.join("results", "checkpoints", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Save run config
    run_config = {
        "model": args.model,
        "model_xml": model_xml,
        "model_name": model_name,
        "phase": args.phase,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "n_steps": args.n_steps,
        "seed": args.seed,
        "device": args.device,
        "timestamp": timestamp,
    }
    with open(os.path.join(log_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"={'=' * 60}")
    print(f"  BIPEDAL LEG PLATFORM — Training")
    print(f"  Model:      {args.model} ({model_name})")
    print(f"  Phase:      {args.phase}")
    print(f"  Envs:       {args.num_envs}")
    print(f"  Iterations: {args.max_iterations}")
    print(f"  n_steps:    {args.n_steps}")
    print(f"  Device:     {args.device}")
    print(f"  Log dir:    {log_dir}")
    print(f"={'=' * 60}")

    # Import here to fail fast if not installed
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, CallbackList, BaseCallback
    )

    # Create vectorized environment
    env = SubprocVecEnv(
        [make_env(model_xml, args.phase, rank=i, seed=args.seed)
         for i in range(args.num_envs)]
    )

    total_timesteps = args.max_iterations * args.num_envs * args.n_steps

    # PPO config matching base_config.py
    if args.resume:
        print(f"[INFO] Resuming from {args.resume}")
        ppo = PPO.load(args.resume, env=env, device=args.device)
    else:
        ppo = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=max(args.num_envs * args.n_steps // 4, 64),
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=1.0,
            verbose=1,
            tensorboard_log=log_dir,
            device=args.device,
            seed=args.seed,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            ),
        )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 1000),
        save_path=log_dir,
        name_prefix=f"model_{args.model}",
    )

    class RewardLogger(BaseCallback):
        """Log average reward periodically."""
        def __init__(self, log_interval=10):
            super().__init__()
            self.log_interval = log_interval
            self.rewards_buffer = []

        def _on_step(self):
            # Collect rewards from info
            if self.locals.get("infos"):
                for info in self.locals["infos"]:
                    if "episode" in info:
                        self.rewards_buffer.append(info["episode"]["r"])
            return True

        def _on_rollout_end(self):
            if self.rewards_buffer:
                mean_r = np.mean(self.rewards_buffer[-100:])
                print(f"  [iter {self.num_timesteps:>8d}] mean_reward={mean_r:.3f} "
                      f"(n_episodes={len(self.rewards_buffer)})")
                self.rewards_buffer = []

    callbacks = CallbackList([checkpoint_cb, RewardLogger()])

    print(f"\n[INFO] Starting training: {total_timesteps} total timesteps")
    start_time = time.time()

    ppo.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )

    elapsed = time.time() - start_time
    print(f"\n[DONE] Training completed in {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    # Save final model
    final_path = os.path.join(log_dir, f"model_{args.model}_final")
    ppo.save(final_path)
    print(f"[SAVED] {final_path}.zip")

    env.close()


if __name__ == "__main__":
    main()

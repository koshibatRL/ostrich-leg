"""
Bipedal Leg Platform — Training Script

Usage:
    python train.py --model A --phase 1 --num_envs 4096 --max_iterations 2000
    python train.py --model B --phase 2 --num_envs 4096 --max_iterations 15000
    python train.py --model C --phase 2 --num_envs 4096 --max_iterations 20000

This script is a template. The actual training loop depends on which
framework is available on the H200 server:
  - Isaac Lab / legged_gym: Use their native training loop
  - MuJoCo + SB3: Use Stable-Baselines3 PPO
  - MuJoCo + MJX: Use custom JAX-based PPO

The reward computation logic below is framework-independent and should be
integrated into whichever training framework is used.
"""

import argparse
import os
import json
import time
from datetime import datetime

import numpy as np


# ============================================================
# REWARD COMPUTATION (Framework-independent)
# ============================================================
class BipedalRewardComputer:
    """
    Computes reward for bipedal walking task.
    This class is designed to be called from any training framework.

    All 3 models use the same reward function for fair comparison.
    The structural differences (fewer actuators, springs) naturally
    lead to different energy consumption without reward bias.
    """

    def __init__(self, phase=2, target_vel=1.0, target_height=0.85):
        self.target_vel = target_vel
        self.target_height = target_height

        # Select reward weights based on training phase
        if phase == 1:
            self.weights = {
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
        else:  # phase 2
            self.weights = {
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

        self.prev_actions = None

    def compute(self, obs_dict):
        """
        Compute total reward from observation dictionary.

        Args:
            obs_dict: dict with keys:
                - base_vel_x: float, forward velocity (m/s)
                - projected_gravity_z: float, dot(torso_up, world_up)
                - torso_height: float, z position of torso (m)
                - joint_torques: np.array, actuator torques
                - joint_velocities: np.array, joint angular velocities
                - joint_accelerations: np.array, joint angular accelerations
                - actions: np.array, current control actions
                - foot_velocities_xy: np.array, (n_feet, 2) horizontal foot vel
                - foot_contacts: np.array, (n_feet,) boolean contact flags
                - joint_positions: np.array, current joint positions
                - joint_limits_lower: np.array, lower joint limits
                - joint_limits_upper: np.array, upper joint limits

        Returns:
            total_reward: float
            reward_components: dict (for logging)
        """
        components = {}

        # --- Forward velocity tracking ---
        vel_error = obs_dict["base_vel_x"] - self.target_vel
        components["forward_vel"] = float(np.exp(-4.0 * vel_error ** 2))

        # --- Alive bonus ---
        components["alive"] = 1.0

        # --- Torso upright ---
        gz = obs_dict["projected_gravity_z"]
        components["torso_upright"] = float((gz + 1.0) / 2.0)

        # --- Energy penalty ---
        torques = obs_dict["joint_torques"]
        joint_vels = obs_dict["joint_velocities"]
        # Only sum over ACTUATED joints (this is where Model B/C naturally win)
        energy = float(np.sum(np.abs(torques * joint_vels[:len(torques)])))
        components["energy"] = energy

        # --- Joint acceleration penalty ---
        joint_acc = obs_dict["joint_accelerations"]
        components["joint_acc"] = float(np.sum(joint_acc ** 2))

        # --- Torso height variation ---
        height_error = obs_dict["torso_height"] - self.target_height
        components["torso_height_var"] = float(height_error ** 2)

        # --- Action rate penalty ---
        actions = obs_dict["actions"]
        if self.prev_actions is not None:
            action_diff = actions - self.prev_actions
            components["action_rate"] = float(np.sum(action_diff ** 2))
        else:
            components["action_rate"] = 0.0
        self.prev_actions = actions.copy()

        # --- Foot slip penalty ---
        foot_vels = obs_dict["foot_velocities_xy"]
        foot_contacts = obs_dict["foot_contacts"]
        slip = float(np.sum(
            np.linalg.norm(foot_vels, axis=-1) * foot_contacts
        ))
        components["foot_slip"] = slip

        # --- Joint limit penalty ---
        positions = obs_dict["joint_positions"]
        lower = obs_dict["joint_limits_lower"]
        upper = obs_dict["joint_limits_upper"]
        margin = 0.1  # radians
        below = np.sum(np.clip(lower + margin - positions, 0, None) ** 2)
        above = np.sum(np.clip(positions - (upper - margin), 0, None) ** 2)
        components["joint_limit"] = float(below + above)

        # --- Weighted sum ---
        total = 0.0
        for key, value in components.items():
            weight = self.weights.get(key, 0.0)
            total += weight * value

        return float(total), components


# ============================================================
# COST OF TRANSPORT COMPUTATION
# ============================================================
def compute_cost_of_transport(torques_history, joint_vel_history,
                               total_mass, distance_traveled, gravity=9.81):
    """
    Compute Cost of Transport (CoT).

    CoT = E_total / (m * g * d)

    Args:
        torques_history: list of np.array, torque at each timestep
        joint_vel_history: list of np.array, joint velocities at each timestep
        total_mass: float, robot total mass (kg)
        distance_traveled: float, total forward distance (m)
        gravity: float, 9.81 m/s^2

    Returns:
        cot: float (dimensionless, lower is better)
    """
    total_energy = 0.0
    dt = 0.005 * 4  # physics_dt * decimation = control dt

    for torques, joint_vels in zip(torques_history, joint_vel_history):
        # Mechanical power = sum(|torque * angular_velocity|)
        power = np.sum(np.abs(torques * joint_vels[:len(torques)]))
        total_energy += power * dt

    if distance_traveled < 0.01:
        return float("inf")

    cot = total_energy / (total_mass * gravity * distance_traveled)
    return cot


# ============================================================
# TRAINING ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train bipedal leg models")
    parser.add_argument("--model", type=str, required=True, choices=["A", "B", "C"],
                        help="Model to train: A (forward knee), B (reverse knee), C (bidirectional)")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2],
                        help="Training phase: 1 (minimal reward), 2 (full reward)")
    parser.add_argument("--num_envs", type=int, default=4096,
                        help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Max training iterations (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run without visualization (default: True)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Training device")

    args = parser.parse_args()

    # Load model-specific config
    model_configs = {
        "A": ("configs.model_a_config", "models/model_a_forward_knee.xml"),
        "B": ("configs.model_b_config", "models/model_b_reverse_knee.xml"),
        "C": ("configs.model_c_config", "models/model_c_bidirectional_knee.xml"),
    }

    config_module, model_xml = model_configs[args.model]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = {"A": "forward_knee", "B": "reverse_knee", "C": "bidirectional_knee"}[args.model]
    run_name = f"{model_name}_phase{args.phase}_{timestamp}"
    log_dir = os.path.join("results", "checkpoints", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Save run config
    run_config = {
        "model": args.model,
        "model_xml": model_xml,
        "phase": args.phase,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "device": args.device,
        "timestamp": timestamp,
    }
    with open(os.path.join(log_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"=" * 60)
    print(f"  BIPEDAL LEG PLATFORM — Training")
    print(f"  Model:      {args.model} ({model_name})")
    print(f"  Phase:      {args.phase}")
    print(f"  Envs:       {args.num_envs}")
    print(f"  Device:     {args.device}")
    print(f"  Log dir:    {log_dir}")
    print(f"=" * 60)

    # ============================================================
    # FRAMEWORK DETECTION AND TRAINING
    # ============================================================
    # Try frameworks in order of preference

    try:
        # Option 1: Isaac Lab (preferred)
        import isaaclab  # noqa
        print("[INFO] Isaac Lab detected. Using Isaac Lab training loop.")
        print("[TODO] Integrate with Isaac Lab's ManagerBasedRLEnv.")
        print("       See: isaac-sim.github.io/IsaacLab/main/source/overview/environments.html")
        print("       Use rsl_rl PPO with our reward function.")
        raise ImportError("Isaac Lab integration not yet implemented — falling through to next option")

    except ImportError:
        pass

    try:
        # Option 2: Isaac Gym (Preview) + legged_gym
        import isaacgym  # noqa
        print("[INFO] Isaac Gym detected. Using legged_gym training loop.")
        print("[TODO] Register custom environment in legged_gym/envs/")
        print("       Copy g1_config.py pattern, point to our MJCF model.")
        print("       Use rsl_rl PPO with our reward function.")
        raise ImportError("legged_gym integration not yet implemented — falling through to next option")

    except ImportError:
        pass

    try:
        # Option 3: MuJoCo + Stable-Baselines3
        import mujoco
        import stable_baselines3  # noqa
        print("[INFO] MuJoCo + Stable-Baselines3 detected.")
        print("[INFO] This will be slower than Isaac Gym but works without NVIDIA-specific dependencies.")
        _train_with_sb3(args, model_xml, log_dir)
        return

    except ImportError:
        pass

    try:
        # Option 4: MuJoCo + Gymnasium (minimal, for testing)
        import mujoco
        import gymnasium  # noqa
        print("[INFO] MuJoCo + Gymnasium detected (no SB3).")
        print("[INFO] Running model validation only. Install stable-baselines3 for training.")
        _validate_only(model_xml)
        return

    except ImportError:
        pass

    print("[ERROR] No suitable training framework found.")
    print("        Install one of: isaaclab, isaacgym, stable-baselines3")
    print("        See CLAUDE_CODE_INSTRUCTIONS.md for setup options.")


def _validate_only(model_xml):
    """Fallback: just validate the model loads and runs physics."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    print(f"[OK] Model loaded: {model.nv} DOF, {model.nu} actuators")

    mujoco.mj_resetData(model, data)
    for i in range(1000):
        mujoco.mj_step(model, data)
    print(f"[OK] 1000 physics steps completed. Torso z = {data.qpos[2]:.4f}")


def _train_with_sb3(args, model_xml, log_dir):
    """Training with Stable-Baselines3 PPO (fallback option)."""
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback

    # Note: For SB3, num_envs is limited by CPU cores (not GPU parallelism)
    num_envs = min(args.num_envs, 64)  # SB3 doesn't scale to 4096
    print(f"[WARN] SB3 mode: reducing num_envs to {num_envs}")

    # Create vectorized environment
    def make_env():
        def _init():
            env = gym.make(
                "Humanoid-v5",
                xml_file=model_xml,
                healthy_z_range=(0.3, 1.5),
                ctrl_cost_weight=0.001,  # Maps to our energy penalty
                forward_reward_weight=1.0,
                reset_noise_scale=0.02,
                frame_skip=4,
                max_episode_steps=1000,
            )
            return env
        return _init

    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    max_iters = args.max_iterations or (2000 if args.phase == 1 else 15000)
    total_timesteps = max_iters * num_envs * 32  # horizon * num_envs * iterations

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=32,
        batch_size=num_envs * 32 // 4,  # 4 minibatches
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir,
        device=args.device if "cuda" in args.device else "auto",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        ),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 15, 10000),
        save_path=log_dir,
        name_prefix=f"model_{args.model}",
    )

    print(f"[INFO] Starting SB3 PPO training: {total_timesteps} timesteps")
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    elapsed = time.time() - start_time
    print(f"[DONE] Training completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Save final model
    final_path = os.path.join(log_dir, f"model_{args.model}_final")
    model.save(final_path)
    print(f"[SAVED] {final_path}")

    env.close()


if __name__ == "__main__":
    main()

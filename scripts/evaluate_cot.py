"""
Evaluate Cost of Transport (CoT) for all trained models.

Usage:
    python evaluate_cot.py --all_models
    python evaluate_cot.py --model A --checkpoint results/checkpoints/model_a_final.pt

CoT = Total Energy / (mass * gravity * distance)
Lower is better. Typical values:
  - Human walking: ~0.05
  - Human running: ~0.08
  - Ostrich running: ~0.03-0.04 (50% less than human)
  - ASIMO: ~3.0
  - Modern humanoids: ~1.0-3.0
"""

import argparse
import json
import os

import numpy as np

try:
    import mujoco
except ImportError:
    print("[ERROR] MuJoCo not installed. Run: pip install mujoco")
    exit(1)


MODEL_CONFIGS = {
    "A": {
        "xml": "models/model_a_forward_knee.xml",
        "name": "Forward Knee (Baseline)",
    },
    "B": {
        "xml": "models/model_b_reverse_knee.xml",
        "name": "Reverse Knee (Ostrich)",
    },
    "C": {
        "xml": "models/model_c_bidirectional_knee.xml",
        "name": "Bidirectional Knee (Proposed)",
    },
}

MODEL_NAME_MAP = {
    "A": "forward_knee",
    "B": "reverse_knee",
    "C": "bidirectional_knee",
}


def _find_checkpoint(checkpoint_dir, model_key):
    """Find the latest checkpoint for a given model key."""
    import glob
    name = MODEL_NAME_MAP[model_key]
    # Look for directories matching the model name
    pattern = os.path.join(checkpoint_dir, f"{name}_*", "ckpt_final.npz")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]  # Latest
    # Also try direct path
    direct = os.path.join(checkpoint_dir, f"{name}_ckpt_final.npz")
    if os.path.exists(direct):
        return direct
    return None


def evaluate_cot(model_xml, policy=None, duration_s=30.0,
                 target_vel=1.0, num_episodes=10, payload_mass=0.0):
    """
    Run evaluation episodes and compute Cost of Transport.

    Args:
        model_xml: path to MJCF file
        policy: trained policy (callable: obs -> action). If None, use zero actions (passive test).
        duration_s: episode duration in seconds
        target_vel: target forward velocity (m/s)
        num_episodes: number of episodes to average over
        payload_mass: additional mass on torso (kg)

    Returns:
        dict with CoT statistics
    """
    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)

    dt = model.opt.timestep
    decimation = 2  # match training env
    policy_dt = dt * decimation
    total_policy_steps = int(duration_s / policy_dt)
    total_mass = sum(model.body_mass) + payload_mass

    nq_free = 7
    nu = model.nu

    # Get default joint positions
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.85
    mujoco.mj_forward(model, data)
    default_joint_pos = data.qpos[nq_free:].copy()

    # Add payload mass to torso if specified
    if payload_mass > 0:
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        model.body_mass[torso_id] += payload_mass

    cots = []
    distances = []
    avg_speeds = []
    total_energies = []

    for ep in range(num_episodes):
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.85
        data.qpos[2] += np.random.uniform(-0.01, 0.01)
        mujoco.mj_forward(model, data)

        energy_sum = 0.0
        initial_x = data.qpos[0]
        prev_actions = np.zeros(nu, dtype=np.float32)

        for step in range(total_policy_steps):
            # Get action from policy (or zero for passive test)
            if policy is not None:
                obs = _get_obs(model, data, default_joint_pos, nu, prev_actions)
                action = policy(obs)
                data.ctrl[:] = action
                prev_actions = action.copy()
            else:
                data.ctrl[:] = 0.0

            # Step physics with decimation
            for _ in range(decimation):
                mujoco.mj_step(model, data)

            # Accumulate energy: sum(|torque * velocity|) * dt
            torques = data.actuator_force.copy()
            joint_vels = data.qvel[6:]
            power = np.sum(np.abs(torques * joint_vels[:len(torques)]))
            energy_sum += power * policy_dt

            # Check termination
            torso_z = data.qpos[2]
            if torso_z < 0.3:
                break

        # Compute metrics
        final_x = data.qpos[0]
        distance = final_x - initial_x
        elapsed = step * policy_dt

        if distance > 0.1:  # Only count if robot actually moved
            cot = energy_sum / (total_mass * 9.81 * distance)
            cots.append(cot)

        distances.append(distance)
        avg_speeds.append(distance / max(elapsed, 0.01))
        total_energies.append(energy_sum)

    results = {
        "cot_mean": float(np.mean(cots)) if cots else float("inf"),
        "cot_std": float(np.std(cots)) if cots else 0.0,
        "cot_min": float(np.min(cots)) if cots else float("inf"),
        "cot_max": float(np.max(cots)) if cots else float("inf"),
        "distance_mean": float(np.mean(distances)),
        "speed_mean": float(np.mean(avg_speeds)),
        "energy_mean": float(np.mean(total_energies)),
        "success_rate": len(cots) / num_episodes,
        "total_mass": total_mass,
        "payload_mass": payload_mass,
        "num_episodes": num_episodes,
        "num_actuators": model.nu,
    }

    return results


def _get_obs(model, data, default_joint_pos, nu, prev_actions):
    """Extract observation matching BipedalWalkEnv/BipedalMJXEnv format."""
    from load_mjx_policy import get_obs_from_mujoco
    obs = get_obs_from_mujoco(model, data, default_joint_pos, nu)
    # Replace zero prev_actions with actual
    obs[-nu:] = prev_actions
    return obs


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cost of Transport")
    parser.add_argument("--all_models", action="store_true", help="Evaluate all 3 models")
    parser.add_argument("--model", type=str, choices=["A", "B", "C"], help="Single model to evaluate")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Dir with ckpt_final.npz for each model")
    parser.add_argument("--duration", type=float, default=30.0, help="Episode duration (seconds)")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--payload", type=float, default=0.0, help="Payload mass (kg)")
    parser.add_argument("--output", type=str, default="results/metrics/cot_comparison.json")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    models_to_eval = list(MODEL_CONFIGS.keys()) if args.all_models else [args.model]

    all_results = {}

    for model_key in models_to_eval:
        config = MODEL_CONFIGS[model_key]
        print(f"\n{'='*60}")
        print(f"  Evaluating Model {model_key}: {config['name']}")
        print(f"  Payload: {args.payload}kg")
        print(f"{'='*60}")

        policy = None
        if args.checkpoint_dir:
            from load_mjx_policy import load_mjx_policy
            # Find checkpoint for this model
            ckpt = _find_checkpoint(args.checkpoint_dir, model_key)
            if ckpt:
                m = mujoco.MjModel.from_xml_path(config["xml"])
                obs_dim = 3 + 3 + 3 + (m.nq - 7) + (m.nv - 6) + m.nu
                policy = load_mjx_policy(ckpt, obs_dim, m.nu)
                print(f"  Loaded policy: {ckpt}")
            else:
                print(f"  [WARN] No checkpoint found for model {model_key}")

        results = evaluate_cot(
            model_xml=config["xml"],
            policy=policy,
            duration_s=args.duration,
            num_episodes=args.num_episodes,
            payload_mass=args.payload,
        )

        all_results[f"Model_{model_key}"] = {
            "name": config["name"],
            **results,
        }

        print(f"  CoT:          {results['cot_mean']:.4f} ± {results['cot_std']:.4f}")
        print(f"  Distance:     {results['distance_mean']:.2f} m")
        print(f"  Avg Speed:    {results['speed_mean']:.2f} m/s")
        print(f"  Energy:       {results['energy_mean']:.2f} J")
        print(f"  Success Rate: {results['success_rate']*100:.0f}%")
        print(f"  Actuators:    {results['num_actuators']}")

    # Comparison summary
    if len(all_results) > 1 and "Model_A" in all_results:
        baseline_cot = all_results["Model_A"]["cot_mean"]
        print(f"\n{'='*60}")
        print(f"  COMPARISON (relative to Model A baseline)")
        print(f"{'='*60}")
        for key, res in all_results.items():
            if baseline_cot > 0 and baseline_cot != float("inf"):
                relative = res["cot_mean"] / baseline_cot
                print(f"  {key}: CoT = {res['cot_mean']:.4f} ({relative:.1%} of baseline)")
            else:
                print(f"  {key}: CoT = {res['cot_mean']:.4f}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {args.output}")


if __name__ == "__main__":
    main()

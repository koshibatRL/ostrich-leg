"""
Evaluate payload tolerance and external disturbance robustness.

Usage:
    python evaluate_payload_robustness.py --all_models --checkpoint_dir results/checkpoints
"""

import argparse
import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    import mujoco
except ImportError:
    print("[ERROR] MuJoCo not installed.")
    exit(1)

from load_mjx_policy import load_mjx_policy, get_obs_from_mujoco


MODEL_CONFIGS = {
    "A": {"xml": "models/model_a_forward_knee.xml", "name": "Forward Knee (Baseline)"},
    "B": {"xml": "models/model_b_reverse_knee.xml", "name": "Reverse Knee (Ostrich)"},
    "C": {"xml": "models/model_c_bidirectional_knee.xml", "name": "Bidirectional Knee (Proposed)"},
}

MODEL_NAME_MAP = {
    "A": "forward_knee",
    "B": "reverse_knee",
    "C": "bidirectional_knee",
}

DECIMATION = 2  # must match training


def _find_checkpoint(checkpoint_dir, model_key):
    name = MODEL_NAME_MAP[model_key]
    pattern = os.path.join(checkpoint_dir, f"{name}_*", "ckpt_final.npz")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _make_obs(model, data, default_joint_pos, nu, prev_actions):
    obs = get_obs_from_mujoco(model, data, default_joint_pos, nu)
    obs[-nu:] = prev_actions
    return obs


def evaluate_payload(model_xml, policy=None, masses=[0, 5, 10, 20],
                     duration_s=20.0, num_trials=20):
    results = {}

    for mass in masses:
        model = mujoco.MjModel.from_xml_path(model_xml)
        data = mujoco.MjData(model)
        nu = model.nu
        nq_free = 7

        # Default joint pos
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.85
        mujoco.mj_forward(model, data)
        default_joint_pos = data.qpos[nq_free:].copy()

        if mass > 0:
            torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            model.body_mass[torso_id] += mass

        total_mass = sum(model.body_mass)
        policy_dt = model.opt.timestep * DECIMATION
        total_policy_steps = int(duration_s / policy_dt)

        successes = 0
        distances = []
        cots = []

        for trial in range(num_trials):
            mujoco.mj_resetData(model, data)
            data.qpos[2] = 0.85
            data.qpos[:2] += np.random.uniform(-0.01, 0.01, 2)
            mujoco.mj_forward(model, data)

            energy_sum = 0.0
            initial_x = data.qpos[0]
            survived = True
            prev_actions = np.zeros(nu, dtype=np.float32)

            for step in range(total_policy_steps):
                if policy is not None:
                    obs = _make_obs(model, data, default_joint_pos, nu, prev_actions)
                    action = policy(obs)
                    data.ctrl[:] = action
                    prev_actions = action.copy()
                else:
                    data.ctrl[:] = 0.0

                for _ in range(DECIMATION):
                    mujoco.mj_step(model, data)

                torques = data.actuator_force
                joint_vels = data.qvel[6:]
                power = np.sum(np.abs(torques * joint_vels[:len(torques)]))
                energy_sum += power * policy_dt

                if data.qpos[2] < 0.3:
                    survived = False
                    break

            distance = data.qpos[0] - initial_x
            distances.append(distance)

            if survived and distance > 0.5:
                successes += 1
                cot = energy_sum / (total_mass * 9.81 * max(distance, 0.01))
                cots.append(cot)

        results[f"{mass}kg"] = {
            "payload_mass": mass,
            "total_mass": float(sum(model.body_mass)),
            "success_rate": successes / num_trials,
            "distance_mean": float(np.mean(distances)),
            "distance_std": float(np.std(distances)),
            "cot_mean": float(np.mean(cots)) if cots else float("inf"),
            "cot_std": float(np.std(cots)) if cots else 0.0,
            "num_trials": num_trials,
        }

    return results


def evaluate_robustness(model_xml, policy=None, forces=[50, 100, 150],
                        push_duration=0.1, num_trials=30, push_time=3.0,
                        episode_duration=10.0):
    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    nu = model.nu
    nq_free = 7

    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.85
    mujoco.mj_forward(model, data)
    default_joint_pos = data.qpos[nq_free:].copy()

    policy_dt = model.opt.timestep * DECIMATION
    results = {}

    for force in forces:
        recoveries = 0
        recovery_times = []

        push_start_step = int(push_time / policy_dt)
        push_end_step = int((push_time + push_duration) / policy_dt)
        total_policy_steps = int(episode_duration / policy_dt)

        for trial in range(num_trials):
            mujoco.mj_resetData(model, data)
            data.qpos[2] = 0.85
            mujoco.mj_forward(model, data)

            post_push_recovered = False
            fell = False
            prev_actions = np.zeros(nu, dtype=np.float32)

            for step in range(total_policy_steps):
                if policy is not None:
                    obs = _make_obs(model, data, default_joint_pos, nu, prev_actions)
                    action = policy(obs)
                    data.ctrl[:] = action
                    prev_actions = action.copy()
                else:
                    data.ctrl[:] = 0.0

                # Apply lateral push
                torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
                if push_start_step <= step <= push_end_step:
                    push_dir = 1.0 if trial % 2 == 0 else -1.0
                    data.xfrc_applied[torso_id, 1] = force * push_dir
                else:
                    data.xfrc_applied[torso_id, :] = 0.0

                for _ in range(DECIMATION):
                    mujoco.mj_step(model, data)

                if data.qpos[2] < 0.3:
                    fell = True
                    break

                if step > push_end_step and not post_push_recovered:
                    quat_w = data.qpos[3]
                    if abs(quat_w) > 0.95:
                        post_push_recovered = True
                        recovery_time = (step - push_end_step) * policy_dt
                        recovery_times.append(recovery_time)

            if not fell and post_push_recovered:
                recoveries += 1

        results[f"{force}N"] = {
            "force_N": force,
            "push_duration_s": push_duration,
            "recovery_rate": recoveries / num_trials,
            "recovery_time_mean": float(np.mean(recovery_times)) if recovery_times else float("inf"),
            "recovery_time_std": float(np.std(recovery_times)) if recovery_times else 0.0,
            "num_trials": num_trials,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate payload and robustness")
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--model", type=str, choices=["A", "B", "C"])
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    parser.add_argument("--test", type=str, default="both", choices=["payload", "robustness", "both"])
    parser.add_argument("--masses", nargs="+", type=float, default=[0, 5, 10, 20])
    parser.add_argument("--forces", nargs="+", type=float, default=[50, 100, 150])
    parser.add_argument("--output_dir", type=str, default="results/metrics")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    models_to_eval = list(MODEL_CONFIGS.keys()) if args.all_models else [args.model]

    all_results = {}

    for model_key in models_to_eval:
        config = MODEL_CONFIGS[model_key]
        print(f"\n{'='*60}")
        print(f"  Model {model_key}: {config['name']}")
        print(f"{'='*60}")

        # Load policy
        policy = None
        ckpt = _find_checkpoint(args.checkpoint_dir, model_key)
        if ckpt:
            m = mujoco.MjModel.from_xml_path(config["xml"])
            obs_dim = 3 + 3 + 3 + (m.nq - 7) + (m.nv - 6) + m.nu
            policy = load_mjx_policy(ckpt, obs_dim, m.nu)
            print(f"  Loaded: {ckpt}")
        else:
            print(f"  [WARN] No checkpoint for model {model_key}")

        model_results = {}

        if args.test in ["payload", "both"]:
            print(f"\n  --- Payload Test (masses: {args.masses}) ---")
            payload_results = evaluate_payload(
                config["xml"], policy=policy, masses=args.masses
            )
            model_results["payload"] = payload_results

            for mass_key, res in payload_results.items():
                print(f"    {mass_key}: success={res['success_rate']*100:.0f}%, "
                      f"dist={res['distance_mean']:.1f}m, CoT={res['cot_mean']:.3f}")

        if args.test in ["robustness", "both"]:
            print(f"\n  --- Robustness Test (forces: {args.forces}) ---")
            robust_results = evaluate_robustness(
                config["xml"], policy=policy, forces=args.forces
            )
            model_results["robustness"] = robust_results

            for force_key, res in robust_results.items():
                print(f"    {force_key}: recovery={res['recovery_rate']*100:.0f}%, "
                      f"time={res['recovery_time_mean']:.2f}s")

        all_results[f"Model_{model_key}"] = {
            "name": config["name"],
            **model_results,
        }

    # Save
    output_path = os.path.join(args.output_dir, "payload_robustness_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {output_path}")


if __name__ == "__main__":
    main()

"""
Render walking videos from MJX-trained checkpoints.

Usage:
    python scripts/render_mjx_policy.py --checkpoint_dir results/checkpoints
    python scripts/render_mjx_policy.py --checkpoint results/checkpoints/forward_knee_.../ckpt_final.npz --model A
"""
import os
os.environ["MUJOCO_GL"] = "osmesa"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import numpy as np
import mujoco
import imageio

from load_mjx_policy import load_mjx_policy, get_obs_from_mujoco


MODEL_CONFIGS = {
    "A": "models/model_a_forward_knee.xml",
    "B": "models/model_b_reverse_knee.xml",
    "C": "models/model_c_bidirectional_knee.xml",
}

MODEL_NAME_MAP = {
    "A": "forward_knee",
    "B": "reverse_knee",
    "C": "bidirectional_knee",
}


def render_video(model_xml, policy, output_path, duration_s=10.0, fps=30):
    """Render video using trained MJX policy."""
    m = mujoco.MjModel.from_xml_path(model_xml)
    d = mujoco.MjData(m)

    nq_free = 7
    nu = m.nu
    decimation = 4

    # Default state
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.85
    mujoco.mj_forward(m, d)
    default_joint_pos = d.qpos[nq_free:].copy()

    # Renderer
    renderer = mujoco.Renderer(m, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.elevation = -20.0
    camera.azimuth = 90.0
    camera.lookat[:] = [0, 0, 0.6]

    # Reset
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.85
    mujoco.mj_forward(m, d)

    frames = []
    total_frames = int(duration_s * fps)
    policy_dt = m.opt.timestep * decimation
    steps_per_frame = max(1, int(1.0 / (fps * policy_dt)))
    prev_actions = np.zeros(nu, dtype=np.float32)

    for frame_idx in range(total_frames):
        for _ in range(steps_per_frame):
            obs = get_obs_from_mujoco(m, d, default_joint_pos, nu)
            obs[-nu:] = prev_actions  # fill prev_actions

            action = policy(obs)
            action = np.clip(action, -1, 1)
            d.ctrl[:] = action
            prev_actions = action.copy()

            for _ in range(decimation):
                mujoco.mj_step(m, d)

            if d.qpos[2] < 0.2:
                break

        camera.lookat[0] = d.qpos[0]
        camera.lookat[2] = 0.6
        renderer.update_scene(d, camera)
        frame = renderer.render()
        frames.append(frame.copy())

        if d.qpos[2] < 0.2:
            break

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                 quality=8, pixelformat="yuv420p")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    renderer.close()

    print(f"    {output_path}: {len(frames)} frames ({len(frames)/fps:.1f}s)")
    sys.stdout.flush()


def find_checkpoint(checkpoint_dir, model_key):
    """Find latest checkpoint for a model."""
    name = MODEL_NAME_MAP[model_key]
    pattern = os.path.join(checkpoint_dir, f"{name}_*", "ckpt_final.npz")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Render MJX policy videos")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Direct path to .npz")
    parser.add_argument("--model", type=str, default=None, choices=["A", "B", "C"])
    parser.add_argument("--output_dir", type=str, default="results/videos")
    parser.add_argument("--duration", type=float, default=10.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.checkpoint and args.model:
        # Single model
        models = [(args.model, args.checkpoint)]
    else:
        # All models
        models = []
        for key in ["A", "B", "C"]:
            ckpt = find_checkpoint(args.checkpoint_dir, key)
            if ckpt:
                models.append((key, ckpt))
            else:
                print(f"  [SKIP] No checkpoint for Model {key}")

    for model_key, ckpt_path in models:
        xml = MODEL_CONFIGS[model_key]
        m = mujoco.MjModel.from_xml_path(xml)
        obs_dim = 3 + 3 + 3 + (m.nq - 7) + (m.nv - 6) + m.nu

        print(f"  Rendering Model {model_key} from {ckpt_path}...")
        sys.stdout.flush()

        policy = load_mjx_policy(ckpt_path, obs_dim, m.nu)
        output = os.path.join(args.output_dir, f"model_{model_key}_mjx.mp4")
        render_video(xml, policy, output, duration_s=args.duration)

    print(f"\n  Done. Videos in {args.output_dir}/")


if __name__ == "__main__":
    main()

"""
Render walking videos for trained policies.

Usage:
    python render_video.py --model A --checkpoint results/checkpoints/.../model_A_final.zip
    python render_video.py --model B --checkpoint ... --payload 20
    python render_video.py --all_models --checkpoint_dir results/checkpoints/

Outputs MP4 video files to results/videos/

Dependencies: mujoco, imageio, imageio-ffmpeg
"""

import argparse
import os
import json

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("[ERROR] MuJoCo not installed. Run: pip install mujoco")
    exit(1)

try:
    import imageio
except ImportError:
    print("[ERROR] imageio not installed. Run: pip install imageio imageio-ffmpeg")
    exit(1)


MODEL_CONFIGS = {
    "A": {"xml": "models/model_a_forward_knee.xml", "name": "Forward Knee (Baseline)", "color": "gray"},
    "B": {"xml": "models/model_b_reverse_knee.xml", "name": "Reverse Knee (Ostrich)", "color": "orange"},
    "C": {"xml": "models/model_c_bidirectional_knee.xml", "name": "Bidirectional Knee (Proposed)", "color": "cyan"},
}


def load_policy(checkpoint_path):
    """
    Load a trained policy from checkpoint.
    Supports: PyTorch (.pt), Stable-Baselines3 (.zip), ONNX (.onnx)

    Returns: callable policy(obs) -> action
    """
    if checkpoint_path is None:
        return None

    ext = os.path.splitext(checkpoint_path)[1]

    if ext == ".zip":
        # Stable-Baselines3
        from stable_baselines3 import PPO
        model = PPO.load(checkpoint_path)
        def policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action
        return policy

    elif ext == ".pt":
        # PyTorch (rsl_rl / legged_gym format)
        import torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # rsl_rl saves policy as 'model_state_dict' or 'actor'
        # This needs to be adapted based on actual checkpoint structure
        print(f"[INFO] Loaded PyTorch checkpoint. Keys: {list(checkpoint.keys())}")
        print("[WARN] PyTorch policy loading needs adaptation to actual checkpoint format.")
        print("       For now, using zero-action fallback.")
        return None

    elif ext == ".onnx":
        # ONNX Runtime
        import onnxruntime as ort
        session = ort.InferenceSession(checkpoint_path)
        input_name = session.get_inputs()[0].name
        def policy(obs):
            result = session.run(None, {input_name: obs.reshape(1, -1).astype(np.float32)})
            return result[0].flatten()
        return policy

    else:
        print(f"[WARN] Unknown checkpoint format: {ext}. Using zero-action fallback.")
        return None


def render_video(model_xml, output_path, policy=None, duration_s=10.0,
                 fps=30, width=1280, height=720, payload_mass=0.0,
                 camera_distance=3.0, camera_elevation=-20.0,
                 camera_tracking=True, push_at=None, push_force=0.0):
    """
    Render a video of the robot walking.

    Args:
        model_xml: path to MJCF file
        output_path: path for output MP4
        policy: trained policy callable, or None for passive demo
        duration_s: video duration in seconds
        fps: video frame rate
        width, height: video resolution
        payload_mass: additional mass on torso (kg)
        camera_distance: camera distance from robot
        camera_elevation: camera elevation angle (degrees)
        camera_tracking: if True, camera follows the robot
        push_at: time (s) to apply a push, or None
        push_force: force of push (N), lateral direction
    """
    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)

    # Add payload
    if payload_mass > 0:
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        model.body_mass[torso_id] += payload_mass

    # Setup renderer
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Camera setup
    camera = mujoco.MjvCamera()
    camera.distance = camera_distance
    camera.elevation = camera_elevation
    camera.azimuth = 90  # Side view
    camera.lookat[:] = [0, 0, 0.8]

    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.85  # Initial height

    dt = model.opt.timestep
    steps_per_frame = max(1, int(1.0 / (fps * dt)))
    total_frames = int(duration_s * fps)

    frames = []

    print(f"  Rendering {total_frames} frames ({duration_s}s at {fps}fps)...")

    for frame_idx in range(total_frames):
        # Step physics (multiple steps per frame for smooth simulation)
        for _ in range(steps_per_frame):
            if policy is not None:
                obs = _get_obs(model, data)
                action = policy(obs)
                data.ctrl[:len(action)] = action
            else:
                data.ctrl[:] = 0.0

            # Apply push if specified
            current_time = frame_idx / fps
            if push_at is not None and abs(current_time - push_at) < 0.1:
                torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
                data.xfrc_applied[torso_id, 1] = push_force  # Lateral push
            else:
                data.xfrc_applied[:] = 0.0

            mujoco.mj_step(model, data)

            # Check termination
            if data.qpos[2] < 0.2:
                break

        # Update camera to track robot
        if camera_tracking:
            camera.lookat[0] = data.qpos[0]  # Follow x position
            camera.lookat[2] = 0.6

        # Render frame
        renderer.update_scene(data, camera)
        frame = renderer.render()
        frames.append(frame.copy())

        if data.qpos[2] < 0.2:
            print(f"  Robot fell at frame {frame_idx} ({frame_idx/fps:.1f}s)")
            break

    # Save video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                 quality=8, pixelformat="yuv420p")
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"  Saved: {output_path} ({len(frames)} frames)")
    renderer.close()

    return len(frames)


def _get_obs(model, data):
    """Extract observation vector."""
    obs = np.concatenate([
        data.qpos[7:].copy(),
        data.qvel[6:].copy(),
        data.sensordata.copy(),
    ])
    return obs


def main():
    parser = argparse.ArgumentParser(description="Render walking videos")
    parser.add_argument("--model", type=str, choices=["A", "B", "C"])
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/videos")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--payload", type=float, default=0.0)
    parser.add_argument("--push", action="store_true", help="Add a push at t=5s")
    parser.add_argument("--push_force", type=float, default=100.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    models_to_render = list(MODEL_CONFIGS.keys()) if args.all_models else [args.model]

    for model_key in models_to_render:
        config = MODEL_CONFIGS[model_key]
        print(f"\n{'='*60}")
        print(f"  Rendering Model {model_key}: {config['name']}")
        if args.payload > 0:
            print(f"  Payload: {args.payload}kg")
        if args.push:
            print(f"  Push: {args.push_force}N at t=5s")
        print(f"{'='*60}")

        # Find checkpoint
        checkpoint = args.checkpoint
        if checkpoint is None and args.checkpoint_dir:
            # Auto-find latest checkpoint for this model
            pattern = f"model_{model_key}"
            for fname in sorted(os.listdir(args.checkpoint_dir), reverse=True):
                if pattern.lower() in fname.lower():
                    checkpoint = os.path.join(args.checkpoint_dir, fname)
                    break

        policy = load_policy(checkpoint)
        if policy is None:
            print("  [INFO] No policy loaded. Rendering passive dynamics (zero-action).")

        # Build output filename
        suffix_parts = []
        if args.payload > 0:
            suffix_parts.append(f"payload{int(args.payload)}kg")
        if args.push:
            suffix_parts.append(f"push{int(args.push_force)}N")
        suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

        output_path = os.path.join(
            args.output_dir,
            f"model_{model_key}_{config['name'].split('(')[0].strip().lower().replace(' ', '_')}{suffix}.mp4"
        )

        render_video(
            model_xml=config["xml"],
            output_path=output_path,
            policy=policy,
            duration_s=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            payload_mass=args.payload,
            push_at=5.0 if args.push else None,
            push_force=args.push_force if args.push else 0.0,
        )

    print(f"\n[DONE] All videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

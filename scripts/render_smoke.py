"""Render videos from saved smoke test checkpoints using the same env as training."""
import os
os.environ["MUJOCO_GL"] = "osmesa"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco
import imageio
from stable_baselines3 import PPO
from bipedal_env import BipedalWalkEnv


MODELS = {
    "A": "models/model_a_forward_knee.xml",
    "B": "models/model_b_reverse_knee.xml",
    "C": "models/model_c_bidirectional_knee.xml",
}


def render_video(model_xml, policy_path, output_path, duration_s=8.0, fps=30):
    # Use the actual training env to get correct observations
    env = BipedalWalkEnv(model_xml=model_xml, phase=1, episode_length_s=duration_s)
    ppo = PPO.load(policy_path, device="cpu")

    m = env.mj_model
    renderer = mujoco.Renderer(m, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.elevation = -20.0
    camera.azimuth = 90.0
    camera.lookat[:] = [0, 0, 0.6]

    obs, _ = env.reset()
    frames = []
    total_frames = int(duration_s * fps)
    policy_dt = env.dt * env.decimation
    steps_per_frame = max(1, int(1.0 / (fps * policy_dt)))

    for frame_idx in range(total_frames):
        for _ in range(steps_per_frame):
            action, _ = ppo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

        camera.lookat[0] = env.mj_data.qpos[0]
        camera.lookat[2] = 0.6
        renderer.update_scene(env.mj_data, camera)
        frame = renderer.render()
        frames.append(frame.copy())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                 quality=8, pixelformat="yuv420p")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    renderer.close()
    env.close()

    print(f"    {output_path}: {len(frames)} frames ({duration_s}s)")
    sys.stdout.flush()


def main():
    os.makedirs("results/videos", exist_ok=True)

    for key, xml in MODELS.items():
        ckpt = f"results/checkpoints/model_{key}_smoke.zip"
        if not os.path.exists(ckpt):
            print(f"  [SKIP] {ckpt} not found")
            continue

        print(f"  Rendering Model {key}...")
        sys.stdout.flush()
        render_video(xml, ckpt, f"results/videos/model_{key}_smoke.mp4")

    print("\n  Done. Videos in results/videos/")


if __name__ == "__main__":
    main()

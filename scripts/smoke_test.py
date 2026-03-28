"""
Smoke test: train all 3 models briefly, evaluate, and render videos.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from bipedal_env import BipedalWalkEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import time
import json


def make_env(model_xml, phase, rank=0):
    def _init():
        env = BipedalWalkEnv(
            model_xml=model_xml,
            phase=phase,
            episode_length_s=5.0,
        )
        env = Monitor(env)
        return env
    return _init


def evaluate(model, env, n_episodes=10):
    rewards = []
    lengths = []
    obs = env.reset()
    ep_reward = np.zeros(env.num_envs)
    ep_len = np.zeros(env.num_envs)
    while len(rewards) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        ep_len += 1
        for i, d in enumerate(done):
            if d:
                rewards.append(ep_reward[i])
                lengths.append(ep_len[i])
                ep_reward[i] = 0
                ep_len[i] = 0
    return np.mean(rewards), np.mean(lengths)


def render_video(model_xml, policy_path, output_path, duration_s=8.0, fps=30):
    """Render a video using trained policy (headless via OSMesa)."""
    os.environ["MUJOCO_GL"] = "osmesa"
    import mujoco
    import imageio

    m = mujoco.MjModel.from_xml_path(model_xml)
    d = mujoco.MjData(m)

    # Load policy
    ppo = PPO.load(policy_path, device="cpu")

    renderer = mujoco.Renderer(m, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.elevation = -20.0
    camera.azimuth = 90.0
    camera.lookat[:] = [0, 0, 0.6]

    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.85

    dt = m.opt.timestep
    decimation = 4
    steps_per_frame = max(1, int(1.0 / (fps * dt)))

    nq_free = 7
    nv_free = 6
    nu = m.nu
    nq_joints = m.nq - nq_free
    default_joint_pos = d.qpos[nq_free:].copy()
    prev_actions = np.zeros(nu, dtype=np.float32)

    frames = []
    total_frames = int(duration_s * fps)

    for frame_idx in range(total_frames):
        for _ in range(steps_per_frame):
            # Build obs matching BipedalWalkEnv
            quat = d.qpos[3:7]
            angvel = d.qvel[3:6].copy()

            # Projected gravity
            w, x, y, z = quat
            proj_gravity = np.array([
                2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)
            ])
            # Rotate inverse for gravity
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
            joint_pos = d.qpos[nq_free:] - default_joint_pos
            joint_vel = d.qvel[nv_free:]

            obs = np.concatenate([
                angvel, proj_g, cmd, joint_pos, joint_vel, prev_actions
            ]).astype(np.float32)

            action, _ = ppo.predict(obs, deterministic=True)
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

    print(f"    Video saved: {output_path} ({len(frames)} frames)")


def main():
    models_to_test = {
        "A": "models/model_a_forward_knee.xml",
        "B": "models/model_b_reverse_knee.xml",
        "C": "models/model_c_bidirectional_knee.xml",
    }

    os.makedirs("results/videos", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)

    all_results = {}

    for model_key, xml_path in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"  MODEL {model_key}: {xml_path}")
        print(f"{'='*60}")
        sys.stdout.flush()

        n_envs = 8
        env = SubprocVecEnv([make_env(xml_path, phase=1, rank=i) for i in range(n_envs)])

        ppo = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=64, batch_size=128,
            n_epochs=5, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, max_grad_norm=1.0,
            verbose=0, device="cpu",
            policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
        )

        # Evaluate before training
        r_before, l_before = evaluate(ppo, env, n_episodes=10)
        print(f"  Before training:  reward={r_before:.2f}, ep_len={l_before:.1f}")
        sys.stdout.flush()

        # Train
        train_steps = 50000
        t0 = time.time()
        ppo.learn(total_timesteps=train_steps, progress_bar=False)
        elapsed = time.time() - t0
        print(f"  Trained {train_steps} steps in {elapsed:.1f}s ({train_steps/elapsed:.0f} sps)")
        sys.stdout.flush()

        # Evaluate after training
        r_after, l_after = evaluate(ppo, env, n_episodes=10)
        print(f"  After training:   reward={r_after:.2f}, ep_len={l_after:.1f}")

        improved = r_after > r_before
        print(f"  Reward improved:  {'YES' if improved else 'NO'} ({r_before:.2f} -> {r_after:.2f})")
        sys.stdout.flush()

        # Save model
        save_path = f"results/checkpoints/model_{model_key}_smoke"
        ppo.save(save_path)
        print(f"  Saved: {save_path}.zip")

        env.close()

        # Render video
        print(f"  Rendering video...")
        sys.stdout.flush()
        video_path = f"results/videos/model_{model_key}_smoke.mp4"
        render_video(xml_path, f"{save_path}.zip", video_path, duration_s=8.0)

        all_results[model_key] = {
            "reward_before": float(r_before),
            "reward_after": float(r_after),
            "ep_len_before": float(l_before),
            "ep_len_after": float(l_after),
            "improved": bool(improved),
            "train_steps": train_steps,
            "train_time_s": elapsed,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<10} {'Before':>10} {'After':>10} {'Change':>10} {'EpLen':>10}")
    for k, v in all_results.items():
        change = v['reward_after'] - v['reward_before']
        print(f"  {k:<10} {v['reward_before']:>10.2f} {v['reward_after']:>10.2f} {change:>+10.2f} {v['ep_len_after']:>10.1f}")

    with open("results/metrics/smoke_test_results.json", "w") as f:
        os.makedirs("results/metrics", exist_ok=True)
        json.dump(all_results, f, indent=2)

    print(f"\n  Videos: results/videos/model_{{A,B,C}}_smoke.mp4")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

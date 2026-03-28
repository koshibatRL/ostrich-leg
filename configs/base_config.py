"""
Base configuration shared by all 3 bipedal leg models.
Contains: reward function, PPO hyperparameters, domain randomization, termination conditions.

This config follows the legged_gym / rsl_rl convention.
Model-specific overrides are in model_a/b/c_config.py.

References for reward weight choices:
- legged_gym ANYmal config (Rudin et al., 2022)
- unitree_rl_gym G1 config
- Berkeley humanoid locomotion (Radosavovic et al., 2024)
"""

import numpy as np


# ============================================================
# ENVIRONMENT CONFIG
# ============================================================
class BaseEnvCfg:
    """Environment parameters shared across all models."""

    # --- Simulation ---
    dt = 0.005                  # Physics timestep (s). 0.005 = 200Hz
    decimation = 4              # Policy runs at 200/4 = 50Hz
    num_envs = 4096             # Number of parallel environments
    episode_length_s = 20.0     # Max episode length (seconds)

    # --- Terrain ---
    # Start with flat terrain. Add rough terrain after basic walking works.
    terrain_type = "plane"      # "plane", "rough", "stairs"
    terrain_friction = 1.0
    terrain_restitution = 0.0

    # --- Commands ---
    target_forward_vel = 1.0    # m/s (target walking speed)
    target_lateral_vel = 0.0    # m/s
    target_yaw_rate = 0.0       # rad/s

    # --- Initialization ---
    init_height = 0.85          # Initial torso height (m)
    init_noise_pos = 0.02       # Position noise at reset (m)
    init_noise_vel = 0.1        # Velocity noise at reset (m/s)
    init_noise_joint = 0.1      # Joint position noise at reset (rad)

    # --- Termination ---
    termination_height = 0.3    # Terminate if torso z < this (m)
    termination_tilt = 1.2      # Terminate if torso tilt > this (rad, ~69°)

    # --- Observation ---
    # Proprioceptive observations (no vision):
    #   - base angular velocity (3)
    #   - projected gravity (3)
    #   - commands: vx, vy, yaw_rate (3)
    #   - joint positions (nq - 7, minus freejoint)
    #   - joint velocities (nv - 6, minus freejoint)
    #   - previous actions (nu)
    # Exact dim depends on model (computed at runtime)
    observe_heights = False     # No heightmap for flat terrain


# ============================================================
# REWARD CONFIG
# ============================================================
class BaseRewardCfg:
    """
    Reward function design.

    Design principles:
    1. Same reward function for all 3 models (fair comparison)
    2. Energy penalty included (to surface CoT differences)
    3. Phased training: start simple, add complexity

    Phase 1 (quick sanity check):
      Only forward_vel + alive. Get the robot walking first.

    Phase 2 (full training):
      All rewards active. This is the comparison run.
    """

    class weights:
        # --- Positive rewards ---
        forward_vel = 1.0           # Track target forward velocity
        alive = 0.5                 # Stay alive bonus per step
        torso_upright = 0.5         # Keep torso level

        # --- Negative rewards (penalties) ---
        energy = -0.001             # Actuator energy consumption
        joint_acc = -2.5e-7         # Joint acceleration smoothness
        torso_height_var = -1.0     # Torso height deviation from target
        action_rate = -0.01         # Action smoothness (jitter penalty)
        foot_slip = -0.5            # Foot sliding on ground during contact

        # --- Additional penalties ---
        joint_limit = -10.0         # Approaching joint limits
        collision = -1.0            # Self-collision penalty

    class scales:
        """Multiplier for phase-based training."""

        # Phase 1: minimal rewards to get walking
        phase1 = {
            "forward_vel": 1.0,
            "alive": 0.5,
            "torso_upright": 0.3,
            "energy": 0.0,           # OFF in phase 1
            "joint_acc": 0.0,        # OFF in phase 1
            "torso_height_var": -0.5,
            "action_rate": -0.005,
            "foot_slip": 0.0,        # OFF in phase 1
            "joint_limit": -5.0,
            "collision": -1.0,
        }

        # Phase 2: full reward for comparison
        phase2 = {
            "forward_vel": 1.0,
            "alive": 0.5,
            "torso_upright": 0.5,
            "energy": -0.001,
            "joint_acc": -2.5e-7,
            "torso_height_var": -1.0,
            "action_rate": -0.01,
            "foot_slip": -0.5,
            "joint_limit": -10.0,
            "collision": -1.0,
        }

    @staticmethod
    def compute_forward_vel_reward(base_vel_x, target_vel):
        """Exponential tracking reward. Max=1.0 when vel matches target."""
        error = base_vel_x - target_vel
        return np.exp(-4.0 * error ** 2)

    @staticmethod
    def compute_torso_upright_reward(projected_gravity_z):
        """
        Reward for keeping torso upright.
        projected_gravity_z = dot(torso_up, world_up).
        1.0 when perfectly upright, decreases with tilt.
        """
        return (projected_gravity_z + 1.0) / 2.0  # normalize from [-1,1] to [0,1]

    @staticmethod
    def compute_energy_penalty(torques, joint_velocities):
        """
        Sum of |torque * angular_velocity| for all actuated joints.
        This directly measures mechanical power consumption.
        Model B/C will have structurally lower values due to fewer actuators.
        """
        return -np.sum(np.abs(torques * joint_velocities))

    @staticmethod
    def compute_foot_slip_penalty(foot_velocities_xy, foot_contacts):
        """
        Penalty for foot sliding while in contact with ground.
        foot_velocities_xy: (n_feet, 2) horizontal velocity
        foot_contacts: (n_feet,) boolean contact flags
        """
        slip = np.sum(np.linalg.norm(foot_velocities_xy, axis=-1) * foot_contacts)
        return -slip


# ============================================================
# PPO CONFIG
# ============================================================
class BasePPOCfg:
    """
    PPO hyperparameters.
    Based on rsl_rl defaults with adjustments for bipedal locomotion.
    """

    # --- Algorithm ---
    algorithm = "PPO"
    num_learning_epochs = 5         # PPO epochs per update
    num_mini_batches = 4            # Mini-batches per epoch
    clip_param = 0.2                # PPO clip parameter
    gamma = 0.99                    # Discount factor
    lam = 0.95                      # GAE lambda
    value_loss_coef = 1.0           # Value function loss coefficient
    entropy_coef = 0.01             # Entropy bonus (exploration)
    learning_rate = 3e-4            # Adam learning rate
    max_grad_norm = 1.0             # Gradient clipping
    schedule = "adaptive"           # LR schedule: "adaptive" or "fixed"
    desired_kl = 0.01               # Target KL for adaptive LR

    # --- Network ---
    policy_hidden_dims = [256, 256, 128]   # Actor network
    value_hidden_dims = [256, 256, 128]    # Critic network
    activation = "elu"                      # Activation function

    # --- Training ---
    max_iterations = 15000          # Training iterations (phase 2)
    max_iterations_phase1 = 2000    # Quick phase 1 check
    save_interval = 1000            # Save checkpoint every N iterations
    log_interval = 100              # Log metrics every N iterations

    # --- Normalization ---
    normalize_obs = True            # Running mean/std normalization
    normalize_value = True          # Value function normalization


# ============================================================
# DOMAIN RANDOMIZATION CONFIG
# ============================================================
class BaseDomainRandCfg:
    """
    Domain randomization for sim-to-real robustness.
    Start with mild randomization. Increase if policies are brittle.
    """

    randomize = True

    # --- Physics ---
    friction_range = [0.5, 1.5]         # Ground friction multiplier
    restitution_range = [0.0, 0.2]      # Ground bounciness
    added_mass_range = [-1.0, 2.0]      # Random mass on torso (kg)

    # --- Actuator ---
    motor_strength_range = [0.85, 1.15]  # Motor gear multiplier
    motor_offset_range = [-0.02, 0.02]   # Joint position offset (rad)

    # --- Delays ---
    action_delay_range = [0, 3]          # Action delay in sim steps (0-15ms)

    # --- External forces ---
    push_robots = True
    push_interval_s = 8.0                # Push every N seconds
    push_force_range = [0.0, 50.0]       # Random push force (N)

    # --- Spring parameters (for Model B/C) ---
    knee_stiffness_range = [0.8, 1.2]    # Multiplier on knee spring stiffness
    toe_stiffness_range = [0.8, 1.2]     # Multiplier on toe spring stiffness


# ============================================================
# EVALUATION CONFIG
# ============================================================
class BaseEvalCfg:
    """Parameters for post-training evaluation."""

    # --- CoT measurement ---
    eval_duration_s = 30.0          # Evaluation episode length
    eval_target_vel = 1.0           # Target velocity during eval
    num_eval_episodes = 100         # Number of episodes for statistics

    # --- Payload test ---
    payload_masses = [0, 5, 10, 20]  # kg added to torso

    # --- Robustness test ---
    push_forces = [50, 100, 150]     # N, lateral push
    push_duration = 0.1              # seconds
    num_push_trials = 50             # trials per force level

    # --- Video ---
    video_length_s = 10.0
    video_fps = 30
    camera_distance = 3.0
    camera_elevation = -20.0

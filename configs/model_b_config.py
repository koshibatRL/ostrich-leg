"""
Model B: Reverse-Knee (Ostrich-type) Bipedal

- Knee bends backward only (0° to 120°)
- NO knee actuator — passive spring (stiffness=40, tensegrity analog)
- Toe spring for elastic energy storage (SEA, stiffness=25)
- 10 actuators (5 per leg): hip_yaw, hip_roll, hip_pitch, ankle_pitch, toe
- Distal mass reduced (lighter shin), proximal mass increased (heavier thigh)
- Based on: MIT Haberland & Kim (2015), Rubenson et al. (2011), Wen et al. (2025)
"""

from configs.base_config import BaseEnvCfg, BaseRewardCfg, BasePPOCfg, BaseDomainRandCfg

MODEL_XML = "models/model_b_reverse_knee.xml"
MODEL_NAME = "reverse_knee"


class ModelBEnvCfg(BaseEnvCfg):
    model_xml = MODEL_XML
    model_name = MODEL_NAME

    num_actuators = 10  # 2 fewer than Model A (no knee motors)
    num_joints = 12     # joints are same count (knee still has joint, just no motor)

    actuator_names = [
        "r_hip_yaw_act", "r_hip_roll_act", "r_hip_pitch_act",
        "r_ankle_pitch_act", "r_toe_act",
        "l_hip_yaw_act", "l_hip_roll_act", "l_hip_pitch_act",
        "l_ankle_pitch_act", "l_toe_act",
    ]

    joint_names = [
        "r_hip_yaw", "r_hip_roll", "r_hip_pitch",
        "r_knee", "r_ankle_pitch", "r_toe_joint",
        "l_hip_yaw", "l_hip_roll", "l_hip_pitch",
        "l_knee", "l_ankle_pitch", "l_toe_joint",
    ]

    # Default joint positions (reverse knee standing pose)
    # Knee at slight extension (spring pushes toward 5° = near straight)
    default_joint_positions = {
        "r_hip_yaw": 0.0, "r_hip_roll": 0.0, "r_hip_pitch": 0.2,
        "r_knee": 0.1,  # Slight bend backward (spring holds near extension)
        "r_ankle_pitch": -0.1, "r_toe_joint": 0.0,
        "l_hip_yaw": 0.0, "l_hip_roll": 0.0, "l_hip_pitch": 0.2,
        "l_knee": 0.1,
        "l_ankle_pitch": -0.1, "l_toe_joint": 0.0,
    }

    foot_body_names = ["r_foot", "l_foot"]

    # Note: r_toe and l_toe also contact ground but are separate bodies

    kp = 80.0
    kd = 4.0

    # Spring parameters (for reference / logging)
    knee_spring_stiffness = 40.0    # Nm/rad (set in MJCF)
    knee_spring_ref = 5.0           # degrees (natural angle in MJCF)
    toe_spring_stiffness = 25.0     # Nm/rad
    toe_spring_ref = 0.0            # degrees


class ModelBRewardCfg(BaseRewardCfg):
    """
    Same reward as Model A. This is intentional — fair comparison.
    The energy penalty will naturally be lower because:
    1. Fewer actuators (10 vs 12) → fewer terms in energy sum
    2. Knee work is done by spring (free energy) instead of motor
    3. Toe spring stores/releases energy passively
    """
    pass


class ModelBPPOCfg(BasePPOCfg):
    """No overrides."""
    pass


class ModelBDomainRandCfg(BaseDomainRandCfg):
    """Include spring parameter randomization for robustness."""
    knee_stiffness_range = [0.8, 1.2]   # ±20% variation on knee spring
    toe_stiffness_range = [0.8, 1.2]    # ±20% variation on toe spring

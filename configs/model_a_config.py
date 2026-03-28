"""
Model A: Conventional Forward-Knee Bipedal (Baseline)

- Standard human-like knee (bends forward only)
- 12 actuators (6 per leg): hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll
- No springs, no elastic energy storage
- This is the control group for the comparison experiment
"""

from configs.base_config import BaseEnvCfg, BaseRewardCfg, BasePPOCfg, BaseDomainRandCfg

MODEL_XML = "models/model_a_forward_knee.xml"
MODEL_NAME = "forward_knee"


class ModelAEnvCfg(BaseEnvCfg):
    model_xml = MODEL_XML
    model_name = MODEL_NAME

    num_actuators = 12
    num_joints = 12  # excluding freejoint

    # Actuator mapping (order matches MJCF)
    actuator_names = [
        "r_hip_yaw_act", "r_hip_roll_act", "r_hip_pitch_act",
        "r_knee_act", "r_ankle_pitch_act", "r_ankle_roll_act",
        "l_hip_yaw_act", "l_hip_roll_act", "l_hip_pitch_act",
        "l_knee_act", "l_ankle_pitch_act", "l_ankle_roll_act",
    ]

    # Joint names for observation
    joint_names = [
        "r_hip_yaw", "r_hip_roll", "r_hip_pitch",
        "r_knee", "r_ankle_pitch", "r_ankle_roll",
        "l_hip_yaw", "l_hip_roll", "l_hip_pitch",
        "l_knee", "l_ankle_pitch", "l_ankle_roll",
    ]

    # Default joint positions (standing pose, radians)
    default_joint_positions = {
        "r_hip_yaw": 0.0, "r_hip_roll": 0.0, "r_hip_pitch": -0.2,
        "r_knee": 0.4, "r_ankle_pitch": 0.2, "r_ankle_roll": 0.0,
        "l_hip_yaw": 0.0, "l_hip_roll": 0.0, "l_hip_pitch": -0.2,
        "l_knee": 0.4, "l_ankle_pitch": 0.2, "l_ankle_roll": 0.0,
    }

    # Foot contact body names (for slip detection)
    foot_body_names = ["r_foot", "l_foot"]

    # PD gains for position control (if using position targets instead of torque)
    kp = 80.0   # Proportional gain
    kd = 4.0    # Derivative gain


class ModelARewardCfg(BaseRewardCfg):
    """No overrides — uses base reward as-is."""
    pass


class ModelAPPOCfg(BasePPOCfg):
    """No overrides — uses base PPO config."""
    pass


class ModelADomainRandCfg(BaseDomainRandCfg):
    """No spring randomization needed (no springs in Model A)."""
    knee_stiffness_range = None  # Not applicable
    toe_stiffness_range = None   # Not applicable

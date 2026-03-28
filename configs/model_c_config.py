"""
Model C: Bidirectional Knee Bipedal (PROPOSED DESIGN)

- Knee bends BOTH forward (-90°) and backward (+120°)
- NO knee actuator — spring-loaded bidirectional (stiffness=35)
- Toe spring for elastic energy storage (SEA, stiffness=25)
- 10 actuators (5 per leg): same as Model B
- RL freely discovers when to use forward vs backward knee
- Hypothesis: walking uses backward knee, tasks use forward knee

This is Koki Shibata's unique proposal — combining:
1. MIT paper's reverse knee efficiency
2. Tensegrity knee's motorless mechanism
3. Bidirectional capability for versatile human-like tasks
"""

from configs.base_config import BaseEnvCfg, BaseRewardCfg, BasePPOCfg, BaseDomainRandCfg

MODEL_XML = "models/model_c_bidirectional_knee.xml"
MODEL_NAME = "bidirectional_knee"


class ModelCEnvCfg(BaseEnvCfg):
    model_xml = MODEL_XML
    model_name = MODEL_NAME

    num_actuators = 10
    num_joints = 12

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

    # Default: start near extension (spring equilibrium)
    # RL will discover which direction to bend from here
    default_joint_positions = {
        "r_hip_yaw": 0.0, "r_hip_roll": 0.0, "r_hip_pitch": 0.0,
        "r_knee": 0.05,  # Near extension, slight backward lean
        "r_ankle_pitch": 0.0, "r_toe_joint": 0.0,
        "l_hip_yaw": 0.0, "l_hip_roll": 0.0, "l_hip_pitch": 0.0,
        "l_knee": 0.05,
        "l_ankle_pitch": 0.0, "l_toe_joint": 0.0,
    }

    foot_body_names = ["r_foot", "l_foot"]

    kp = 80.0
    kd = 4.0

    # Spring parameters
    knee_spring_stiffness = 35.0    # Slightly softer than Model B for bidirectional flex
    knee_spring_ref = 5.0           # degrees
    toe_spring_stiffness = 25.0
    toe_spring_ref = 0.0


class ModelCRewardCfg(BaseRewardCfg):
    """
    Same base reward as A and B for fair comparison.

    The key question for Model C:
    Will RL discover that backward knee is more efficient for walking?
    We DON'T add any reward bias toward either knee direction.
    The energy penalty alone should drive the RL toward the efficient direction.

    Post-training analysis will check:
    - knee_angle > 0 → backward (ostrich mode)
    - knee_angle < 0 → forward (human mode)
    - We log this ratio to show emergent behavior
    """
    pass


class ModelCPPOCfg(BasePPOCfg):
    """
    Potentially needs more iterations than A/B because:
    - Larger exploration space (knee can go both directions)
    - RL needs to discover the efficient direction

    Start with same params, increase if needed.
    """
    max_iterations = 20000          # 33% more than base (15000)
    max_iterations_phase1 = 3000    # 50% more for initial exploration


class ModelCDomainRandCfg(BaseDomainRandCfg):
    """Include spring randomization."""
    knee_stiffness_range = [0.8, 1.2]
    toe_stiffness_range = [0.8, 1.2]


# ============================================================
# Additional analysis for Model C
# ============================================================
class ModelCAnalysisCfg:
    """
    Post-training analysis specific to Model C.
    Tracks which knee direction the RL policy prefers.
    """

    # Knee direction classification thresholds (radians)
    # These correspond to the MJCF joint angle convention:
    #   negative = forward (human-like)
    #   positive = backward (ostrich-like)
    backward_threshold = 0.05   # rad (~3°): above this = clearly backward
    forward_threshold = -0.05   # rad: below this = clearly forward
    neutral_zone = [-0.05, 0.05]  # Near extension, direction ambiguous

    # Metrics to compute
    compute_knee_direction_ratio = True  # % time in backward vs forward
    compute_knee_transition_count = True  # How often does knee switch direction
    compute_phase_knee_correlation = True  # Correlation between gait phase and knee direction

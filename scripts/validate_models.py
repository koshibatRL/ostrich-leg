"""
Validate all 3 bipedal leg models in MuJoCo.
Checks: model loading, DOF count, actuator count, total mass, joint ranges, physics step.
"""
import mujoco
import numpy as np
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../models"

models = {
    "Model A (Forward Knee)":    "model_a_forward_knee.xml",
    "Model B (Reverse Knee)":    "model_b_reverse_knee.xml",
    "Model C (Bidirectional)":   "model_c_bidirectional_knee.xml",
}

print("=" * 70)
print("BIPEDAL LEG PLATFORM — Model Validation")
print("=" * 70)

for name, filename in models.items():
    path = os.path.join(MODEL_DIR, filename)
    print(f"\n{'─' * 70}")
    print(f"  {name}")
    print(f"  File: {filename}")
    print(f"{'─' * 70}")

    try:
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"  ❌ FAILED TO LOAD: {e}")
        continue

    print(f"  ✅ Loaded successfully")
    print(f"  DOF (nv):        {model.nv}")
    print(f"  Bodies (nbody):  {model.nbody}")
    print(f"  Joints (njnt):   {model.njnt}")
    print(f"  Actuators (nu):  {model.nu}")
    print(f"  Sensors:         {model.nsensor}")

    # Total mass
    total_mass = sum(model.body_mass)
    print(f"  Total mass:      {total_mass:.2f} kg")

    # Joint info
    print(f"\n  Joints:")
    for i in range(model.njnt):
        jnt_name = model.joint(i).name
        jnt_type = model.jnt_type[i]
        type_str = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(jnt_type, "?")
        if jnt_type == 3:  # hinge
            lo = np.degrees(model.jnt_range[i, 0])
            hi = np.degrees(model.jnt_range[i, 1])
            stiff = model.jnt_stiffness[i]
            stiff_str = f"  stiffness={stiff:.1f}" if stiff > 0 else ""
            print(f"    {jnt_name:25s} [{lo:7.1f}°, {hi:7.1f}°]{stiff_str}")
        else:
            print(f"    {jnt_name:25s} ({type_str})")

    # Actuator info
    print(f"\n  Actuators:")
    for i in range(model.nu):
        act_name = model.actuator(i).name
        gear = model.actuator_gear[i, 0]
        print(f"    {act_name:25s}  gear={gear:.0f}")

    # Quick physics test (100 steps)
    mujoco.mj_resetData(model, data)
    initial_pos = data.qpos[2]  # z-position of torso
    for _ in range(100):
        mujoco.mj_step(model, data)
    final_pos = data.qpos[2]

    print(f"\n  Physics test (100 steps):")
    print(f"    Initial torso z: {initial_pos:.4f} m")
    print(f"    Final torso z:   {final_pos:.4f} m")
    if final_pos > 0.1:
        print(f"    ✅ Robot still above ground")
    else:
        print(f"    ⚠️  Robot may have collapsed (check model)")

print(f"\n{'=' * 70}")
print("COMPARISON SUMMARY")
print(f"{'=' * 70}")

# Reload for comparison
results = {}
for name, filename in models.items():
    path = os.path.join(MODEL_DIR, filename)
    model = mujoco.MjModel.from_xml_path(path)
    results[name] = {
        "actuators": model.nu,
        "dof": model.nv,
        "mass": sum(model.body_mass),
        "joints": model.njnt,
    }

print(f"\n  {'Model':<30s} {'Actuators':>10s} {'DOF':>6s} {'Mass (kg)':>10s} {'Joints':>8s}")
print(f"  {'─'*30} {'─'*10} {'─'*6} {'─'*10} {'─'*8}")
for name, r in results.items():
    short = name.split("(")[1].rstrip(")")
    print(f"  {short:<30s} {r['actuators']:>10d} {r['dof']:>6d} {r['mass']:>10.2f} {r['joints']:>8d}")

# Cost reduction estimate
a_act = results["Model A (Forward Knee)"]["actuators"]
b_act = results["Model B (Reverse Knee)"]["actuators"]
reduction = (a_act - b_act) / a_act * 100
print(f"\n  Actuator reduction (A→B): {a_act} → {b_act} ({reduction:.0f}% fewer)")
print(f"  Estimated cost saving: ${(a_act - b_act) * 500}-${(a_act - b_act) * 2000} per unit")
print(f"\n{'=' * 70}")

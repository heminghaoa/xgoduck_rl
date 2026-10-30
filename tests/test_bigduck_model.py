import math

import mujoco
import numpy as np
import pytest

from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot import bigduck_constants as bc


@pytest.fixture(scope="module")
def walk_spec():
    return bc.get_walk_spec()


def test_bigduck_mass_and_balance(walk_spec):
    r = bc.measure(walk_spec)
    # docs/big-duck.md estimates about 1.9 kg with the electronics in the back bay
    assert 1.8 < r["mass"] < 2.0
    # The stand height the HOME frame spawns at is the one this model measures.
    assert r["stand_z"] == pytest.approx(bc.BIGDUCK_STAND_Z, abs=5e-4)
    # The longer heels put the soles' middle under the centre of mass, as on Microduck.
    m = walk_spec.compile()
    d = mujoco.MjData(m)
    d.qpos[:] = bc._home_qpos(m)
    mujoco.mj_kinematics(m, d)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    sole_middle_x = d.geom_xpos[g][0]
    assert 0.0 < sole_middle_x - r["com"][0] < 0.015


def test_bigduck_feet_and_neck(walk_spec):
    m = walk_spec.compile()
    assert m.nu == 14
    micro = mujoco.MjModel.from_xml_path(str(bc.MICRODUCK_WALK_XML))
    for side in ("left", "right"):
        name = f"{side}_foot_collision"
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        sole = bc._geom_points(micro, mujoco.mj_name2id(micro, mujoco.mjtObj.mjOBJ_GEOM, name))
        microduck_length = np.ptp(sole, axis=0).max()
        assert 2 * m.geom_size[g].max() == pytest.approx(microduck_length * bc.SCALE + bc.HEEL, abs=1e-3)
    yaw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "head_yaw")
    assert m.jnt_range[yaw] == pytest.approx([-math.pi / 2, math.pi / 2])


def test_bigduck_walk_task_scales_lengths():
    cfg = load_env_cfg("Mjlab-Velocity-Flat-BigDuck")
    micro = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    for name in ("foot_clearance", "foot_swing_height"):
        assert cfg.rewards[name].params["target_height"] == pytest.approx(
            micro.rewards[name].params["target_height"] * bc.SCALE)
    assert cfg.events["reset_base"].params["pose_range"]["z"] == (0.0, 0.0)
    assert cfg.scene.entities["robot"].init_state.pos[2] == bc.BIGDUCK_STAND_Z


def test_bigduck_standup_task_scales_heights():
    from mjlab_microduck.tasks.bigduck_standup_env_cfg import (
        BACK_BAY_PRONE_LIFT,
        REWARD_LENGTHS,
        RESET_LENGTHS,
    )

    cfg = load_env_cfg("Mjlab-StandUp-Flat-BigDuck")
    micro = load_env_cfg("Mjlab-StandUp-Flat-MicroDuck")
    for name, keys in REWARD_LENGTHS.items():
        for key in keys:
            assert cfg.rewards[name].params[key] == pytest.approx(
                micro.rewards[name].params[key] * bc.SCALE)
    for key in RESET_LENGTHS:
        lift = BACK_BAY_PRONE_LIFT if key.startswith("prone") else 0.0
        assert cfg.events["set_ground_state"].params[key] == pytest.approx(
            micro.events["set_ground_state"].params[key] * bc.SCALE + lift)
    assert cfg.commands["body_pose"].ranges[2] == pytest.approx(
        tuple(v * bc.SCALE for v in micro.commands["body_pose"].ranges[2]))
    # Microduck's measured stand sits a few mm under its HOME frame; the scaled one does too.
    stand = cfg.rewards["height_stand"].params["target_height"]
    assert 0.0 < bc.BIGDUCK_STAND_Z - stand < 0.006

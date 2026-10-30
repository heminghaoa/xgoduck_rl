"""Big duck getting up: Microduck's stand-up recipe on the 1.4x ST3215 model.

The recipe starts the duck face down, face up, sitting or standing and pays it for reaching the
standing height upright. Every trunk height it gives in metres (targets, their spreads, reward
gates, reset bands, the body-pose z command) grows with the duck. Weights, speeds and noise stay
Microduck's, as in the walk task.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab_microduck.robot.bigduck_constants import (
    BIGDUCK_STANDUP_ROBOT_CFG,
    HEAD_BODY_NAMES,
    SCALE,
)
from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    MicroduckStandUpRlCfg,
    make_microduck_standup_env_cfg,
)

# TensorBoard logs under logs/rsl_rl/bigduck_standup/.
BigduckStandUpRlCfg = replace(
    MicroduckStandUpRlCfg,
    logger="tensorboard",
    experiment_name="bigduck_standup",
    run_name="bigduck",
    upload_model=False,
)

# Reward parameters that are trunk heights or height spreads in metres.
REWARD_LENGTHS = {
    "head_pose_bias": ("gate_height_low", "gate_height_high"),
    "height_stand": ("std", "target_height"),
    "height_stand_sharp": ("std", "target_height"),
    "height_stand_l1": ("target_height",),
    "com_upward_velocity": ("max_height",),
    "arrival_damping": ("height_low", "height_high"),
    "upright_sharp": ("height_low", "height_high"),
    "standing_composite": ("target_height", "height_std"),
    "body_pose_tracking": ("nominal_height", "z_std"),
}
# Trunk height bands set_random_ground_state spawns each start pose in.
RESET_LENGTHS = (
    "prone_z_min", "prone_z_max",
    "sitting_z_min", "sitting_z_max",
    "standing_z_min", "standing_z_max",
)
# Lying on its back, the duck rests on the back bay: the trunk clears the floor at 0.091 m against
# 0.067 m for Microduck scaled up. Face down and face up share one band, so it all moves up by the
# difference and face-up starts no deeper in the floor than Microduck's do.
BACK_BAY_PRONE_LIFT = 0.024


def _scale_z(ranges):
    """Body-pose ranges are (x, y, z, roll, pitch, yaw); only z is a length here."""
    ranges = list(ranges)
    ranges[2] = tuple(v * SCALE for v in ranges[2])
    return tuple(ranges)


def make_bigduck_standup_env_cfg(**kwargs):
    cfg = make_microduck_standup_env_cfg(**kwargs)
    cfg.scene.entities = {"robot": deepcopy(BIGDUCK_STANDUP_ROBOT_CFG)}
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["asset_cfg"].body_names = HEAD_BODY_NAMES

    for name, keys in REWARD_LENGTHS.items():
        if name in cfg.rewards:
            for key in keys:
                cfg.rewards[name].params[key] *= SCALE
    ground = cfg.events["set_ground_state"].params
    for key in RESET_LENGTHS:
        ground[key] *= SCALE
    ground["prone_z_min"] += BACK_BAY_PRONE_LIFT
    ground["prone_z_max"] += BACK_BAY_PRONE_LIFT

    if "body_pose" in cfg.commands:
        cmd = cfg.commands["body_pose"]
        cmd.ranges = _scale_z(cmd.ranges)
        for stage in cfg.curriculum["body_pose_range"].params["range_stages"]:
            stage["ranges"] = _scale_z(stage["ranges"])
    return cfg

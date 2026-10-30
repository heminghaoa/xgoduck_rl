"""Big duck walking: Microduck's walk recipe on the 1.4x ST3215 model.

Lengths the recipe gives in metres grow with the duck (foot clearance, swing height, the ring of
the foot height scan). Speeds, reward weights and noise stay Microduck's until the big duck has
walked on them.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.sensor import RingPatternCfg

from mjlab_microduck.robot.bigduck_constants import (
    BIGDUCK_WALK_ROBOT_CFG,
    HEAD_BODY_NAMES,
    SCALE,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# TensorBoard logs under logs/rsl_rl/bigduck_velocity/.
BigduckRlCfg = replace(
    MicroduckRlCfg,
    logger="tensorboard",
    experiment_name="bigduck_velocity",
    run_name="bigduck",
    upload_model=False,
)


def make_bigduck_velocity_env_cfg(**kwargs):
    cfg = make_microduck_velocity_env_cfg(**kwargs)
    cfg.scene.entities = {"robot": deepcopy(BIGDUCK_WALK_ROBOT_CFG)}
    # reset_root_state_uniform adds this range to HOME's height, which already leaves the soles
    # 1 mm up.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["asset_cfg"].body_names = HEAD_BODY_NAMES
    for name in ("foot_clearance", "foot_swing_height"):
        cfg.rewards[name].params["target_height"] *= SCALE
    cfg.scene.sensors = tuple(
        replace(s, pattern=RingPatternCfg.single_ring(radius=0.04 * SCALE, num_samples=2))
        if s.name == "foot_height_scan" else s
        for s in cfg.scene.sensors
    )
    return cfg

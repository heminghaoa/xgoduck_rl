"""Big duck: Microduck scaled up 1.4 times around 14 Feetech ST3215-C018 servos.

Plan and layout: xgoduck_runtime_arduino, branch big-duck, docs/big-duck.md.

The model is derived from the Microduck XMLs as it loads instead of being kept as a copy:

- every length is multiplied by SCALE;
- each body's printed parts weigh STRUCT_MASS_FACTOR times Microduck's and its XL330s become
  55 g ST3215s; the mouth servo is left out, the jaw is fixed in the first build;
- Microduck's battery and head electronics give way to the big duck's: battery, Pi5, buck
  converter, servo adapter and amp in an enlarged back bay, the IMU between the hip servos,
  only the camera, microphones and speaker in the head;
- each sole becomes a box with a heel 16 mm longer, and head yaw stops at ±90° so the camera
  ribbon through the neck survives.

Masses are estimates. The Onshape export of the redrawn parts, with weighed parts, replaces
this model. tests/test_bigduck_model.py checks the built model's mass, feet and stand height.
"""

import math
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import bam
import mujoco
import numpy as np
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.actuator import FrictionDRBamActuatorCfg
from mjlab_microduck.robot.microduck_constants import (
    FULL_COLLISION,
    HOME_FRAME,
    MICRODUCK_ALLCOLLISIONS_XML,
    MICRODUCK_WALK_XML,
)

SCALE = 1.4
# Printed parts get heavier somewhere between wall-limited (SCALE**2) and solid (SCALE**3).
STRUCT_MASS_FACTOR = (SCALE**2 + SCALE**3) / 2
HEEL = 0.016  # m added behind each sole
HEAD_YAW_LIMIT = math.pi / 2
# Bodies the tasks shift the head CoM on. MicroDuck's legacy list includes bearing_roll, which is
# its right hip.
HEAD_BODY_NAMES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")

XL330_MASS = 0.018
ST3215_MASS = 0.055
# ST3215 body in the XL330 mesh's axes: x width, y along the shaft, z length.
ST3215_BOX = (0.0247, 0.035, 0.0452)
MICRODUCK_BATTERY_MASS = 0.080  # NP-F550 class pack in the trunk
MICRODUCK_HEAD_ELECTRONICS_MASS = 0.035  # Pi Zero 2 W, HAT, camera, speaker

# trunk_base frame (x forward, y left, z up), metres, already at SCALE:
# name -> (mass kg, centre, box size). Positions follow docs/big-duck.md in the runtime repo.
TRUNK_PARTS = {
    "battery": (0.120, (-0.0525, -0.0095, -0.0225), (0.025, 0.035, 0.075)),
    "buck": (0.025, (-0.0525, 0.0175, -0.0325), (0.025, 0.015, 0.055)),
    "pi5": (0.075, (-0.076, 0.0, -0.0175), (0.016, 0.056, 0.085)),
    "servo_adapter": (0.016, (-0.0495, 0.0, 0.0275), (0.033, 0.042, 0.0185)),
    "amp": (0.003, (-0.060, 0.028, 0.0), (0.005, 0.018, 0.020)),
    "bay_walls": (0.030, (-0.063, 0.0, -0.0115), (0.056, 0.062, 0.101)),
    "imu": (0.005, (0.0, 0.0, 0.023), (0.027, 0.006, 0.021)),
}
BACK_BAY = ((-0.091, -0.035), (-0.031, 0.031), (-0.062, 0.039))  # outer box, trunk frame
CAMERA_MASS = 0.004
SPEAKER_AND_MICS_MASS = 0.017

# Placeholder actuator until the C018 is identified on the test bench: BAM's STS3215 (7.4 V)
# fit, run at 6.0-7.0 V so its stall torque (kt * V / R, about 2.9 N·m) lands on the C018's
# rated 30 kg·cm. Firmware P = 32 is the Feetech default the OpenDuckMini runtime keeps.
_STS3215_JSON = Path(bam.__file__).parent / "params" / "feetech_sts3215_7_4V" / "m1.json"
assert _STS3215_JSON.exists(), f"BAM STS3215 params not found: {_STS3215_JSON}"


def _box_inertia(mass, size):
    a, b, c = size
    return np.diag([b * b + c * c, a * a + c * c, a * a + b * b]) * mass / 12.0


def _quat_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def _mat_quat(r):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(r, dtype=float).reshape(9))
    return q


@lru_cache(maxsize=None)
def _microduck(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(xml)


def _geom_points(m: mujoco.MjModel, g: int) -> np.ndarray:
    """A mesh geom's vertices in its body's frame."""
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return v @ _quat_mat(m.geom_quat[g]).T + m.geom_pos[g]


def _visual_geoms(m: mujoco.MjModel, mesh: str):
    for g in range(m.ngeom):
        if (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2
                and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) == mesh):
            yield g


def _centre(points):
    return (points.min(0) + points.max(0)) / 2


def _mass_parts(m: mujoco.MjModel):
    """Per body: the parts that replace its inertial, as (mass, centre, inertia) at SCALE."""
    servos = {}
    for g in _visual_geoms(m, "xl330"):
        p = _geom_points(m, g)
        # the mesh frame's axes in the body frame, from the geom orientation
        servos.setdefault(m.geom_bodyid[g], []).append((_centre(p), _quat_mat(m.geom_quat[g])))

    head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    roll = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "head_roll")
    # The head holds the head-roll servo, on the roll axis, and the mouth servo, off it.
    axis_p, axis_d = m.jnt_pos[roll], m.jnt_axis[roll]
    off_axis = [np.linalg.norm(np.cross(c - axis_p, axis_d)) for c, _ in servos[head]]
    mouth = int(np.argmax(off_axis))

    removed = {
        trunk: [(MICRODUCK_BATTERY_MASS, _centre(_geom_points(m, next(_visual_geoms(m, "np_f970")))))],
        head: [(MICRODUCK_HEAD_ELECTRONICS_MASS,
                _centre(_geom_points(m, next(_visual_geoms(m, "pcb__raspberry_pi_zero_2_w")))))],
    }
    lens = _centre(_geom_points(m, next(_visual_geoms(m, "lens")))) * SCALE
    speaker = _centre(_geom_points(m, next(_visual_geoms(m, "speaker")))) * SCALE

    parts = {}
    for b in range(1, m.nbody):
        mass, com = m.body_mass[b], m.body_ipos[b]
        r = _quat_mat(m.body_iquat[b])
        inertia = r @ np.diag(m.body_inertia[b]) @ r.T
        out_m = [(XL330_MASS, c) for c, _ in servos.get(b, [])] + removed.get(b, [])
        struct_m = mass - sum(w for w, _ in out_m)
        struct_c = (mass * com - sum(w * c for w, c in out_m)) / struct_m
        # the structure keeps its share of the body's inertia, grown by mass and length squared
        items = [(struct_m * STRUCT_MASS_FACTOR, struct_c * SCALE,
                  inertia * (struct_m / mass) * STRUCT_MASS_FACTOR * SCALE**2)]
        for i, (c, rot) in enumerate(servos.get(b, [])):
            if b == head and i == mouth:
                continue
            items.append((ST3215_MASS, c * SCALE, rot @ _box_inertia(ST3215_MASS, ST3215_BOX) @ rot.T))
        if b == trunk:
            items += [(w, np.array(c), _box_inertia(w, s)) for w, c, s in TRUNK_PARTS.values()]
        if b == head:
            items += [(CAMERA_MASS, lens, np.zeros((3, 3))),
                      (SPEAKER_AND_MICS_MASS, speaker, np.zeros((3, 3)))]
        parts[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)] = items
    return parts


def _combine(items):
    mass = sum(w for w, _, _ in items)
    com = sum(w * c for w, c, _ in items) / mass
    inertia = np.zeros((3, 3))
    for w, c, i in items:
        d = c - com
        inertia += i + w * (d @ d * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def _home_qpos(m: mujoco.MjModel) -> np.ndarray:
    """Microduck's HOME joint angles (first matching pattern wins), trunk at the origin."""
    q = m.qpos0.copy()
    q[:7] = [0, 0, 0, 1, 0, 0, 0]
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        for pattern, value in HOME_FRAME.joint_pos.items():
            if re.fullmatch(pattern, name):
                q[m.jnt_qposadr[j]] = value
                break
    return q


def _heel_boxes(spec: mujoco.MjSpec) -> None:
    """Replace each sole mesh with a box around it, its heel HEEL longer.

    The box's axes follow the world's in the HOME pose, so its bottom lies flat when the duck
    stands and the extra length goes straight back.
    """
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[:] = _home_qpos(m)
    mujoco.mj_kinematics(m, d)
    boxes = {}
    for side in ("left", "right"):
        name = f"{side}_foot_collision"
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        axes = d.xmat[m.geom_bodyid[g]].reshape(3, 3).T  # world x, y, z in the foot's frame
        p = _geom_points(m, g) @ axes  # vertex coordinates along the world axes
        lo, hi = p.min(0), p.max(0)
        lo[0] -= HEEL
        boxes[name] = dict(size=(hi - lo) / 2, pos=axes @ ((hi + lo) / 2), quat=_mat_quat(axes))
    # Fresh geoms: turning the mesh geom into a box still fits the box to the old mesh. Both
    # old geoms go before either new one comes, or the second name still counts as taken.
    owners = {}
    for name in boxes:
        old = spec.geom(name)
        owners[name] = (old.parent, old.material)
        spec.delete(old)
    for name, box in boxes.items():
        body, material = owners[name]
        body.add_geom(default=spec.find_default("collision"), name=name,
                      type=mujoco.mjtGeom.mjGEOM_BOX, material=material, **box)


def build_spec(xml: Path, full_collision: bool = False) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml))
    parts = _mass_parts(_microduck(str(xml)))

    for mesh in spec.meshes:
        mesh.scale = np.asarray(mesh.scale) * SCALE
    for body in spec.bodies[1:]:
        body.pos = np.asarray(body.pos) * SCALE
        mass, com, inertia = _combine(parts[body.name])
        body.mass, body.ipos = mass, com
        body.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                            inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    for el in list(spec.geoms) + list(spec.sites) + list(spec.cameras):
        el.pos = np.asarray(el.pos) * SCALE
    for site in spec.sites:
        site.size = np.asarray(site.size) * SCALE
    for joint in spec.joints:
        joint.pos = np.asarray(joint.pos) * SCALE
    spec.joint("head_yaw").range = [-HEAD_YAW_LIMIT, HEAD_YAW_LIMIT]

    # Microduck's battery, Pi Zero and HAT leave; the back bay and its contents show up.
    for geom in [g for g in spec.geoms if g.meshname in (
            "np_f970", "pcb__raspberry_pi_zero_2_w", "elec_rpi_robot_hat_pcb", "banana_pcb_locker")]:
        spec.delete(geom)
    trunk = spec.body("trunk_base")
    for name, (_, centre, size) in TRUNK_PARTS.items():
        if name != "bay_walls":
            trunk.add_geom(name=f"{name}_visual", type=mujoco.mjtGeom.mjGEOM_BOX, pos=centre,
                           size=np.asarray(size) / 2, group=2, contype=0, conaffinity=0,
                           rgba=(0.3, 0.5, 0.9, 1))
    bay_lo, bay_hi = np.array(BACK_BAY).T
    trunk.add_geom(name="back_bay_visual", type=mujoco.mjtGeom.mjGEOM_BOX, pos=(bay_lo + bay_hi) / 2,
                   size=(bay_hi - bay_lo) / 2, group=2, contype=0, conaffinity=0, rgba=(0.9, 0.9, 0.9, 0.3))
    if full_collision:
        # Like Microduck's link meshes in this model: unnamed, so the XML defaults decide contacts.
        trunk.add_geom(default=spec.find_default("collision"), type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=(bay_lo + bay_hi) / 2, size=(bay_hi - bay_lo) / 2)

    _heel_boxes(spec)
    # spec.geom(name) only finds the new foot geoms after a compile; CollisionCfg looks them up.
    spec.compile()
    return spec


def get_walk_spec() -> mujoco.MjSpec:
    return build_spec(MICRODUCK_WALK_XML)


def get_standup_spec() -> mujoco.MjSpec:
    return build_spec(MICRODUCK_ALLCOLLISIONS_XML, full_collision=True)


def measure(spec: mujoco.MjSpec) -> dict:
    """Mass, centre of mass and the trunk height that leaves the soles 1 mm up, in HOME."""
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[:] = _home_qpos(m)
    mujoco.mj_forward(m, d)
    low = min(
        (d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ (np.array(corner) * m.geom_size[g]))[2]
        for g in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                  for n in ("left_foot_collision", "right_foot_collision"))
        for corner in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    )
    stand_z = 0.001 - low
    return dict(mass=float(m.body_subtreemass[1]), com=d.subtree_com[1] + [0, 0, stand_z],
                stand_z=float(stand_z))


# Trunk height in HOME with the soles 1 mm above the floor (measure(get_walk_spec())).
BIGDUCK_STAND_Z = 0.1651

BIGDUCK_HOME_FRAME = deepcopy(HOME_FRAME)
BIGDUCK_HOME_FRAME.pos = (0.0, 0.0, BIGDUCK_STAND_Z)

actuators = FrictionDRBamActuatorCfg(
    json_path=str(_STS3215_JSON),
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=32.0,
    vin_range=(6.0, 7.0),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=5.5,
    delay_min_lag=3,
    delay_max_lag=6,
)

BIGDUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=BIGDUCK_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

BIGDUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=BIGDUCK_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)


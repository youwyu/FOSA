"""Robot specifications and MuJoCo model compilation for generated training scenes."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from fosa import REPO_ROOT
from fosa.sim import scene_contract, warp_setup

warp_setup.configure_warp()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402

DATASET = "SICE-FISH/FOSA"  # the Hugging Face dataset that holds the robot models, under assets/robots/<robot>/


def robot_asset_root() -> Path:
    """The robot models, downloaded from the FOSA dataset into REPO_ROOT/cache on first use."""
    root = REPO_ROOT / "cache" / "assets" / "robots"
    if not (root / ".downloaded").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(DATASET, repo_type="dataset", allow_patterns="assets/robots/*", local_dir=REPO_ROOT / "cache")
        (root / ".downloaded").touch()
    return root


ROBOT_ASSET_ROOT = robot_asset_root()

GO2_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
GO2_HOME_QPOS = (-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8)
DEFAULT_OBJECT_SIZE = (0.12, 0.075, 0.035)
DEFAULT_OBJECT_MASS_KG = 0.35
DEFAULT_OBJECT_RGBA = (0.45, 0.48, 0.42, 1.0)
TASK_HFIELD_NAME = "training_hfield"
CONTACT_CAPACITY_REFERENCE_ENVS = 8192  # RobotSpec contact capacities are sized for this many worlds
SPAWN_RADIUS_M = 0.9  # radius around the origin kept clear for the robot's reset pose
ATTACHED_OVERHANG_FACTOR = 1.25  # cargo may overhang its support surface slightly, like real loads
BOOSTER_JOINT_NAMES = (
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)
BOOSTER_HOME_QPOS = (
    0.0,
    0.0,
    0.0,
    -1.4,
    0.0,
    -0.4,
    0.0,
    1.4,
    0.0,
    0.4,
    0.0,
    -0.2,
    0.0,
    0.0,
    0.4,
    -0.2,
    0.0,
    -0.2,
    0.0,
    0.0,
    0.4,
    -0.2,
    0.0,
)

JACKAL_WHEEL_JOINT_NAMES = ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")

INSPIRE_JOINT_NAMES = (
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_thumb_3_joint",
    "right_thumb_4_joint",
    "right_index_1_joint",
    "right_index_2_joint",
    "right_middle_1_joint",
    "right_middle_2_joint",
    "right_ring_1_joint",
    "right_ring_2_joint",
    "right_little_1_joint",
    "right_little_2_joint",
)
INSPIRE_HOME_QPOS = (
    0.58756485,
    0.11153153,
    0.06970982,
    0.04647115,
    1.37523337,
    1.37527440,
    0.88664161,
    0.88668454,
    1.25627208,
    1.25633351,
    1.63149980,
    1.63143348,
)
INSPIRE_HOME_ACTIONS = (0.58756485, 0.11153153, 1.37523337, 0.88664161, 1.25627208, 1.63149980)


@dataclass(frozen=True)
class RobotSpec:
    name: str
    kind: str
    xml_path: Path
    joint_names: tuple[str, ...]
    home_qpos: tuple[float, ...]
    action_scale: float
    base_height: float
    ctrl_dt: float
    sim_dt: float
    episode_length: int
    command_size: int = 3
    command_components: tuple[str, ...] = ("lin_x", "lin_y", "yaw")
    max_linear_velocity: float = 1.0
    max_angular_velocity: float = 1.0
    wheel_radius: float = 0.095
    track_width: float = 0.37559
    home_actions: tuple[float, ...] | None = None
    freebase_root_body: str | None = None  # the root body when it is not body 1 (the fixed-base hand)
    nconmax: int = 64
    naconmax: int | None = None
    njmax: int = 512
    # Safety termination: a robot has fallen when tipped past ~78 degrees (up_z < termination_min_up),
    # which is terrain-independent; a carried payload is lost by dropping or sliding off.
    termination_min_up: float = 0.2
    payload_drop_z: float = 0.20
    payload_max_offset: float = 0.60
    payload_contact_type: int = 1
    # Home-frame support surface for loose cargo: world [x, y, top_z].
    payload_support_position: tuple[float, float, float] | None = None
    payload_max_footprint: tuple[float, float] | None = None
    heading_control: bool = False
    command_deadzone: float = 0.0
    gait_phase_period: float | None = None
    angular_velocity_sensor: str | None = None
    angular_velocity_obs_scale: float = 0.25
    joint_velocity_obs_scale: float = 0.05
    position_actuator_parameters: tuple[tuple[float, float, float, float], ...] = ()  # (kp, kv, effort, armature)
    ground_contact_geom_tokens: tuple[str, ...] = ()
    foot_contact_geom_token: str = ""
    # Default camera; only published in the simulator capability manifest (fosa.compiler.contracts).
    render_camera_lookat: tuple[float, float, float] = (0.78, 0.0, 0.30)
    render_camera_distance: float = 2.35
    render_camera_azimuth: float = 135.0
    render_camera_elevation: float = -22.0


ROBOT_SPECS: dict[str, RobotSpec] = {
    "go2": RobotSpec(
        name="go2",
        kind="legged",
        xml_path=ROBOT_ASSET_ROOT / "go2" / "assets" / "unitree_go2" / "go2.xml",
        joint_names=GO2_JOINT_NAMES,
        home_qpos=GO2_HOME_QPOS,
        action_scale=0.25,
        base_height=0.32,
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=1000,
        max_linear_velocity=2.0,
        max_angular_velocity=1.0,
        heading_control=True,
        command_deadzone=0.1,
        gait_phase_period=0.6,
        payload_support_position=(0.0, 0.0, 0.502),
        payload_max_footprint=(0.28, 0.20),
        angular_velocity_sensor="imu_ang_vel",
        angular_velocity_obs_scale=1.0,
        joint_velocity_obs_scale=1.0,
        position_actuator_parameters=tuple(
            (40.0, 2.0, 45.0, 0.02) if "calf" in joint_name else (20.0, 1.0, 23.5, 0.01)
            for joint_name in GO2_JOINT_NAMES
        ),
        ground_contact_geom_tokens=(
            "base1_collision",
            "base2_collision",
            "base3_collision",
            "thigh_collision",
            "calf1_collision",
            "calf2_collision",
            "foot_collision",
        ),
        foot_contact_geom_token="foot_collision",
        nconmax=8 * CONTACT_CAPACITY_REFERENCE_ENVS,
        njmax=768,
    ),
    "jackal": RobotSpec(
        name="jackal",
        kind="wheeled",
        xml_path=ROBOT_ASSET_ROOT / "jackal" / "assets" / "xmls" / "jackal.xml",
        joint_names=JACKAL_WHEEL_JOINT_NAMES,
        home_qpos=(0.0, 0.0, 0.0, 0.0),
        action_scale=1.0,
        base_height=0.10,
        ctrl_dt=0.05,
        sim_dt=0.01,
        episode_length=400,
        command_size=2,
        command_components=("lin_x", "yaw"),
        max_linear_velocity=2.0,
        max_angular_velocity=4.0,
        wheel_radius=0.098,
        render_camera_lookat=(0.72, 0.0, 0.12),
        render_camera_distance=2.0,
        render_camera_elevation=-24.0,
        # Eight wheel contacts plus room for both lidar collision manifolds.
        nconmax=5 * len(JACKAL_WHEEL_JOINT_NAMES) * CONTACT_CAPACITY_REFERENCE_ENVS,
        naconmax=6 * len(JACKAL_WHEEL_JOINT_NAMES) * CONTACT_CAPACITY_REFERENCE_ENVS,
        njmax=128,
        payload_drop_z=0.08,
        payload_max_offset=0.55,
        payload_support_position=(-0.18, 0.0, 0.284),
        payload_max_footprint=(0.20, 0.30),
    ),
    "booster": RobotSpec(
        name="booster",
        kind="humanoid",
        xml_path=ROBOT_ASSET_ROOT / "booster" / "assets" / "booster_t1" / "t1_mjx_feetonly.xml",
        joint_names=BOOSTER_JOINT_NAMES,
        home_qpos=BOOSTER_HOME_QPOS,
        action_scale=0.5,
        base_height=0.665,
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        max_linear_velocity=1.0,
        max_angular_velocity=0.8,
        payload_contact_type=4,
        render_camera_lookat=(0.82, 0.0, 0.62),
        render_camera_distance=2.75,
        render_camera_elevation=-18.0,
        nconmax=8 * CONTACT_CAPACITY_REFERENCE_ENVS,
        njmax=128,
    ),
    "inspire_hand": RobotSpec(
        name="inspire_hand",
        kind="hand",
        xml_path=ROBOT_ASSET_ROOT / "inspire_hand" / "assets" / "FTP_right_hand_isaac.xml",
        joint_names=INSPIRE_JOINT_NAMES,
        home_qpos=INSPIRE_HOME_QPOS,
        home_actions=INSPIRE_HOME_ACTIONS,
        action_scale=0.8,
        base_height=0.14,
        ctrl_dt=0.05,
        sim_dt=0.01,
        episode_length=500,
        max_linear_velocity=0.16,
        max_angular_velocity=0.5,
        freebase_root_body="right_hand",
        render_camera_lookat=(0.085, 0.0, 0.075),
        render_camera_distance=0.35,
        render_camera_azimuth=138.0,
        render_camera_elevation=-28.0,
        nconmax=10 * CONTACT_CAPACITY_REFERENCE_ENVS,
        njmax=512,
    ),
}


@dataclass(frozen=True)
class CompiledDynamicObject:
    """Adapter metadata needed to reset one simulator-neutral dynamic object."""

    name: str
    role: str
    carried: bool
    qpos_ids: np.ndarray
    qvel_ids: np.ndarray
    initial_position: tuple[float, float, float]
    initial_orientation: tuple[float, float, float, float]
    reset: dict[str, Any]


@dataclass(frozen=True)
class CompiledRobotModel:
    mj_model: mujoco.MjModel
    mjx_model: mjx.Model
    qpos_ids: np.ndarray
    qvel_ids: np.ndarray
    manipulated_qpos_ids: np.ndarray | None
    manipulated_qvel_ids: np.ndarray | None
    root_body_id: int
    manipulated_body_id: int | None
    palm_site_id: int | None
    xml_path: str
    payload_body_id: int | None = None
    spawn_surface_height: float = 0.0
    dynamic_objects: tuple[CompiledDynamicObject, ...] = ()


def _joint_qpos_ids(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> np.ndarray:
    return np.array([model.joint(name).qposadr[0] for name in joint_names], dtype=np.int32)


def _joint_qvel_ids(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> np.ndarray:
    return np.array([model.joint(name).dofadr[0] for name in joint_names], dtype=np.int32)


def _freejoint_qpos_ids(model: mujoco.MjModel, joint_name: str) -> np.ndarray:
    start = int(model.joint(joint_name).qposadr[0])
    return np.arange(start, start + 7, dtype=np.int32)


def _freejoint_qvel_ids(model: mujoco.MjModel, joint_name: str) -> np.ndarray:
    start = int(model.joint(joint_name).dofadr[0])
    return np.arange(start, start + 6, dtype=np.int32)


def _add_flat_floor(spec: mujoco.MjSpec, *, z: float = 0.0) -> None:
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0.0, 0.0, z),
        size=(20.0, 20.0, 0.01),
        rgba=(0.92, 0.92, 0.90, 1.0),
        contype=1,
        conaffinity=1,
        friction=(0.9, 0.01, 0.001),
    )


def _float_tuple(value: Any, default: tuple[float, ...], *, length: int) -> tuple[float, ...]:
    raw = value if isinstance(value, (list, tuple, np.ndarray)) else default
    return tuple(([float(item) for item in raw] + list(default))[:length])


def _decode_training_scene(training_scene_key: str | None) -> dict[str, Any]:
    if not training_scene_key:
        return {}
    return json.loads(training_scene_key)


def training_scene_key(training_scene: dict[str, Any] | None) -> str | None:
    """A hashable form of the training scene, so compiled models can be cached per scene."""
    if not training_scene:
        return None
    return json.dumps(training_scene, sort_keys=True, separators=(",", ":"))


def _add_training_terrain(spec: mujoco.MjSpec, training_scene: dict[str, Any]) -> bool:
    terrain = training_scene.get("terrain")
    if not terrain or "height" not in terrain:
        return False
    size_x, size_y = terrain["size"]
    nrow, ncol = terrain["nrow"], terrain["ncol"]
    spec.add_hfield(
        name=TASK_HFIELD_NAME,
        size=[size_x, size_y, max(0.005, max(terrain["height"])), 0.03],
        nrow=nrow,
        ncol=ncol,
        userdata=_heightfield_data(terrain, nrow, ncol).ravel(),
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=TASK_HFIELD_NAME,
        rgba=terrain.get("rgba", (0.52, 0.56, 0.46, 1.0)),
        contype=1,
        conaffinity=1,
        friction=_float_tuple(terrain.get("friction"), (0.95, 0.01, 0.001), length=3),
        condim=3,
    )
    return True


def _heightfield_data(terrain: dict[str, Any], nrow: int, ncol: int) -> np.ndarray:
    algorithm = terrain["algorithm"]
    seed_src = json.dumps(terrain, sort_keys=True)
    seed = terrain.get("seed", int(hashlib.sha1(seed_src.encode("utf-8")).hexdigest()[:8], 16))
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, ncol, dtype=np.float64)[None, :]
    y = np.linspace(-1.0, 1.0, nrow, dtype=np.float64)[:, None]
    noise = rng.random((nrow, ncol), dtype=np.float64)
    for _ in range(4):
        noise = (
            noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0) + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)
        ) / 5.0
    wave_freq = 2.0
    waves = 0.5 + 0.25 * np.sin(np.pi * wave_freq * x + rng.uniform(0.0, 2.0 * np.pi))
    waves = waves + 0.25 * np.cos(np.pi * (wave_freq + 0.75) * y + rng.uniform(0.0, 2.0 * np.pi))
    steps = np.floor(noise * 5) / 4
    pyramid = np.maximum(0.0, 1.0 - np.maximum(np.abs(x), np.abs(y)))
    if "wave" in algorithm:
        data = waves
    elif "step" in algorithm or "stair" in algorithm:
        data = steps
    elif "pyramid" in algorithm:
        data = pyramid
    elif "uniform" in algorithm or "noise" in algorithm:
        data = noise
    else:
        data = 0.45 * noise + 0.35 * waves + 0.20 * np.maximum(steps, pyramid)
    data -= float(data.min())
    span = float(data.max())
    if span > 1.0e-8:
        data /= span
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def _training_objects(training_scene: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        scene_contract.normalize_scene_object(_select_object_variant(item), idx)
        for idx, item in enumerate(training_scene.get("objects", []))
    ]


def _select_object_variant(obj: dict[str, Any]) -> dict[str, Any]:
    """Training uses the first of an object's authored variants."""
    variants = obj.get("variants")
    if not variants:
        return obj
    selected = dict(obj)
    variant = variants[0]
    if "parts" not in variant:
        selected.pop("parts", None)
    for key in ("geometry", "rgba", "parts", "mass_kg", "friction", "physics"):
        if key in variant:
            selected[key] = variant[key]
    return selected


def _object_geometry(obj: dict[str, Any]) -> dict[str, Any]:
    geometry = obj.get("geometry")
    return geometry if isinstance(geometry, dict) else obj


def _object_shape(obj: dict[str, Any]) -> str:
    geometry = _object_geometry(obj)
    return str(geometry.get("shape") or geometry.get("type") or obj.get("shape") or "box").lower()


def _dynamic_joint_name(object_name: str) -> str:
    return f"scene_{object_name}_freejoint"


def _dynamic_body_id(model: mujoco.MjModel, object_name: str) -> int:
    return int(model.jnt_bodyid[int(model.joint(_dynamic_joint_name(object_name)).id)])


def _compiled_dynamic_objects(
    model: mujoco.MjModel, training_scene: dict[str, Any]
) -> tuple[CompiledDynamicObject, ...]:
    compiled: list[CompiledDynamicObject] = []
    for obj in _training_objects(training_scene):
        if obj["motion"] not in {"dynamic", "carried"}:
            continue
        joint_name = _dynamic_joint_name(obj["name"])
        qpos_ids = _freejoint_qpos_ids(model, joint_name)
        qvel_ids = _freejoint_qvel_ids(model, joint_name)
        initial = np.asarray(model.qpos0[qpos_ids], dtype=np.float64)
        compiled.append(
            CompiledDynamicObject(
                name=obj["name"],
                role=obj["role"],
                carried=obj["motion"] == "carried",
                qpos_ids=qpos_ids,
                qvel_ids=qvel_ids,
                initial_position=tuple(float(item) for item in initial[:3]),
                initial_orientation=tuple(float(item) for item in initial[3:7]),
                reset=obj.get("reset", {}),
            )
        )
    return tuple(compiled)


def _object_half_height(obj: dict[str, Any], shape: str) -> float:
    parts = obj.get("parts", [])
    if parts:
        top = 0.0
        for part in parts:
            geom_type = _geom_type(_object_shape(part))
            size = _geom_size(part, geom_type)
            half_z = (
                size[0]
                if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE
                else size[1] + (size[0] if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)
                if geom_type in {mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE}
                else size[2]
            )
            top = max(top, part["position_m"][2] + half_z)
        if top > 0.0:
            return top
    geom_type = _geom_type(shape)
    size = _geom_size(obj, geom_type)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return size[0]
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return size[1] + size[0]
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return size[1]
    return size[2]


def _attachment_local_pos(obj: dict[str, Any], spec: RobotSpec) -> tuple[float, float, float]:
    attachment = obj["attachment"]
    if "position_m" in attachment:
        return tuple(attachment["position_m"])
    pos = list(obj["pose"]["position_m"])
    pos_frame = str(attachment.get("pos_frame") or "").lower()
    if pos_frame in {"world", "global"} or (not pos_frame and pos[2] > 0.75 * spec.base_height):
        pos[2] -= spec.base_height
    return tuple(pos)


def _primitive_half_extents(geom_type: mujoco.mjtGeom, size: Any) -> tuple[float, float, float] | None:
    """Axis-aligned half extents of a primitive; None for meshes/hfields (unknown at spec time)."""
    x, y, z = (float(size[axis]) for axis in range(3))
    if geom_type in {mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID}:
        return (x, y, z)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return (x, x, x)
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return (x, x, y)
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return (x, x, y + x)
    return None


def _parent_support_surface(parent: Any, xy: tuple[float, float]) -> tuple[float, float, float] | None:
    """(top_z, half_x, half_y) of the highest primitive geom under ``parent`` containing ``xy``.

    Attached objects (payloads, sensors) must rest on the parent's surface; LLM-authored
    attachment heights frequently land inside the chassis hull, making them invisible.
    Rotations are ignored, which is exact for the axis-aligned deck geoms this targets.
    """
    best: tuple[float, float, float] | None = None

    def visit(body: Any, offset: tuple[float, float, float]) -> None:
        nonlocal best
        for geom in body.geoms:
            half = _primitive_half_extents(geom.type, geom.size)
            if half is None:
                continue
            gx, gy, gz = (offset[axis] + float(geom.pos[axis]) for axis in range(3))
            if abs(xy[0] - gx) <= half[0] and abs(xy[1] - gy) <= half[1]:
                top = gz + half[2]
                if best is None or top > best[0]:
                    best = (top, half[0], half[1])
        for child in body.bodies:
            visit(child, tuple(offset[axis] + float(child.pos[axis]) for axis in range(3)))

    visit(parent, (0.0, 0.0, 0.0))
    return best


def _object_half_footprint(obj: dict[str, Any], shape: str) -> tuple[float, float]:
    """Horizontal half extents of an object's authored geometry, including part offsets."""
    parts = obj.get("parts", [])
    if not parts:
        geom_type = _geom_type(shape)
        half = _primitive_half_extents(geom_type, _geom_size(obj, geom_type))
        return (half[0], half[1])
    half_x = half_y = 0.0
    for part in parts:
        geom_type = _geom_type(_object_shape(part))
        half = _primitive_half_extents(geom_type, _geom_size(part, geom_type))
        local = part["position_m"]
        half_x = max(half_x, abs(local[0]) + half[0])
        half_y = max(half_y, abs(local[1]) + half[1])
    return (half_x, half_y)


def _scale_object_geometry(obj: dict[str, Any], scale: float) -> dict[str, Any]:
    """Uniformly scale an object's authored metric geometry (and part offsets)."""

    def scale_keys(container: dict[str, Any], keys: tuple[str, ...]) -> None:
        for key in keys:
            value = container.get(key)
            if isinstance(value, float):
                container[key] = value * scale
            elif isinstance(value, list):
                container[key] = [item * scale for item in value]

    scaled = copy.deepcopy(obj)
    parts = scaled.get("parts", [])
    for entry in (scaled, *parts):
        if "geometry" in entry:
            scale_keys(entry["geometry"], ("dimensions_m", "radius_m", "length_m"))
    for part in parts:
        scale_keys(part, ("position_m",))
    return scaled


def _add_training_objects(mj_spec: mujoco.MjSpec, spec: RobotSpec, training_scene: dict[str, Any]) -> None:
    for obj in _training_objects(training_scene):
        _add_training_object(mj_spec, spec, obj)


def _add_training_object(mj_spec: mujoco.MjSpec, spec: RobotSpec, obj: dict[str, Any]) -> None:
    # Simulator names are namespaced to avoid collisions with robot assets; domain-randomization
    # selectors still match the contract name through aliases.
    name = f"scene_{obj['name']}"
    shape = _object_shape(obj)
    attached = obj["motion"] == "attached"
    carried = obj["motion"] == "carried"
    if carried and spec.payload_max_footprint is not None:
        half_x, half_y = _object_half_footprint(obj, shape)
        max_x, max_y = spec.payload_max_footprint
        scale = min(1.0, 0.5 * max_x / max(half_x, 1.0e-6), 0.5 * max_y / max(half_y, 1.0e-6))
        if scale < 1.0:
            obj = _scale_object_geometry(obj, scale)
    parent = mj_spec.body(obj["attachment"]["parent_body"]) if attached else mj_spec.worldbody
    body_pos = _attachment_local_pos(obj, spec) if attached else obj["pose"]["position_m"]
    if carried and spec.payload_support_position is not None:
        support_x, support_y, support_z = spec.payload_support_position
        body_pos = (support_x, support_y, support_z + _object_half_height(obj, shape) + 0.002)
    if attached:
        # The scene compiler owns sizing and placement; this only enforces two physical invariants
        # no prompt can guarantee: cargo fits its carrier, and it rests on the support surface
        # instead of inside the parent's hull.
        support = _parent_support_surface(parent, (body_pos[0], body_pos[1]))
        if support is not None:
            top_z, support_x, support_y = support
            half_x, half_y = _object_half_footprint(obj, shape)
            scale = min(
                1.0,
                ATTACHED_OVERHANG_FACTOR * support_x / max(half_x, 1.0e-6),
                ATTACHED_OVERHANG_FACTOR * support_y / max(half_y, 1.0e-6),
            )
            if scale < 1.0:
                obj = _scale_object_geometry(obj, scale)
            body_pos = (body_pos[0], body_pos[1], max(body_pos[2], top_z + _object_half_height(obj, shape)))
    body = parent.add_body(name=name, pos=body_pos, quat=obj["pose"]["orientation_wxyz"])
    if obj["motion"] in {"dynamic", "carried"}:
        joint = body.add_freejoint()
        joint.name = _dynamic_joint_name(obj["name"])
    mesh_path = _mesh_path(obj)
    rgba = obj.get("rgba", DEFAULT_OBJECT_RGBA)
    collidable = obj["collidable"]
    collision = obj["collision"]
    geom_name = name
    if mesh_path is not None:
        mesh_name = f"{geom_name}_mesh"
        mesh = mj_spec.add_mesh(name=mesh_name, file=str(mesh_path))
        mesh.scale = _float_tuple(obj.get("scale"), (1.0, 1.0, 1.0), length=3)
        body.add_geom(
            name=f"{geom_name}_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            rgba=rgba,
            mass=0.0 if collision != "mesh" else obj.get("mass_kg", DEFAULT_OBJECT_MASS_KG),
            **_contact_kwargs(obj, collidable and collision == "mesh", spec),
        )
        if collision == "mesh":
            _apply_object_physics(body, obj)
            return
    parts = obj.get("parts")
    if parts:
        collidable_parts = [part for part in parts if part["collidable"]]
        total_mass = obj.get("mass_kg", DEFAULT_OBJECT_MASS_KG)
        default_part_mass = total_mass / max(1, len(collidable_parts))
        for part_idx, part in enumerate(parts):
            part_obj = dict(obj)
            part_obj.update(part)
            part_obj["mass_kg"] = default_part_mass
            _add_object_geom(
                body,
                spec,
                part_obj,
                f"{geom_name}_{part_idx}_{part['name']}",
                _object_shape(part),
                part.get("rgba") or rgba,
                collidable=collidable and part["collidable"],
                pos=part["position_m"],
            )
        _apply_object_physics(body, obj)
        return
    _add_object_geom(body, spec, obj, geom_name, shape, rgba, collidable=collidable)
    _apply_object_physics(body, obj)


def _apply_object_physics(body: Any, obj: dict[str, Any]) -> None:
    if "physics" not in obj:
        return
    mass = obj.get("mass_kg", DEFAULT_OBJECT_MASS_KG)
    half_x, half_y = _object_half_footprint(obj, _object_shape(obj))
    half_z = max(1.0e-3, _object_half_height(obj, _object_shape(obj)))
    half_x, half_y = max(1.0e-3, half_x), max(1.0e-3, half_y)
    body.explicitinertial = True
    body.mass = mass
    body.ipos = obj["physics"]["center_of_mass_offset_m"]
    body.iquat = (1.0, 0.0, 0.0, 0.0)
    body.inertia = (
        mass * (half_y * half_y + half_z * half_z) / 3.0,
        mass * (half_x * half_x + half_z * half_z) / 3.0,
        mass * (half_x * half_x + half_y * half_y) / 3.0,
    )


def _mesh_path(obj: dict[str, Any]) -> Path | None:
    """The object's mesh file; a missing file falls back to the primitive geometry."""
    value = obj.get("mesh_path") or obj.get("asset_path") or obj.get("file")
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path if path.is_file() else None


def _add_object_geom(
    body: Any,
    spec: RobotSpec,
    obj: dict[str, Any],
    name: str,
    shape: str,
    rgba: tuple[float, float, float, float],
    *,
    collidable: bool,
    pos: tuple[float, float, float] | None = None,
) -> None:
    geom_type = _geom_type(shape)
    moving = obj["motion"] in {"dynamic", "carried"}
    kwargs = {
        "name": f"{name}_collider" if collidable else f"{name}_visual",
        "type": geom_type,
        "quat": (1.0, 0.0, 0.0, 0.0),
        "size": _geom_size(obj, geom_type),
        "mass": obj.get("mass_kg", DEFAULT_OBJECT_MASS_KG) if collidable or moving else 0.0,
        "rgba": rgba,
        **_contact_kwargs(obj, collidable, spec),
    }
    if pos is not None:
        kwargs["pos"] = pos
    body.add_geom(**kwargs)


def _geom_type(shape: str) -> mujoco.mjtGeom:
    return {
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    }.get(shape, mujoco.mjtGeom.mjGEOM_BOX)


def _geom_size(value: dict[str, Any], geom_type: mujoco.mjtGeom) -> tuple[float, float, float]:
    raw = scene_contract.Primitive.from_mapping(_object_geometry(value)).dimensions_m
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return (raw[0], raw[0], raw[0])
    if geom_type in {mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE}:
        half_height = 0.5 * raw[1] if len(raw) > 1 else 0.10
        return (raw[0], half_height, 0.0)
    dimensions = _float_tuple(raw, DEFAULT_OBJECT_SIZE, length=3)
    return tuple(0.5 * item for item in dimensions)  # type: ignore[return-value]


def _contact_kwargs(obj: dict[str, Any], collidable: bool, spec: RobotSpec) -> dict[str, Any]:
    if not collidable:
        return {"contype": 0, "conaffinity": 0}
    payload = obj["motion"] == "carried"
    contact_type = spec.payload_contact_type if payload else 1
    return {
        "contype": contact_type,
        "conaffinity": contact_type,
        "friction": _float_tuple(obj.get("friction"), (1.1, 0.02, 0.002), length=3),
        "condim": 4 if payload else 3,
    }


def _add_position_actuators(mj_spec: mujoco.MjSpec, robot_spec: RobotSpec) -> None:
    for joint_name, (kp, kv, effort_limit, _armature) in zip(
        robot_spec.joint_names, robot_spec.position_actuator_parameters, strict=True
    ):
        actuator = mj_spec.add_actuator()
        actuator.name = f"{joint_name}_position"
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.target = joint_name
        actuator.set_to_position(kp, kv)
        actuator.inheritrange = 0.9
        actuator.forcelimited = True
        actuator.forcerange = (-effort_limit, effort_limit)


def _set_spec_capacity(mj_spec: mujoco.MjSpec, spec: RobotSpec) -> None:
    mj_spec.nconmax = spec.nconmax
    mj_spec.njmax = spec.njmax


def robot_spec_with_capacity(
    spec: RobotSpec, *, nconmax: int | None = None, naconmax: int | None = None, njmax: int | None = None
) -> RobotSpec:
    updates = {
        key: value
        for key, value in (("nconmax", nconmax), ("naconmax", naconmax), ("njmax", njmax))
        if value is not None
    }
    return replace(spec, **updates)


def _slots_per_world(capacity: int) -> int:
    return (capacity + CONTACT_CAPACITY_REFERENCE_ENVS - 1) // CONTACT_CAPACITY_REFERENCE_ENVS


def training_contact_capacity(robot: str, num_envs: int) -> dict[str, int]:
    """Scale a robot's tested per-world Warp capacity to a training batch."""
    spec = ROBOT_SPECS[robot]
    contact_slots = max(1, _slots_per_world(spec.nconmax))
    if spec.naconmax is not None:
        broadphase_slots = max(contact_slots, _slots_per_world(spec.naconmax))
    else:
        broadphase_slots = 3 * contact_slots
    return {
        "num_worlds": num_envs,
        "nconmax": contact_slots * num_envs,
        "naconmax": broadphase_slots * num_envs,
        "njmax": spec.njmax,
    }


def _compile_locomotion_model(spec: RobotSpec, training_scene: dict[str, Any]) -> mujoco.MjModel:
    mj_spec = mujoco.MjSpec.from_file(str(spec.xml_path))
    _set_spec_capacity(mj_spec, spec)
    if not _add_training_terrain(mj_spec, training_scene):
        _add_flat_floor(mj_spec)
    if spec.position_actuator_parameters:
        _add_position_actuators(mj_spec, spec)
    _add_training_objects(mj_spec, spec, training_scene)
    model = mj_spec.compile()
    model.opt.timestep = spec.sim_dt
    _apply_position_actuator_parameters(model, spec)
    _configure_floor_body_contacts(model, spec)
    # Disabled collision proxies may still touch scene objects, not the floor
    # or each other (e.g. the humanoid's hands and torso).
    for geom_id in range(model.ngeom):
        if model.geom_group[geom_id] == 3 and not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            model.geom_contype[geom_id] = 8
        if model.geom(geom_id).name.startswith("scene_") and model.geom_contype[geom_id]:
            model.geom_conaffinity[geom_id] |= 8
    return model


def _apply_position_actuator_parameters(model: mujoco.MjModel, spec: RobotSpec) -> None:
    if not spec.position_actuator_parameters:
        return
    for joint_name, (_kp, _kv, effort_limit, armature) in zip(
        spec.joint_names, spec.position_actuator_parameters, strict=True
    ):
        joint = model.joint(joint_name)
        joint_id = int(joint.id)
        model.dof_armature[int(joint.dofadr[0])] = armature
        model.jnt_actfrcrange[joint_id] = (-effort_limit, effort_limit)
        model.jnt_actfrclimited[joint_id] = 1
        actuator_id = int(model.actuator(f"{joint_name}_position").id)
        model.actuator_forcerange[actuator_id] = (-effort_limit, effort_limit)
        model.actuator_forcelimited[actuator_id] = 1


def _configure_floor_body_contacts(model: mujoco.MjModel, spec: RobotSpec) -> None:
    """Apply an asset-declared ground-contact allowlist while avoiding self-collision."""
    if not spec.ground_contact_geom_tokens:
        return
    scene_contact_geoms = {
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom(geom_id).name.startswith("scene_")
        and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
    }
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if name == "floor":
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 1
        elif geom_id in scene_contact_geoms:
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 1
            model.geom_condim[geom_id] = max(int(model.geom_condim[geom_id]), 3)
        elif any(token in name for token in spec.ground_contact_geom_tokens):
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 0
            if spec.foot_contact_geom_token and spec.foot_contact_geom_token in name:
                model.geom_condim[geom_id] = 3
                model.geom_friction[geom_id, 0] = 0.6
                model.geom_solimp[geom_id, :3] = (0.9, 0.95, 0.023)
            else:
                model.geom_condim[geom_id] = 1


def _compile_hand_model(spec: RobotSpec, training_scene: dict[str, Any]) -> mujoco.MjModel:
    hand = mujoco.MjSpec.from_file(str(spec.xml_path))
    _set_spec_capacity(hand, spec)
    has_manipulated_object = any(
        obj["motion"] in {"dynamic", "carried"} and obj["role"] in scene_contract.MANIPULATED_ROLES
        for obj in _training_objects(training_scene)
    )
    if not has_manipulated_object:
        raise ValueError(f"{spec.name} manipulation requires one LLM-authored manipulated object")
    _add_flat_floor(hand, z=-0.25)
    _add_training_objects(hand, spec, training_scene)
    model = hand.compile()
    model.opt.timestep = spec.sim_dt
    return model


def _body_is_static(model: mujoco.MjModel, body_id: int) -> bool:
    while body_id != 0:
        if model.body_jntnum[body_id] > 0:
            return False
        body_id = int(model.body_parentid[body_id])
    return True


def _hfield_top_near(model: mujoco.MjModel, geom_id: int, pos: np.ndarray, spawn_radius: float) -> float:
    hfield_id = int(model.geom_dataid[geom_id])
    nrow = int(model.hfield_nrow[hfield_id])
    ncol = int(model.hfield_ncol[hfield_id])
    size_x, size_y, z_top, _ = (float(value) for value in model.hfield_size[hfield_id])
    adr = int(model.hfield_adr[hfield_id])
    heights = np.asarray(model.hfield_data[adr : adr + nrow * ncol], dtype=np.float64).reshape(nrow, ncol)
    xs = np.linspace(-size_x, size_x, ncol) + float(pos[0])
    ys = np.linspace(-size_y, size_y, nrow) + float(pos[1])
    col_mask = np.abs(xs) <= spawn_radius
    row_mask = np.abs(ys) <= spawn_radius
    if not col_mask.any() or not row_mask.any():
        return 0.0
    return float(pos[2]) + float(heights[np.ix_(row_mask, col_mask)].max()) * z_top


def _geom_world_half_extents(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> np.ndarray:
    """World-axis-aligned half extents (hx, hy, hz) of a primitive geom."""
    geom_type = int(model.geom_type[geom_id])
    mat = data.geom_xmat[geom_id].reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.full(3, float(size[0]))
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return np.abs(mat) @ size
    if geom_type in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        return np.abs(mat[:, 2]) * size[1] + size[0]
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return np.linalg.norm(mat * size[None, :], axis=1)
    return np.full(3, float(model.geom_rbound[geom_id]))


def spawn_surface_height(model: mujoco.MjModel, *, spawn_radius: float = SPAWN_RADIUS_M) -> float:
    """Highest static collidable surface under the spawn area (world origin).

    Generated scenes can raise the support surface (terrain elevation, platforms, stepping
    stones). Reset code spawns the robot base at ``base_height`` above this value so the robot
    never starts intersecting scene geometry, whatever the scene generator authored. Only
    geometry whose footprint overlaps the spawn area counts, so distant props and walls do not
    inflate the spawn height.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    top = 0.0
    for geom_id in range(model.ngeom):
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        if not _body_is_static(model, int(model.geom_bodyid[geom_id])):
            continue
        geom_type = int(model.geom_type[geom_id])
        pos = data.geom_xpos[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if geom_type == mujoco.mjtGeom.mjGEOM_HFIELD:
            top = max(top, _hfield_top_near(model, geom_id, pos, spawn_radius))
            continue
        half = _geom_world_half_extents(model, data, geom_id)
        if abs(float(pos[0])) > spawn_radius + float(half[0]) or abs(float(pos[1])) > spawn_radius + float(half[1]):
            continue
        top = max(top, float(pos[2]) + float(half[2]))
    return top


@lru_cache(maxsize=4)
def compile_robot_model(
    robot: str,
    *,
    impl: str = "warp",
    training_scene_key: str | None = None,
    nconmax: int | None = None,
    naconmax: int | None = None,
    njmax: int | None = None,
) -> CompiledRobotModel:
    """Compile the robot with its generated training scene, plus the index maps the envs need."""
    spec = robot_spec_with_capacity(ROBOT_SPECS[robot], nconmax=nconmax, naconmax=naconmax, njmax=njmax)
    training_scene = _decode_training_scene(training_scene_key)
    if spec.kind == "hand":
        model = _compile_hand_model(spec, training_scene)
    else:
        model = _compile_locomotion_model(spec, training_scene)
    dynamic_objects = _compiled_dynamic_objects(model, training_scene)
    mx_model = mjx.put_model(model, impl=impl)
    qpos_ids = _joint_qpos_ids(model, spec.joint_names)
    qvel_ids = _joint_qvel_ids(model, spec.joint_names)
    manipulated_qpos_ids = None
    manipulated_qvel_ids = None
    manipulated_body_id = None
    palm_site_id = None
    if spec.kind == "hand":
        manipulated_object = next(item for item in dynamic_objects if item.role in scene_contract.MANIPULATED_ROLES)
        manipulated_qpos_ids = manipulated_object.qpos_ids
        manipulated_qvel_ids = manipulated_object.qvel_ids
        manipulated_body_id = _dynamic_body_id(model, manipulated_object.name)
        palm_site_id = int(model.site("grasp_site").id)
    carried_object = next((item for item in dynamic_objects if item.carried), None)
    payload_body_id = _dynamic_body_id(model, carried_object.name) if carried_object is not None else None
    root_body_id = int(model.body(spec.freebase_root_body).id) if spec.freebase_root_body else 1
    return CompiledRobotModel(
        mj_model=model,
        mjx_model=mx_model,
        qpos_ids=qpos_ids,
        qvel_ids=qvel_ids,
        manipulated_qpos_ids=manipulated_qpos_ids,
        manipulated_qvel_ids=manipulated_qvel_ids,
        root_body_id=root_body_id,
        manipulated_body_id=manipulated_body_id,
        palm_site_id=palm_site_id,
        xml_path=str(spec.xml_path),
        payload_body_id=payload_body_id,
        spawn_surface_height=spawn_surface_height(model),
        dynamic_objects=dynamic_objects,
    )

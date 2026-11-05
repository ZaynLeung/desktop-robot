"""恐龙模型在 MuJoCo 里的加载、站姿、新关节动作与短时稳定性。

没有安装 MuJoCo 时整个文件被跳过，静态测试仍然可用。
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="未安装 mujoco，跳过恐龙模型动态校验")

from desktop_robot import sim  # noqa: E402
from desktop_robot.builder import MODELS_DIR  # noqa: E402
from desktop_robot.dino import DEFAULT_DINO_SPEC, solve_base_z  # noqa: E402

_SIGNS = np.array(
    [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=float,
)


def _box_corners(pos: np.ndarray, mat: np.ndarray, half: np.ndarray) -> np.ndarray:
    """geom 的 8 个顶点（世界系）；``mat`` 是 3x3 旋转，``half`` 是半尺寸。"""
    return np.asarray(pos) + (_SIGNS * np.asarray(half)) @ np.asarray(mat).T


def _fresh_data(model):
    """从 stand 关键帧复位（零位=站姿），返回 data。"""
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return data


def _set_joint(model, data, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def _body_pos(model, data, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body_id].copy()


@pytest.fixture(scope="module")
def scene_model():
    return mujoco.MjModel.from_xml_path(str(MODELS_DIR / "dino_scene.xml"))


def test_dino_model_loads_standalone():
    model = mujoco.MjModel.from_xml_path(str(MODELS_DIR / "dino.xml"))
    assert model.nq == DEFAULT_DINO_SPEC.nq
    assert model.nv == DEFAULT_DINO_SPEC.nv
    assert model.nu == DEFAULT_DINO_SPEC.nu


def test_dino_scene_dimensions(scene_model):
    assert scene_model.nq == DEFAULT_DINO_SPEC.nq
    assert scene_model.nv == DEFAULT_DINO_SPEC.nv
    assert scene_model.nu == DEFAULT_DINO_SPEC.nu
    assert scene_model.nbody == DEFAULT_DINO_SPEC.n_bodies + 1  # +1 是 world


def test_dino_scene_has_ground_lights_and_cameras(scene_model):
    assert mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    for camera in ("front", "iso", "ref"):
        assert (
            mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_CAMERA, camera) >= 0
        ), camera
    assert mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_KEY, "stand") >= 0


def test_dino_actuators_cover_all_joints(scene_model):
    for name in DEFAULT_DINO_SPEC.joint_names:
        joint_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id > 0, name
        assert scene_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE


def test_dino_stand_keyframe_matches_kinematics(scene_model):
    data = _fresh_data(scene_model)
    assert data.qpos[2] == pytest.approx(solve_base_z(DEFAULT_DINO_SPEC), abs=1e-6)

    # 足 box 与两个前趾都要落在地面上（顶点最低点 z ≈ 0，不依赖是否已产生接触）
    for side in ("L", "R"):
        for geom_name in (
            f"foot_geom_{side}",
            f"toe_geom_{side}1",
            f"toe_geom_{side}2",
        ):
            geom_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            assert geom_id >= 0, geom_name
            corners = _box_corners(
                data.geom_xpos[geom_id],
                data.geom_xmat[geom_id].reshape(3, 3),
                scene_model.geom_size[geom_id],
            )
            assert corners[:, 2].min() == pytest.approx(0.0, abs=1e-5)


def test_new_head_yaw_turns_the_head(scene_model):
    data = _fresh_data(scene_model)
    head_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, "head")
    before = data.xmat[head_id].reshape(3, 3)[:, 0].copy()
    _set_joint(scene_model, data, "neck_yaw", 0.30)
    after = data.xmat[head_id].reshape(3, 3)[:, 0]
    assert after[1] > 0.15, "头的前向轴没有转向 +y（左）"
    assert not np.allclose(before, after)


def test_new_hip_roll_is_outward_on_both_sides(scene_model):
    data = _fresh_data(scene_model)
    foot_l0 = _body_pos(scene_model, data, "foot_L")
    foot_r0 = _body_pos(scene_model, data, "foot_R")

    _set_joint(scene_model, data, "hip_roll_L", 0.30)
    assert _body_pos(scene_model, data, "foot_L")[1] > foot_l0[1] + 0.02

    data = _fresh_data(scene_model)
    _set_joint(scene_model, data, "hip_roll_R", 0.30)
    assert _body_pos(scene_model, data, "foot_R")[1] < foot_r0[1] - 0.02


def test_new_tail_yaw_swings_the_tail_sideways(scene_model):
    data = _fresh_data(scene_model)
    tail0 = _body_pos(scene_model, data, "tail_4")

    _set_joint(scene_model, data, "tail1_yaw", 0.30)
    moved = _body_pos(scene_model, data, "tail_4") - tail0
    assert moved[1] < -0.02, "尾根 yaw 正角应该把尾尖带向 -y（右）"
    assert abs(moved[2]) < 0.02

    data = _fresh_data(scene_model)
    _set_joint(scene_model, data, "tail3_yaw", 0.30)
    moved = _body_pos(scene_model, data, "tail_4") - tail0
    assert moved[1] < -0.01


def test_new_tail_pitch_bends_the_tail_down(scene_model):
    data = _fresh_data(scene_model)
    tail0 = _body_pos(scene_model, data, "tail_4")

    _set_joint(scene_model, data, "tail2_pitch", 0.20)
    moved = _body_pos(scene_model, data, "tail_4") - tail0
    assert moved[2] < -0.01, "尾 2 pitch 正角应该把尾巴压下去"

    # 尾 4 的关节在它自己 body 的原点上，body 原点不动，改看尾尖几何
    data = _fresh_data(scene_model)
    geom_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_GEOM, "tail4_geom")
    before = data.geom_xpos[geom_id].copy()
    _set_joint(scene_model, data, "tail4_pitch", 0.20)
    assert not np.allclose(before, data.geom_xpos[geom_id])


def test_dino_holding_stand_keeps_base_height():
    stats = sim.run_headless(MODELS_DIR / "dino_scene.xml", steps=500)
    assert abs(stats["z_drift"]) < 0.02, f"基座漂移 {stats['z_drift']:+.4f} m"
    assert stats["z_min"] > 0.15

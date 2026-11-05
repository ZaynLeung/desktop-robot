"""恐龙模型在 MuJoCo 里的加载、站姿与短时稳定性。

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
    data = mujoco.MjData(scene_model)
    key_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(scene_model, data, key_id)
    mujoco.mj_forward(scene_model, data)
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


def test_dino_holding_stand_keeps_base_height():
    stats = sim.run_headless(MODELS_DIR / "dino_scene.xml", steps=500)
    assert abs(stats["z_drift"]) < 0.02, f"基座漂移 {stats['z_drift']:+.4f} m"
    assert stats["z_min"] > 0.15

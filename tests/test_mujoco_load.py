"""MuJoCo 加载与短时仿真。

没有安装 MuJoCo 时整个文件被跳过，所以 ``build`` / ``verify`` / 数学测试
在裸环境下依然全绿；装了 MuJoCo 之后这些用例就是"模型真的能用"的证据。
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco", reason="未安装 mujoco，跳过动态校验")

from desktop_robot import sim  # noqa: E402
from desktop_robot.builder import MODELS_DIR  # noqa: E402
from desktop_robot.config import DEFAULT_SPEC  # noqa: E402
from desktop_robot.kinematics import solve_pelvis_z  # noqa: E402


@pytest.fixture(scope="module")
def scene_model():
    return mujoco.MjModel.from_xml_path(str(MODELS_DIR / "scene.xml"))


def test_robot_model_loads_standalone():
    """``biped.xml`` 本身就要是合法模型，可以脱离场景单独加载。"""
    model = mujoco.MjModel.from_xml_path(str(MODELS_DIR / "biped.xml"))
    assert model.nq == DEFAULT_SPEC.nq
    assert model.nv == DEFAULT_SPEC.nv
    assert model.nu == DEFAULT_SPEC.nu


def test_scene_has_expected_dimensions(scene_model):
    assert scene_model.nq == DEFAULT_SPEC.nq
    assert scene_model.nv == DEFAULT_SPEC.nv
    assert scene_model.nu == DEFAULT_SPEC.nu
    assert scene_model.nbody == DEFAULT_SPEC.n_bodies + 1  # +1 是 world


def test_scene_has_ground_and_lights(scene_model):
    assert mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    assert mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_CAMERA, "front") >= 0
    assert mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_KEY, "stand") >= 0


def test_actuators_cover_all_joints(scene_model):
    assert scene_model.nu == 9
    for name in DEFAULT_SPEC.joint_names:
        joint_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id > 0, name
        assert scene_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE


def test_stand_keyframe_puts_feet_on_the_ground(scene_model):
    data = mujoco.MjData(scene_model)
    key_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(scene_model, data, key_id)
    mujoco.mj_forward(scene_model, data)

    assert data.qpos[2] == pytest.approx(solve_pelvis_z(DEFAULT_SPEC), abs=1e-6)
    # 每只脚至少有一个接触点落在地面上（接触点 z 接近 0）
    foot_geom_ids = {
        mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_geom_{side}")
        for side in ("L", "R")
    }
    touching = {
        contact.geom1 if contact.geom2 == 0 else contact.geom2
        for contact in data.contact[: data.ncon]
    }
    assert foot_geom_ids & touching, "站立关键帧下两只脚都没有接触地面"


def test_holding_the_stand_pose_does_not_collapse():
    """位置源保持站姿 1 秒，基座不应该明显下坠。"""
    stats = sim.run_headless(MODELS_DIR / "scene.xml", steps=500)
    assert abs(stats["z_drift"]) < 0.02, f"基座漂移 {stats['z_drift']:+.4f} m"
    assert stats["z_min"] > 0.15


def test_dynamic_check_reports_no_problems():
    result = sim.dynamic_check(MODELS_DIR / "scene.xml", steps=500)
    assert result["problems"] == []

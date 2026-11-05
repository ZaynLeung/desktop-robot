"""恐龙机型（dino）的参数自洽性、生成物确定性与"长度可调"的传导。"""

from __future__ import annotations

from dataclasses import replace
import math
import xml.etree.ElementTree as ET

import pytest

from desktop_robot.builder import MODELS_DIR, vec
from desktop_robot.config import joint_name
from desktop_robot.dino import (
    DEFAULT_DINO_SPEC,
    head_top_z,
    render_dino_scene_xml,
    render_dino_xml,
    sole_lowest_z,
    solve_base_z,
    verify_dino_static,
)

#: 期望的关节顺序（= body 深度优先遍历 + body 内关节定义顺序）。
EXPECTED_JOINTS = [
    "neck_pitch",
    "neck_yaw",
    "shoulder_pitch_L",
    "shoulder_pitch_R",
    "hip_pitch_L",
    "hip_roll_L",
    "hip_yaw_L",
    "knee_L",
    "ankle_L",
    "hip_pitch_R",
    "hip_roll_R",
    "hip_yaw_R",
    "knee_R",
    "ankle_R",
    "tail1_yaw",
    "tail2_pitch",
    "tail3_yaw",
    "tail4_pitch",
]

#: 期望的轴：俯仰 (0,-1,0)；偏航 (0,0,1)；内外翻 x 镜像、脚尖偏航 z 镜像。
EXPECTED_AXES = {
    "neck_pitch": (0.0, -1.0, 0.0),
    "neck_yaw": (0.0, 0.0, 1.0),
    "shoulder_pitch_L": (0.0, -1.0, 0.0),
    "shoulder_pitch_R": (0.0, -1.0, 0.0),
    "hip_pitch_L": (0.0, -1.0, 0.0),
    "hip_pitch_R": (0.0, -1.0, 0.0),
    "hip_roll_L": (1.0, 0.0, 0.0),
    "hip_roll_R": (-1.0, 0.0, 0.0),
    "hip_yaw_L": (0.0, 0.0, 1.0),
    "hip_yaw_R": (0.0, 0.0, -1.0),
    "knee_L": (0.0, -1.0, 0.0),
    "knee_R": (0.0, -1.0, 0.0),
    "ankle_L": (0.0, -1.0, 0.0),
    "ankle_R": (0.0, -1.0, 0.0),
    "tail1_yaw": (0.0, 0.0, 1.0),
    "tail2_pitch": (0.0, -1.0, 0.0),
    "tail3_yaw": (0.0, 0.0, 1.0),
    "tail4_pitch": (0.0, -1.0, 0.0),
}


def test_dof_counts():
    spec = DEFAULT_DINO_SPEC
    assert spec.nq == 25  # 7 自由基座 + 18 个驱动关节
    assert spec.nv == 24
    assert spec.nu == 18
    assert spec.n_bodies == 14  # 每条腿的 3 个关节都挂在 thigh body 上，不新增 body
    assert len(spec.joint_names) == 18


def test_joint_order_matches_mjcf_traversal():
    """qpos 顺序必须与 body 的深度优先遍历一致。"""
    assert DEFAULT_DINO_SPEC.joint_names == EXPECTED_JOINTS


def test_joint_axes():
    """18 个关节的轴：俯仰 / 偏航 / 左右镜像的内外翻与脚尖偏航。"""
    for joint, side in DEFAULT_DINO_SPEC.actuated():
        name = joint_name(joint, side)
        axis = tuple(joint.axis_for(side))
        assert axis == EXPECTED_AXES[name], name
        assert math.dist(axis, (0.0, 0.0, 0.0)) == pytest.approx(1.0)


def test_ranges_are_valid_and_contain_stand_pose():
    spec = DEFAULT_DINO_SPEC
    stand = spec.stand_q()
    for joint, side in spec.actuated():
        low, high = joint.range
        assert low < high, f"{joint.name} 的 range 非法"
        assert low <= stand[joint_name(joint, side)] <= high, joint.name


def test_stand_pose_puts_both_feet_on_the_ground():
    spec = DEFAULT_DINO_SPEC
    base = solve_base_z(spec)
    for side in ("L", "R"):
        assert base + sole_lowest_z(spec, side) == pytest.approx(0.0, abs=1e-9)


def test_stand_pose_is_plausible():
    spec = DEFAULT_DINO_SPEC
    height = head_top_z(spec)
    assert solve_base_z(spec) > 0.1
    assert 0.25 < height < 0.45
    assert height > solve_base_z(spec)  # 头顶一定在基座之上


def test_mass_is_plausible_for_a_desktop_robot():
    mass = DEFAULT_DINO_SPEC.total_mass()
    assert 0.5 < mass < 6.0, f"总质量 {mass:.3f} kg 不像桌面机器人"


def test_rendered_structure():
    spec = DEFAULT_DINO_SPEC
    root = ET.fromstring(render_dino_xml(spec))
    assert root.get("model") == spec.name

    compilers = root.findall("compiler")
    assert len(compilers) == 1
    assert compilers[0].get("angle") == "radian"

    bodies = root.findall(".//body")
    assert len(bodies) == spec.n_bodies
    assert len({b.get("name") for b in bodies}) == len(bodies)

    joints = root.findall(".//joint")
    assert [j.get("name") for j in joints] == spec.joint_names
    assert len(root.find("actuator").findall("position")) == spec.nu

    key = root.find("keyframe/key")
    assert key is not None
    qpos = [float(v) for v in key.get("qpos").split()]
    assert len(qpos) == spec.nq
    ctrl = [float(v) for v in key.get("ctrl").split()]
    assert len(ctrl) == spec.nu
    assert math.sqrt(sum(v * v for v in qpos[3:7])) == pytest.approx(1.0)
    assert qpos[2] == pytest.approx(solve_base_z(spec), abs=1e-6)
    # 零位即站姿：关键帧的关节角和 ctrl 全是 0
    assert all(v == 0.0 for v in qpos[7:])
    assert all(v == 0.0 for v in ctrl)


def test_head_camera_is_mounted_on_the_head_body():
    """``cam_head`` 必须写在 head body 里，这样它会随颈 pitch / yaw 一起转。"""
    spec = DEFAULT_DINO_SPEC
    root = ET.fromstring(render_dino_xml(spec))
    head = root.find(".//body[@name='head']")
    assert head is not None
    camera = head.find("camera[@name='cam_head']")
    assert camera is not None, "head body 里缺少 cam_head 相机"
    assert camera.get("pos") == vec(spec.head_camera_pos)
    assert camera.get("xyaxes") == spec.head_camera_xyaxes
    assert float(camera.get("fovy")) == pytest.approx(spec.head_camera_fovy)


def test_tail_chain_nesting():
    """4 节尾巴必须串成 tail_1 → tail_2 → tail_3 → tail_4，不能都挂在躯干上。"""
    root = ET.fromstring(render_dino_xml(DEFAULT_DINO_SPEC))
    parent = root.find(".//body[@name='tail_1']")
    assert parent is not None
    for index in (2, 3, 4):
        child = parent.find(f"body[@name='tail_{index}']")
        assert child is not None, f"tail_{index} 不是 tail_{index - 1} 的子 body"
        parent = child


def test_every_joint_has_a_bearing_cap():
    root = ET.fromstring(render_dino_xml(DEFAULT_DINO_SPEC))
    cap_names = {
        g.get("name")
        for g in root.findall(".//geom")
        if (g.get("name") or "").startswith("cap_")
    }
    assert cap_names == {f"cap_{name}" for name in DEFAULT_DINO_SPEC.joint_names}


def test_scene_has_include_ground_and_cameras():
    root = ET.fromstring(render_dino_scene_xml(DEFAULT_DINO_SPEC))
    assert root.find("include").get("file") == "dino.xml"
    assert root.find("compiler") is None
    assert root.find(".//geom[@name='floor']") is not None
    camera_names = {c.get("name") for c in root.findall(".//camera")}
    # cam_head 写在本体 dino.xml 的 head body 里（include 之后才会出现在场景中）
    assert {"front", "iso", "ref"} <= camera_names


def test_rendering_is_deterministic():
    assert render_dino_xml(DEFAULT_DINO_SPEC) == render_dino_xml(DEFAULT_DINO_SPEC)
    assert render_dino_scene_xml(DEFAULT_DINO_SPEC) == render_dino_scene_xml(
        DEFAULT_DINO_SPEC
    )


@pytest.mark.parametrize(
    ("field", "value", "body_name", "attr", "expected"),
    [
        ("thigh_size", (0.048, 0.040, 0.120), "shank_L", "pos", "0 0 -0.12"),
        ("shank_size", (0.044, 0.036, 0.100), "foot_R", "pos", "0 0 -0.1"),
        ("tail1_size", (0.120, 0.070, 0.086), "tail_2", "pos", "-0.12 0 0"),
        ("tail2_size", (0.100, 0.065, 0.074), "tail_3", "pos", "-0.1 0 0"),
        ("tail3_size", (0.090, 0.061, 0.062), "tail_4", "pos", "-0.09 0 0"),
    ],
)
def test_link_length_changes_propagate(field, value, body_name, attr, expected):
    """改一个 link 的长度，下游关节/几何的位置必须自动跟着变。"""
    spec = replace(DEFAULT_DINO_SPEC, **{field: value})
    root = ET.fromstring(render_dino_xml(spec))
    node = root.find(f".//body[@name='{body_name}']")
    assert node is not None
    assert node.get(attr) == expected


def test_arm_and_tail_geometry_tracks_their_size():
    spec = replace(
        DEFAULT_DINO_SPEC,
        arm_size=(0.045, 0.032, 0.090),
        tail4_size=(0.120, 0.056, 0.050),
    )
    root = ET.fromstring(render_dino_xml(spec))
    arm_geom = root.find(".//geom[@name='arm_geom_L']")
    assert arm_geom.get("pos") == "0 0 -0.045"  # 臂几何中心 = −臂长/2
    tail4_geom = root.find(".//geom[@name='tail4_geom']")
    assert tail4_geom.get("pos") == "-0.06 0 0"  # 尾 4 几何中心 = −尾长/2


def test_committed_dino_models_are_up_to_date():
    robot = MODELS_DIR / "dino.xml"
    scene = MODELS_DIR / "dino_scene.xml"
    assert robot.exists(), "缺少 models/dino.xml，请运行 python -m desktop_robot build-dino"
    assert scene.exists(), "缺少 models/dino_scene.xml，请运行 python -m desktop_robot build-dino"
    assert robot.read_text(encoding="utf-8") == render_dino_xml(DEFAULT_DINO_SPEC), (
        "models/dino.xml 与当前参数源不一致，请运行：python -m desktop_robot build-dino"
    )
    assert scene.read_text(encoding="utf-8") == render_dino_scene_xml(DEFAULT_DINO_SPEC), (
        "models/dino_scene.xml 与当前参数源不一致，请运行：python -m desktop_robot build-dino"
    )


def test_static_verifier_passes():
    report = verify_dino_static(DEFAULT_DINO_SPEC, MODELS_DIR)
    assert report.ok, report.errors

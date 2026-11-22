"""恐龙机型（dino）的参数自洽性、生成物确定性与"长度可调"的传导。"""

from __future__ import annotations

from dataclasses import replace
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from desktop_robot.builder import MODELS_DIR, half, vec
from desktop_robot.config import joint_name
from desktop_robot.dino import (
    _CAPS,
    DEFAULT_DINO_ENV_SPEC,
    DEFAULT_DINO_SPEC,
    DinoEnvSpec,
    head_top_z,
    leg_boxes,
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

#: 期望的轴：俯仰 (0,-1,0)；偏航 (0,0,1)；内外翻 x 镜像（髋 roll 与踝）、脚尖偏航 z 镜像。
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
    "ankle_L": (1.0, 0.0, 0.0),
    "ankle_R": (-1.0, 0.0, 0.0),
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
    assert spec.n_bodies == 20  # 每腿 6 个 body：三段髋块 + 大腿 + 小腿 + 足
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
    assert 0.15 < height < 0.45
    assert height > solve_base_z(spec)  # 头顶一定在基座之上


def test_mass_is_modelled_volume_times_material_density():
    """质量 = 建模体积 × 材料密度（只算塑料件，不摊舵机 / 电池）。"""
    spec = DEFAULT_DINO_SPEC
    volume = sum(size[0] * size[1] * size[2] for _, size, _ in spec.body_geoms)
    assert spec.total_mass() == pytest.approx(volume * spec.density)
    assert 0.5 < spec.total_mass() < 20.0, f"总质量 {spec.total_mass():.3f} kg 不合理"
    assert spec.density == 1240.0  # PLA


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


def test_hip_three_joints_have_their_own_cuboid_links():
    """髋部三个关节各挂一个 body、各一块长方体连杆，按 pitch → roll → yaw 串联。"""
    spec = DEFAULT_DINO_SPEC
    root = ET.fromstring(render_dino_xml(spec))
    for side in ("L", "R"):
        pitch = root.find(f".//body[@name='hip_pitch_{side}']")
        assert pitch is not None
        roll = pitch.find(f"body[@name='hip_roll_{side}']")
        assert roll is not None, "内外翻块不是主髋块的子 body"
        yaw = roll.find(f"body[@name='hip_yaw_{side}']")
        assert yaw is not None, "偏航块不是内外翻块的子 body"
        thigh = yaw.find(f"body[@name='thigh_{side}']")
        assert thigh is not None, "大腿不是偏航块的子 body"

        for body, joint_name_, block_name, size in (
            (pitch, "hip_pitch", "hip_pitch_block", spec.hip_pitch_size),
            (roll, "hip_roll", "hip_roll_block", spec.hip_roll_size),
            (yaw, "hip_yaw", "hip_yaw_block", spec.hip_yaw_size),
        ):
            assert [j.get("name") for j in body.findall("joint")] == [
                f"{joint_name_}_{side}"
            ]
            block = body.find(f"geom[@name='{block_name}_{side}']")
            assert block is not None, f"{block_name}_{side} 缺失"
            assert block.get("type") == "box"
            assert block.get("size") == vec(half(size))


def test_hip_block_sizes_and_offsets_are_adjustable():
    """改髋块尺寸 / 偏距后，geom 尺寸与下游 body 位置都必须自动跟着变。"""
    spec = replace(
        DEFAULT_DINO_SPEC,
        hip_pitch_size=(0.060, 0.020, 0.060),
        hip_roll_size=(0.020, 0.050, 0.050),
        hip_yaw_size=(0.050, 0.046, 0.020),
        hip_roll_offset=(0.0, 0.020, 0.0),
        hip_yaw_offset=(0.020, 0.0, 0.0),
        thigh_offset=(0.0, 0.0, -0.020),
    )
    root = ET.fromstring(render_dino_xml(spec))
    assert root.find(".//geom[@name='hip_pitch_block_L']").get("size") == vec(
        half((0.060, 0.020, 0.060))
    )
    pitch_l = root.find(".//body[@name='hip_pitch_L']")
    roll_l = pitch_l.find("body[@name='hip_roll_L']")
    assert roll_l.get("pos") == "0 0.02 0"
    yaw_l = roll_l.find("body[@name='hip_yaw_L']")
    assert yaw_l.get("pos") == "0.02 0 0"
    assert yaw_l.find("body[@name='thigh_L']").get("pos") == "0 0 -0.02"
    # 右腿 y 镜像、x/z 相同
    roll_r = root.find(".//body[@name='hip_pitch_R']").find("body[@name='hip_roll_R']")
    assert roll_r.get("pos") == "0 -0.02 0"


def test_hip_mount_plates_cover_the_xc330_and_leave_room_for_the_body():
    """安装板端面覆盖 XC330 的 20×34 安装面；电机体 26 mm 缩进上一级连杆。"""
    spec = DEFAULT_DINO_SPEC
    w, h, d = spec.hip_motor_size
    assert (w, h, d) == (0.020, 0.034, 0.026)
    for name, axis, size, offset in spec.hip_blocks():
        axial = size[axis]
        face = sorted(value for index, value in enumerate(size) if index != axis)
        assert face[0] >= w - 1e-9, f"{name} 安装板盖不住电机端面"
        assert face[1] >= h - 1e-9, f"{name} 安装板盖不住电机端面"
        assert 0.005 <= axial <= 0.020, f"{name} 安装板厚度 {axial:.3f} 不合常理"
        # 板贴在关节上、沿偏距伸出一半板厚，所以板厚不能超过偏距
        assert axial <= math.dist(offset, (0.0, 0.0, 0.0)) + 1e-9

    # 电机体往「上一级连杆」里缩 26 mm：pitch 要能进躯干、yaw 要能进大腿
    assert spec.hip_pos[1] - d > -spec.torso_size[1] / 2.0
    assert abs(spec.thigh_offset[2]) + spec.thigh_size[2] >= d


def test_joint_motors_are_all_modelled_and_unique():
    """5 个「不可互相嵌入」的关节都要有电机包络（3 髋 + 2 颈），尺寸 = 电机 + 间隙。"""
    spec = DEFAULT_DINO_SPEC
    motors = spec.joint_motors()
    names = {name for name, _, _, _ in motors}
    assert names == {
        "motor_neck_pitch",
        "motor_neck_yaw",
        "motor_hip_pitch_L",
        "motor_hip_roll_L",
        "motor_hip_yaw_L",
        "motor_hip_pitch_R",
        "motor_hip_roll_R",
        "motor_hip_yaw_R",
    }
    gap = 2 * spec.hip_motor_clearance
    xc330 = sorted(value + gap for value in spec.hip_motor_size)  # (W, H, D) + 间隙
    mg90 = sorted(value + gap for value in spec.head_motor_size)  # (W, T, H) + 间隙
    for name, _, size, _ in motors:
        expected = mg90 if name.startswith("motor_neck_") else xc330
        assert sorted(size) == sorted(expected), name


def test_ankle_is_roll_and_the_foot_is_pitch_mounted():
    """踝是内外翻（轴 ±x），足底水平由 foot_mount_pitch 这个安装角保证。"""
    spec = DEFAULT_DINO_SPEC
    assert spec.ankle.axis_for("L") == (1.0, 0.0, 0.0)
    assert spec.ankle.axis_for("R") == (-1.0, 0.0, 0.0)
    assert spec.stand_ankle == 0.0
    assert spec.foot_mount_pitch == pytest.approx(
        -(spec.base_pitch + spec.stand_hip + spec.stand_knee)
    )

    # 站姿下足 box 的世界旋转应当是单位阵（足底与地面平行）
    base_pitch = spec.base_pitch
    c, s = math.cos(base_pitch), math.sin(base_pitch)
    rot_base = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    for side in ("L", "R"):
        _, _, foot_rot = leg_boxes(spec, side, spec.stand_hip, spec.stand_knee)[0]
        assert np.allclose(rot_base @ foot_rot, np.eye(3), atol=1e-12)

    # XML 里踝轴是 ±x，range 不再被站姿角平移
    root = ET.fromstring(render_dino_xml(spec))
    for side, axis in (("L", "1 0 0"), ("R", "-1 0 0")):
        joint = root.find(f".//joint[@name='ankle_{side}']")
        assert joint.get("axis") == axis
        assert joint.get("range") == "-0.35 0.35"


def test_every_joint_has_a_bearing_cap():
    root = ET.fromstring(render_dino_xml(DEFAULT_DINO_SPEC))
    cap_names = {
        g.get("name")
        for g in root.findall(".//geom")
        if (g.get("name") or "").startswith("cap_")
    }
    # 髋部三个关节用长方体连杆代替灰圆柱，不再有 cap_*
    expected = {
        f"cap_{joint_name(joint, side)}"
        for joint, side in DEFAULT_DINO_SPEC.actuated()
        if joint.name in _CAPS
    }
    assert cap_names == expected


def test_scene_has_include_ground_and_cameras():
    root = ET.fromstring(render_dino_scene_xml(DEFAULT_DINO_SPEC))
    assert root.find("include").get("file") == "dino.xml"
    assert root.find("compiler") is None
    assert root.find(".//geom[@name='floor']") is not None
    wall_names = {geom.get("name") for geom in root.findall(".//geom")}
    assert {"wall_north", "wall_south", "wall_east", "wall_west"} <= wall_names
    camera_names = {c.get("name") for c in root.findall(".//camera")}
    # cam_head 写在本体 dino.xml 的 head body 里（include 之后才会出现在场景中）
    assert {"front", "iso", "ref"} <= camera_names


def test_scene_wall_env_spec_controls_wall_size():
    env_spec = DinoEnvSpec(wall_length=2.0, wall_height=0.6)
    root = ET.fromstring(render_dino_scene_xml(DEFAULT_DINO_SPEC, env_spec))
    north = root.find(".//geom[@name='wall_north']")
    east = root.find(".//geom[@name='wall_east']")
    assert north is not None and east is not None
    assert north.get("pos") == "0 1.02 0.3"
    assert north.get("size") == "1 0.02 0.3"
    assert east.get("pos") == "1.02 0 0.3"
    assert east.get("size") == "0.02 1 0.3"
    assert {name for name, _, _ in DEFAULT_DINO_ENV_SPEC.wall_geoms} == {
        "wall_north",
        "wall_south",
        "wall_east",
        "wall_west",
    }


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

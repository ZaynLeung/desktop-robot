"""参数源自洽性测试：这些断言不过，模型一定有问题。"""

from __future__ import annotations

import math

import pytest

from desktop_robot.config import DEFAULT_SPEC, RobotSpec, joint_name


def test_axes_are_unit_vectors():
    spec = DEFAULT_SPEC
    for joint, side in spec.actuated():
        axis = joint.axis_for(side)
        assert math.isclose(math.dist(axis, (0, 0, 0)), 1.0, abs_tol=1e-12), joint.name


def test_ranges_are_valid_and_contain_stand_pose():
    spec = DEFAULT_SPEC
    stand = spec.stand_q()
    for joint, side in spec.actuated():
        low, high = joint.range
        assert low < high, f"{joint.name} 的 range 非法"
        value = stand[joint_name(joint, side)]
        assert low <= value <= high, f"{joint.name} 的标称角度 {value} 超出 range"


def test_dof_counts():
    assert DEFAULT_SPEC.nq == 16  # 自由关节 7 + 9 个驱动关节
    assert DEFAULT_SPEC.nv == 15
    assert DEFAULT_SPEC.nu == 9
    assert DEFAULT_SPEC.n_bodies == 8
    assert len(DEFAULT_SPEC.joint_names) == 9


def test_joint_order_matches_mjcf_traversal():
    """qpos 顺序必须与模型的深度优先遍历一致。"""
    assert DEFAULT_SPEC.joint_names == [
        "waist_yaw",
        "hip_roll_L",
        "hip_pitch_L",
        "knee_L",
        "ankle_L",
        "hip_roll_R",
        "hip_pitch_R",
        "knee_R",
        "ankle_R",
    ]


def test_mass_is_plausible_for_a_desktop_robot():
    mass = DEFAULT_SPEC.total_mass()
    assert 0.5 < mass < 3.0, f"总质量 {mass:.3f} kg 不像桌面机器人"


def test_hip_roll_is_mirrored_between_legs():
    left = DEFAULT_SPEC.hip_roll.axis_for("L")
    right = DEFAULT_SPEC.hip_roll.axis_for("R")
    assert left != right
    assert left[0] == -right[0]


def test_pitch_joints_share_the_same_axis():
    """髋/膝/踝俯仰轴一致，才能让"角度和为零 = 足底水平"这条判据成立。"""
    spec = DEFAULT_SPEC
    assert spec.hip_pitch.axis == spec.knee.axis == spec.ankle.axis


def test_spec_is_frozen():
    with pytest.raises(Exception):
        DEFAULT_SPEC.hip_y = 0.1  # type: ignore[misc]


def test_custom_spec_can_be_derived():
    """参数化：改一个尺寸，派生量跟着变。"""
    from dataclasses import replace

    bigger = replace(DEFAULT_SPEC, thigh_size=(0.04, 0.04, 0.10))
    assert bigger.thigh_len == pytest.approx(0.10)
    assert bigger.leg_len > DEFAULT_SPEC.leg_len
    assert isinstance(bigger, RobotSpec)

"""运动学测试：不依赖 MuJoCo，纯 numpy。"""

from __future__ import annotations

import numpy as np
import pytest

from desktop_robot.config import DEFAULT_SPEC
from desktop_robot.kinematics import (
    foot_corners,
    foot_sole_z,
    leg_fk,
    leg_q,
    numeric_jacobian,
    rot,
    solve_pelvis_z,
    stand_qpos,
    unit,
)

SPEC = DEFAULT_SPEC


def test_rot_is_orthonormal():
    R = rot((0.0, -1.0, 0.0), 0.37)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_unit_rejects_zero_vector():
    with pytest.raises(ValueError):
        unit((0.0, 0.0, 0.0))


def test_zero_pose_puts_foot_straight_below_hip():
    R, p = leg_fk(SPEC, "L", [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(R, np.eye(3), atol=1e-12)
    assert p[0] == pytest.approx(0.0, abs=1e-12)
    assert p[1] == pytest.approx(SPEC.hip_y)
    assert p[2] == pytest.approx(SPEC.hip_z - SPEC.leg_len)


def test_positive_hip_pitch_swings_forward():
    """约定：俯仰正角度 = 子连杆前摆（+x）。"""
    _, p = leg_fk(SPEC, "L", [0.0, 0.3, 0.0, 0.0])
    assert p[0] > 0.0
    assert p[2] > SPEC.hip_z - SPEC.leg_len  # 腿抬起来了


def test_negative_knee_angle_flexes_backward():
    _, p = leg_fk(SPEC, "L", [0.0, 0.0, -0.3, 0.0])
    assert p[0] < 0.0


def test_level_foot_invariant():
    """髋 + 膝 + 踝 = 0 时足部坐标系保持水平（这是标称站姿的设计依据）。"""
    hip_pitch, knee, ankle = SPEC.stand
    assert hip_pitch + knee + ankle == pytest.approx(0.0)
    R, _ = leg_fk(SPEC, "L", [0.0, hip_pitch, knee, ankle])
    assert np.allclose(R, np.eye(3), atol=1e-12)


def test_hip_roll_abducts_outward_on_both_sides():
    _, left = leg_fk(SPEC, "L", [0.15, 0.0, 0.0, 0.0])
    _, right = leg_fk(SPEC, "R", [0.15, 0.0, 0.0, 0.0])
    assert left[1] > SPEC.hip_y  # 左腿向 +y
    assert right[1] < -SPEC.hip_y  # 右腿向 -y
    assert left[1] == pytest.approx(-right[1])


def test_left_right_symmetry():
    q = [0.05, 0.2, -0.4, 0.2]
    _, left = leg_fk(SPEC, "L", q)
    _, right = leg_fk(SPEC, "R", q)
    assert left[0] == pytest.approx(right[0])
    assert left[1] == pytest.approx(-right[1])
    assert left[2] == pytest.approx(right[2])


def test_straight_legs_need_pelvis_at_foot_height_plus_leg_length():
    z = solve_pelvis_z(SPEC, (0.0, 0.0, 0.0))
    expected = SPEC.foot_size[2] / 2.0 - SPEC.foot_center[2] - SPEC.hip_z + SPEC.leg_len
    assert z == pytest.approx(expected, abs=1e-9)


def test_stand_pose_has_both_soles_touching_the_ground():
    q = [0.0, *SPEC.stand]
    z = solve_pelvis_z(SPEC)
    for side in ("L", "R"):
        assert foot_sole_z(SPEC, side, q) + z == pytest.approx(0.0, abs=1e-9)


def test_foot_corners_are_eight_points_in_a_box():
    corners = foot_corners(SPEC, "L", [0.0, 0.0, 0.0, 0.0])
    assert corners.shape == (8, 3)
    spans = corners.max(axis=0) - corners.min(axis=0)
    assert np.allclose(spans, SPEC.foot_size, atol=1e-12)


def test_stand_qpos_shape_and_order():
    qpos = stand_qpos(SPEC)
    assert len(qpos) == SPEC.nq
    assert qpos[3:7] == [1.0, 0.0, 0.0, 0.0]  # 单位四元数
    assert qpos[2] == pytest.approx(solve_pelvis_z(SPEC))


def test_jacobian_matches_finite_difference_of_position():
    q = [0.05, 0.25, -0.5, 0.2]
    J = numeric_jacobian(SPEC, "L", q)
    assert J.shape == (3, 4)
    eps = 1e-6
    for i in range(4):
        dq = list(q)
        dq[i] += eps
        p_plus = leg_fk(SPEC, "L", dq)[1]
        dq[i] -= 2 * eps
        p_minus = leg_fk(SPEC, "L", dq)[1]
        assert np.allclose(J[:, i], (p_plus - p_minus) / (2 * eps), atol=1e-5)


def test_leg_q_reads_from_joint_dictionary():
    q = SPEC.stand_q()
    assert leg_q(SPEC, q, "L") == [0.0, *SPEC.stand]

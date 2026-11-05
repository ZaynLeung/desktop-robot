"""渲染节拍的回归测试：复位把仿真时间归零后，节拍必须最多只领先一帧。"""

from __future__ import annotations

import pytest

from dino_wbc.render import advance_render_clock, clamp_render_clock

DT = 1.0 / 30.0


def test_clamp_pulls_clock_back_after_reset():
    # 复位前已经跑到 4.95 s，data.time 归零到 0.02 s
    assert clamp_render_clock(4.95, 0.02, DT) == pytest.approx(0.02 + DT)


def test_clamp_keeps_normal_clock():
    # 正常运行时 next_render 本来就落后于 sim_time + dt，不应被改动
    assert clamp_render_clock(0.05, 0.04, DT) == pytest.approx(0.05)


def test_advance_follows_normal_cadence():
    assert advance_render_clock(0.0, 0.02, DT) == pytest.approx(DT)


def test_advance_does_not_burst_after_slow_frame():
    # 渲染落后于仿真时间时，只对齐到当前时刻，不补一串帧
    assert advance_render_clock(0.10, 0.40, DT) == pytest.approx(0.40 + DT)


def test_reset_scenario_never_blocks_more_than_one_frame():
    """模拟一次复位：旧节拍 5 s、时间归零，下一次渲染间隔必须 ≤ 一帧。"""
    next_render = 1.0  # 复位后还停在旧节拍
    sim_time = 0.02
    next_render = clamp_render_clock(next_render, sim_time, DT)
    assert next_render == pytest.approx(sim_time + DT)
    assert next_render - sim_time <= DT + 1e-12

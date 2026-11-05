"""课程推进：抬脚高度/速度爬坡、standup 晋级与退档。"""

from __future__ import annotations

import numpy as np

from dino_wbc import config
from dino_wbc.curriculum import Curriculum


def _spam(curriculum: Curriculum, success: bool, rounds: int) -> None:
    for _ in range(rounds):
        curriculum.on_episode(success)


def test_lift_curriculum_grows_with_success():
    curriculum = Curriculum("lift", rng=np.random.default_rng(0))
    start = curriculum.lift_h_max
    _spam(curriculum, True, config.CURRICULUM_WINDOW)
    assert curriculum.lift_h_max > start
    assert curriculum.lift_h_max <= config.LIFT_H_RANGE[1] + 1e-9


def test_walk_curriculum_caps_at_range():
    curriculum = Curriculum("walk", rng=np.random.default_rng(0))
    _spam(curriculum, True, 200)
    assert curriculum.vx_max <= max(abs(v) for v in config.WALK_VX_RANGE) + 1e-9
    assert curriculum.vy_max <= max(abs(v) for v in config.WALK_VY_RANGE) + 1e-9


def test_standup_curriculum_promotes_in_order():
    curriculum = Curriculum("standup", rng=np.random.default_rng(0))
    assert curriculum.sample_standup_pose().name == "crouch"
    _spam(curriculum, True, config.CURRICULUM_WINDOW)
    assert curriculum.sample_standup_pose().name == "side"
    _spam(curriculum, True, config.CURRICULUM_WINDOW * 2)
    assert curriculum.sample_standup_pose().name == "supine"


def test_standup_curriculum_demotes_when_failing():
    curriculum = Curriculum("standup", rng=np.random.default_rng(0))
    curriculum.pose_index = 1
    _spam(curriculum, False, config.CURRICULUM_WINDOW)
    assert curriculum.sample_standup_pose().name == "crouch"

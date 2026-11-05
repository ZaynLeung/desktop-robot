"""桌面双足机器人的参数化 MuJoCo 模型。

最小用法::

    python -m desktop_robot build     # 生成 models/biped.xml 与 models/scene.xml
    python -m desktop_robot verify    # 静态校验（装了 MuJoCo 可加 --dynamic）
    python -m desktop_robot view      # 打开 MuJoCo 查看器
"""

from __future__ import annotations

from .builder import (
    MODELS_DIR,
    build_robot,
    build_scene,
    render_robot_xml,
    render_scene_xml,
    write_models,
)
from .config import DEFAULT_SPEC, JointSpec, RobotSpec
from .kinematics import foot_corners, foot_sole_z, leg_fk, numeric_jacobian, solve_pelvis_z

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SPEC",
    "JointSpec",
    "RobotSpec",
    "MODELS_DIR",
    "build_robot",
    "build_scene",
    "render_robot_xml",
    "render_scene_xml",
    "write_models",
    "foot_corners",
    "foot_sole_z",
    "leg_fk",
    "numeric_jacobian",
    "solve_pelvis_z",
]

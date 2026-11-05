"""提交进仓库的 ``models/*.xml`` 必须和生成器一致。

忘了重新 build、或者手改了 XML 却改错地方，这个测试会立刻失败：
生成物是"编译产物"，但因为它要能被没有 Python 的人直接打开，
所以选择提交入库，并用这个测试保证它不会悄悄过期。
"""

from __future__ import annotations

from desktop_robot.builder import MODELS_DIR, render_robot_xml, render_scene_xml
from desktop_robot.config import DEFAULT_SPEC


def test_committed_robot_model_is_up_to_date():
    path = MODELS_DIR / "biped.xml"
    assert path.exists(), "缺少 models/biped.xml，请运行 python -m desktop_robot build"
    assert path.read_text(encoding="utf-8") == render_robot_xml(DEFAULT_SPEC), (
        "models/biped.xml 与当前参数源不一致，请运行：python -m desktop_robot build"
    )


def test_committed_scene_is_up_to_date():
    path = MODELS_DIR / "scene.xml"
    assert path.exists(), "缺少 models/scene.xml，请运行 python -m desktop_robot build"
    assert path.read_text(encoding="utf-8") == render_scene_xml(DEFAULT_SPEC), (
        "models/scene.xml 与当前参数源不一致，请运行：python -m desktop_robot build"
    )

"""相机与画面合成测试。

需要 mujoco；缺 cv2 时只有画面合成/录制用例 skip。
``cam_head`` 由模型层并行加入：当前 models 还没有该相机时，头部画面用例
明确 ``pytest.skip("models 尚未加入 cam_head")``。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")


def _make_frame(
    height: int | None = None, width: int | None = None, seed: int = 0
) -> np.ndarray:
    from dino_wbc import config

    height = height or config.RENDER_HEIGHT
    width = width or config.RENDER_WIDTH
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


# ----------------------------------------------------------------------
# 跟随相机
# ----------------------------------------------------------------------
def test_follow_camera_tracks_base_x(model_layout):
    """lookat 随基座 x 指数平滑移动（不是瞬移，也不是不动）。"""
    import mujoco

    from dino_wbc import config
    from dino_wbc.cameras import FollowCamera
    from dino_wbc.state import reset_to_stand

    model, layout = model_layout
    data = mujoco.MjData(model)
    reset_to_stand(model, data, layout)
    mujoco.mj_forward(model, data)

    camera = FollowCamera()
    for _ in range(10):
        camera.update(model, data, layout)
    before = np.array(camera.camera.lookat, dtype=np.float64)
    assert abs(before[0]) < 1e-6

    # 把基座往前挪 0.5 m：单帧只走一小步（指数平滑），多帧后基本追上。
    data.qpos[0] = 0.5
    mujoco.mj_forward(model, data)
    camera.update(model, data, layout)
    single = np.array(camera.camera.lookat, dtype=np.float64)
    assert before[0] < single[0] < 0.45

    for _ in range(80):
        camera.update(model, data, layout)
    after = np.array(camera.camera.lookat, dtype=np.float64)
    assert after[0] > 0.45
    assert abs(after[1]) < 1e-6

    # 相机参数来自 config。
    assert camera.camera.distance == pytest.approx(config.FOLLOW_DISTANCE)
    assert camera.camera.azimuth == pytest.approx(config.FOLLOW_AZIMUTH)
    assert camera.camera.elevation == pytest.approx(config.FOLLOW_ELEVATION)


def test_follow_camera_free_mode_does_not_track(model_layout):
    import mujoco

    from dino_wbc.cameras import FollowCamera
    from dino_wbc.state import reset_to_stand

    model, layout = model_layout
    data = mujoco.MjData(model)
    reset_to_stand(model, data, layout)
    mujoco.mj_forward(model, data)

    camera = FollowCamera()
    camera.update(model, data, layout)
    assert camera.set_tracking(False) is None  # 返回 None，仅设置
    camera.update(model, data, layout)
    frozen = np.array(camera.camera.lookat, dtype=np.float64)

    data.qpos[0] = 1.0
    mujoco.mj_forward(model, data)
    for _ in range(20):
        camera.update(model, data, layout)
    np.testing.assert_allclose(camera.camera.lookat, frozen, atol=1e-9)


# ----------------------------------------------------------------------
# 头部相机 + 双画面
# ----------------------------------------------------------------------
def test_head_camera_available_flag(model_layout):
    from dino_wbc.cameras import HeadCamera

    model, _layout = model_layout
    available = HeadCamera.available(model)
    if not available:
        pytest.skip("models 尚未加入 cam_head")
    camera = HeadCamera(model)
    assert camera.name == "cam_head"
    assert camera.camera_id >= 0


def test_dual_renderer_shapes(model_layout):
    """跟随 + 头部各出一张 (360, 480, 3) uint8。"""
    import mujoco

    from dino_wbc import config
    from dino_wbc.cameras import DualRenderer, FollowCamera, HeadCamera
    from dino_wbc.state import reset_to_stand

    model, layout = model_layout
    if not HeadCamera.available(model):
        pytest.skip("models 尚未加入 cam_head")

    data = mujoco.MjData(model)
    reset_to_stand(model, data, layout)
    mujoco.mj_forward(model, data)

    try:
        renderer = DualRenderer(model)
    except RuntimeError as exc:  # pragma: no cover - 无 OpenGL 的机器
        pytest.skip(f"当前环境无法离屏渲染：{exc}")

    try:
        follow_rgb, head_rgb = renderer.render(model, data, FollowCamera())
        assert renderer.head_enabled
        for image in (follow_rgb, head_rgb):
            assert image.shape == (config.RENDER_HEIGHT, config.RENDER_WIDTH, 3)
            assert image.dtype == np.uint8
        # 画面不是全黑（说明 GL 真的渲染出了场景）。
        assert follow_rgb.any()
    finally:
        renderer.close()


def test_dual_renderer_without_head(model_layout):
    """``--no-head`` 路径：不依赖 cam_head，返回全黑占位图。"""
    import mujoco

    from dino_wbc import config
    from dino_wbc.cameras import DualRenderer, FollowCamera
    from dino_wbc.state import reset_to_stand

    model, layout = model_layout
    data = mujoco.MjData(model)
    reset_to_stand(model, data, layout)
    mujoco.mj_forward(model, data)

    try:
        renderer = DualRenderer(model, head=False)
    except RuntimeError as exc:  # pragma: no cover - 无 OpenGL 的机器
        pytest.skip(f"当前环境无法离屏渲染：{exc}")
    try:
        follow_rgb, head_rgb = renderer.render(model, data, FollowCamera())
        assert not renderer.head_enabled
        assert follow_rgb.shape == (config.RENDER_HEIGHT, config.RENDER_WIDTH, 3)
        assert head_rgb.shape == (config.RENDER_HEIGHT, config.RENDER_WIDTH, 3)
        assert head_rgb.dtype == np.uint8
        assert not head_rgb.any()
    finally:
        renderer.close()


# ----------------------------------------------------------------------
# HUD 合成 / 录制（需要 cv2）
# ----------------------------------------------------------------------
def test_compose_frame_hud_and_head_placeholder():
    pytest.importorskip("cv2")
    from dino_wbc import config
    from dino_wbc.render import compose_frame

    follow = _make_frame(seed=1)
    head = _make_frame(seed=2)
    frame = compose_frame(follow, head, ["line one", "line two", "line three"])
    assert frame.dtype == np.uint8
    assert frame.shape == (config.RENDER_HEIGHT, config.RENDER_WIDTH * 2, 3)
    # 左右两个面板内容不同（确实并排了）。
    assert not np.array_equal(
        frame[:, : config.RENDER_WIDTH], frame[:, config.RENDER_WIDTH :]
    )

    # 头部画面关闭时右侧画占位提示。
    off = compose_frame(follow, None, ["only"], head_enabled=False)
    assert off.shape == frame.shape
    assert not np.array_equal(off, frame)


def test_video_recorder_writes_mp4(tmp_path):
    pytest.importorskip("cv2")
    from dino_wbc.render import VideoRecorder

    path = tmp_path / "clip.mp4"
    with VideoRecorder(path, fps=10) as recorder:
        for index in range(5):
            recorder.write(_make_frame(seed=index))
    assert recorder.frames == 5
    assert path.exists()
    assert path.stat().st_size > 0

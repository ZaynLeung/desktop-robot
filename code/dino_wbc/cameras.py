"""相机与双画面离屏渲染。

* :class:`FollowCamera`：第三人称自由相机，``lookat`` 指数平滑地跟到躯干上方；
* :class:`HeadCamera`：头部第一视角相机（只用名字 ``config.HEAD_CAMERA`` 查，
  模型里没有就抛 :class:`CameraNotFoundError`，运行端可据此退回 ``--no-head``）；
* :class:`DualRenderer`：两个 ``mujoco.Renderer``（跟随 + 头部），一次出两张 RGB。

所有渲染都是离屏（``mujoco.Renderer``），不依赖 MuJoCo viewer 窗口。
"""

from __future__ import annotations

import numpy as np

from . import config

try:  # pragma: no cover - 缺 mujoco 时给出可读错误
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dino_wbc.cameras 需要 mujoco，请先执行：\n"
        r"  E:\conda_envs\desktop-robot\python.exe -m pip install -r code/requirements-wbc.txt"
    ) from exc

#: lookat 目标比躯干 body 原点高出的距离（m），让画面中心落在身体中上部。
LOOKAT_ABOVE_TORSO = 0.05


class CameraNotFoundError(RuntimeError):
    """模型里缺少预期相机（多半是 models/*.xml 尚未重新生成）。"""


class FollowCamera:
    """第三人称跟随相机（``mjCAMERA_FREE``）。

    ``lookat`` 用指数平滑跟到「躯干位置 + z 偏移」，避免抬脚/落地时画面抖动。
    关掉 ``tracking`` 后 lookat 停在原地，相当于固定机位（运行端 ``F`` 键切换）。
    """

    def __init__(
        self,
        *,
        distance: float = config.FOLLOW_DISTANCE,
        azimuth: float = config.FOLLOW_AZIMUTH,
        elevation: float = config.FOLLOW_ELEVATION,
        alpha: float = config.FOLLOW_LOOKAT_ALPHA,
    ) -> None:
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = float(distance)
        self.camera.azimuth = float(azimuth)
        self.camera.elevation = float(elevation)
        self.camera.lookat[:] = (0.0, 0.0, 0.3)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.tracking = True
        self.initialized = False

    # ------------------------------------------------------------------
    def toggle(self) -> bool:
        """切换 tracking / free，返回切换后的 tracking 状态。"""
        self.tracking = not self.tracking
        return self.tracking

    def set_tracking(self, tracking: bool) -> None:
        self.tracking = bool(tracking)

    def update(self, model, data, layout) -> None:
        """把 lookat 平滑地移到躯干上方（首次直接对齐，避免从原点飞过去）。"""
        torso = np.asarray(data.xpos[layout.torso_body_id], dtype=np.float64)
        target = torso + np.array([0.0, 0.0, LOOKAT_ABOVE_TORSO], dtype=np.float64)
        if not self.initialized:
            self.camera.lookat[:] = target
            self.initialized = True
            return
        if not self.tracking:
            return
        current = np.asarray(self.camera.lookat, dtype=np.float64)
        self.camera.lookat[:] = (1.0 - self.alpha) * current + self.alpha * target


class HeadCamera:
    """头部第一视角相机（按名字 ``config.HEAD_CAMERA`` 查找）。"""

    def __init__(self, model, name: str = config.HEAD_CAMERA) -> None:
        self.name = str(name)
        self.camera_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.name)
        )
        if self.camera_id < 0:
            raise CameraNotFoundError(
                f"模型里没有相机 {self.name!r}：models/*.xml 还是旧版本。\n"
                "重新生成模型（python -m desktop_robot build-dino），"
                "或运行端加 --no-head。"
            )

    @staticmethod
    def available(model, name: str = config.HEAD_CAMERA) -> bool:
        """模型里是否有该相机（运行端用来决定是否自动关掉头部画面）。"""
        return (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, str(name)) >= 0
        )

    def update(self, model, data) -> int:
        """头部相机跟着 body 走，不需要每帧计算；返回 camera id。"""
        return int(self.camera_id)


class DualRenderer:
    """跟随 + 头部双画面离屏渲染器。

    ``head=False`` 时不创建头部 renderer（模型还没有 ``cam_head`` 或运行端传了
    ``--no-head``），此时 :meth:`render` 返回全黑占位图，保证呼叫方不用分支。
    """

    def __init__(
        self,
        model,
        width: int = config.RENDER_WIDTH,
        height: int = config.RENDER_HEIGHT,
        *,
        head: bool = True,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.follow_renderer = mujoco.Renderer(model, self.height, self.width)
        self.head_camera: HeadCamera | None = None
        self.head_renderer = None
        if head:
            try:
                self.head_camera = HeadCamera(model)
            except CameraNotFoundError:
                self.follow_renderer.close()
                raise
            self.head_renderer = mujoco.Renderer(model, self.height, self.width)

    # ------------------------------------------------------------------
    @property
    def head_enabled(self) -> bool:
        return self.head_renderer is not None

    def render(self, model, data, follow_cam: FollowCamera):
        """渲染一帧 → ``(follow_rgb, head_rgb)``，都是 ``(H, W, 3)`` uint8 RGB。"""
        self.follow_renderer.update_scene(data, camera=follow_cam.camera)
        follow_rgb = np.array(self.follow_renderer.render(), dtype=np.uint8, copy=True)
        if self.head_renderer is not None and self.head_camera is not None:
            self.head_renderer.update_scene(data, camera=self.head_camera.camera_id)
            head_rgb = np.array(
                self.head_renderer.render(), dtype=np.uint8, copy=True
            )
        else:
            head_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return follow_rgb, head_rgb

    def close(self) -> None:
        if self.head_renderer is not None:
            self.head_renderer.close()
            self.head_renderer = None
        if self.follow_renderer is not None:
            self.follow_renderer.close()

    def __enter__(self) -> "DualRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

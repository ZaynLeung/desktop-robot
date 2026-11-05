"""画面合成（左右双画面 + ASCII HUD）与 mp4 录制。

只依赖 OpenCV；``cv2`` 采用延迟导入，未安装时模块仍可 import，调用时才抛
带安装命令的 :class:`ImportError`（运行端 ``--headless`` 不录制时不需要 cv2）。

HUD 全部用 ASCII（``cv2.putText`` 默认字体不支持中文），中文提示走控制台。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_CV2_INSTALL_HINT = (
    "画面合成 / mp4 录制需要 OpenCV：\n"
    r"  E:\conda_envs\desktop-robot\python.exe -m pip install opencv-python"
)

#: HUD 字体与字号。
_HUD_FONT = "FONT_HERSHEY_SIMPLEX"
_HUD_SCALE = 0.45
_HUD_THICKNESS = 1
_HUD_LINE_H = 16
_HUD_PAD = 6
_HUD_COLOR = (235, 235, 235)  # BGR
_HUD_BAR_ALPHA = 0.55


def require_cv2():
    """延迟取 cv2；未安装时给可执行的安装提示。"""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - 只在未装环境触发
        raise ImportError(_CV2_INSTALL_HINT) from exc
    return cv2


def _to_bgr(image) -> np.ndarray:
    """RGB / 灰度 / None → BGR uint8 三通道。"""
    cv2 = require_cv2()
    if image is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    data = np.asarray(image)
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    if data.ndim == 2:
        return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    if data.ndim != 3 or data.shape[2] not in (3, 4):
        raise ValueError(f"不支持的图像形状：{data.shape}")
    if data.shape[2] == 4:
        data = data[:, :, :3]
    return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)


def _draw_hud(frame: np.ndarray, lines: list[str], *, top: bool) -> None:
    """在半透明黑条上画多行 ASCII 文本（就地修改 frame）。"""
    if not lines:
        return
    cv2 = require_cv2()
    bar_height = _HUD_PAD * 2 + _HUD_LINE_H * len(lines)
    y0 = 0 if top else frame.shape[0] - bar_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (frame.shape[1], y0 + bar_height), (0, 0, 0), -1)
    cv2.addWeighted(
        overlay, _HUD_BAR_ALPHA, frame, 1.0 - _HUD_BAR_ALPHA, 0.0, dst=frame
    )
    font = getattr(cv2, _HUD_FONT)
    for index, text in enumerate(lines):
        baseline = y0 + _HUD_PAD + _HUD_LINE_H * (index + 1) - 4
        cv2.putText(
            frame,
            str(text),
            (_HUD_PAD, baseline),
            font,
            _HUD_SCALE,
            _HUD_COLOR,
            _HUD_THICKNESS,
            cv2.LINE_AA,
        )


def _draw_placeholder(frame: np.ndarray, text: str, x0: int, width: int) -> None:
    """在指定面板中央画提示文字（例如头部画面关闭时）。"""
    cv2 = require_cv2()
    font = getattr(cv2, _HUD_FONT)
    scale = 0.6
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, 1)
    x = x0 + max((width - text_w) // 2, 4)
    y = frame.shape[0] // 2 + text_h // 2
    cv2.putText(frame, text, (x, y), font, scale, _HUD_COLOR, 1, cv2.LINE_AA)


def compose_frame(
    follow_rgb,
    head_rgb,
    hud_lines: list[str],
    *,
    head_enabled: bool = True,
    head_label: str = "HEAD OFF (--no-head)",
) -> np.ndarray:
    """左右并排合成双画面并在顶部/底部画 ASCII HUD，返回 **BGR**。

    :param follow_rgb: 跟随相机 RGB ``(H, W, 3)`` uint8；
    :param head_rgb: 头部相机 RGB，``head_enabled=False`` 时可传 ``None``；
    :param hud_lines: HUD 文本行；前半放顶部、后半放底部；
    :param head_enabled: 头部画面是否有效（False 时右侧画占位提示）；
    """
    cv2 = require_cv2()
    left = _to_bgr(follow_rgb)
    right = _to_bgr(head_rgb) if head_rgb is not None else np.zeros_like(left)
    if right.shape[:2] != left.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))

    left_width = left.shape[1]
    frame = np.hstack([left, right])

    if not head_enabled:
        _draw_placeholder(frame, head_label, left_width, right.shape[1])

    lines = [str(line) for line in hud_lines]
    split = (len(lines) + 1) // 2  # 顶部多放一行
    _draw_hud(frame, lines[:split], top=True)
    _draw_hud(frame, lines[split:], top=False)
    return frame


class VideoRecorder:
    """``cv2.VideoWriter`` 的薄封装：懒打开、自动建目录、大小自适应。"""

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float = 30.0,
        fourcc: str = "mp4v",
        size: tuple[int, int] | None = None,
    ) -> None:
        require_cv2()
        self._path = Path(path)
        self.fps = float(fps)
        self.fourcc = str(fourcc)
        self.size = size  # (width, height)
        self.writer = None
        self.frames = 0

    # ------------------------------------------------------------------
    def _open(self, size: tuple[int, int]) -> None:
        cv2 = require_cv2()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, size)
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(
                f"无法创建视频文件：{self.path}（fourcc={self.fourcc!r}）"
            )
        self.size = size

    def write(self, frame) -> None:
        """写一帧（BGR；也接受灰度图）。第一帧决定视频尺寸。"""
        cv2 = require_cv2()
        data = np.asarray(frame)
        if data.dtype != np.uint8:
            data = np.clip(data, 0, 255).astype(np.uint8)
        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        if data.ndim != 3 or data.shape[2] != 3:
            raise ValueError(f"不支持的帧形状：{data.shape}")
        if self.size is None:
            self._open((int(data.shape[1]), int(data.shape[0])))
        if self.size is not None and (
            int(data.shape[1]),
            int(data.shape[0]),
        ) != self.size:
            data = cv2.resize(data, self.size)
        assert self.writer is not None
        self.writer.write(data)
        self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    @property
    def path(self) -> Path:
        """目标 mp4 路径。"""
        return self._path

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

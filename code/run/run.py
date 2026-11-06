"""小恐龙 WBC 运行端：控制器 → MuJoCo 仿真 → 双画面渲染 / HUD / mp4 录制。

从仓库任意目录都能直接执行（脚本自己把 ``code/`` 插进 ``sys.path``）::

    E:\\conda_envs\\desktop-robot\\python.exe code\\run\\run.py
    E:\\conda_envs\\desktop-robot\\python.exe code\\run\\run.py --controller scripted_lift
    E:\\conda_envs\\desktop-robot\\python.exe code\\run\\run.py --controller lift --record out\\lift.mp4
    E:\\conda_envs\\desktop-robot\\python.exe code\\run\\run.py --controller scripted_stand --headless --seconds 1

不 import stable-baselines3；只依赖 numpy / mujoco / torch（仅策略控制器）/ opencv
（仅窗口或录制）。``--headless`` 且不 ``--record`` 时连 GL 渲染器和 cv2 都不需要。

按键（需要窗口）::

    1-5        切换阶段 stand / lift / standup / walk / turn（重新加载控制器并复位）
    W / S      vx ±0.02（行走）
    A / D      vy ±0.02（行走）
    L / R      抬脚目标脚：左脚 / 右脚
    U          命令切成 standup
    Z / C      turn 阶段的绝对朝向指令 ∓5°
    P          手动推一下（5 N × 0.1 s）
    Space      复位到当前阶段起始姿态并重置控制器
    V          standup 阶段切换起始姿态：crouch / side / supine 循环
    F          跟随相机 tracking / free 切换
    Esc / Q    退出
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 允许从仓库任意目录直接执行本脚本（必须放在 import dino_wbc 之前）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dino_wbc import config, model_io
from dino_wbc.action import action_to_ctrl
from dino_wbc.cameras import CameraNotFoundError, DualRenderer, FollowCamera, HeadCamera
from dino_wbc.controllers.registry import load_controller
from dino_wbc.observation import ObsBuilder, make_command
from dino_wbc.render import (
    VideoRecorder,
    advance_render_clock,
    clamp_render_clock,
    compose_frame,
    require_cv2,
)
from dino_wbc.state import apply_standup_pose, read_state, reset_to_stand

try:  # pragma: no cover - 缺包时给出可读错误
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "运行端需要 mujoco，请先执行：\n"
        r"  E:\conda_envs\desktop-robot\python.exe -m pip install -r code/requirements-wbc.txt"
    ) from exc

WINDOW_NAME = "dino_wbc"
ESC = 27
#: 数字键热切换 → registry 名字。
TASK_KEYS = {
    ord("1"): "stand",
    ord("2"): "lift",
    ord("3"): "standup",
    ord("4"): "walk",
    ord("5"): "turn",
}
#: 还没有 checkpoint 的阶段退回到脚本控制器（保证演示可用）。
SCRIPTED_FALLBACK = {"stand": "scripted_stand", "lift": "scripted_lift"}

#: 控制周期（秒）。
CONTROL_DT = 1.0 / config.CONTROL_HZ
#: 渲染周期（秒）。
RENDER_DT = 1.0 / config.RENDER_FPS


def _configure_stdio() -> None:
    """让 Windows 控制台能正常输出中文（复制自 ``desktop_robot/__main__.py``）。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):  # pragma: no cover - 重定向到管道时可能失败
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="小恐龙 WBC 运行端：加载控制器并实时仿真",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--controller",
        default="scripted_stand",
        help="registry 名字 / scripted_stand / scripted_lift / 直接给 .pt 路径",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="覆盖 registry 里记录的 checkpoint 路径",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="模型 xml 路径（默认 models/dino_scene.xml）",
    )
    parser.add_argument(
        "--record",
        default=None,
        metavar="PATH",
        help="把合成后的双画面写成 mp4",
    )
    parser.add_argument("--headless", action="store_true", help="不开窗口（仍可 --record）")
    parser.add_argument("--fast", action="store_true", help="不按实时限速，尽快跑完")
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="仿真到这个秒数自动退出（冒烟测试用）",
    )
    parser.add_argument(
        "--no-head",
        action="store_true",
        help="不渲染头部相机画面（模型还没有 cam_head 时使用）",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="跟随相机不跟踪基座（固定机位）",
    )
    parser.add_argument(
        "--lift-h",
        type=float,
        default=0.025,
        help="lift 阶段的抬脚高度指令（m），会夹在 LIFT_H_RANGE 内",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=0.0,
        help="turn 阶段的绝对朝向指令（度）；stand 固定为 0",
    )
    return parser


class Runner:
    """一次运行的全部状态：模型、控制器、命令、渲染器。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model, self.layout = model_io.load(args.model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.contract = model_io.contract_hash(self.model, self.layout)
        self.controller = load_controller(
            args.controller,
            expected_contract=self.contract,
            checkpoint=args.checkpoint,
        )
        self.obs_builder = ObsBuilder(self.layout)
        self.command = self._new_command(self.controller.stage)
        self.follow_cam = FollowCamera()
        self.follow_cam.set_tracking(not args.no_follow)

        # standup 阶段默认用「仰卧」起始姿态（课程里最难的一档）；其余阶段站姿。
        self._standup_pose_index = 2
        self._reset_episode()

        self._fps = 0.0
        self._fps_frames = 0
        self._fps_t0 = time.perf_counter()
        self._running = True
        self._sim_steps = 0
        # 手动轻推（P 键）：持续 5 个控制步（0.1 s）后清除
        self._manual_push_steps = 0
        self._manual_push_vec = np.zeros(3, dtype=np.float64)

        self.renderer: DualRenderer | None = None
        self.recorder: VideoRecorder | None = None
        self._head_enabled = False
        self._need_render = bool(args.record) or not args.headless

        model_path = model_io.resolve_model_path(args.model)
        print(f"模型：{model_path}")
        print(f"契约：{self.contract}")
        print(
            f"控制器：{self.controller.name}"
            f"（stage={self.controller.stage}，act_dim={self.controller.act_dim}）"
        )

        if self._need_render:
            self._setup_render()
        else:
            # 纯 headless 冒烟：不建 GL 上下文、不 import cv2，速度最快。
            print("headless 模式：不渲染画面（只推进物理）")

    # ------------------------------------------------------------------
    # 初始化辅助
    # ------------------------------------------------------------------
    def _new_command(self, stage: str) -> np.ndarray:
        """按阶段构造命令：lift 给抬脚高度，turn 给绝对朝向，stand 朝向固定 0。"""
        if stage == "turn":
            return make_command("turn", yaw=float(np.radians(self.args.yaw)))
        command = make_command(stage)
        if stage == "lift":
            low, high = config.LIFT_H_RANGE
            command[config.CMD_LIFT_H] = float(
                np.clip(self.args.lift_h, low, high)
            )
            command[config.CMD_TARGET_FOOT] = -1.0  # 默认抬左脚
        return command

    def _sync_target_foot(self) -> None:
        """让脚本控制器和命令里的目标脚保持一致（L/R 键）。"""
        setter = getattr(self.controller, "set_target_foot", None)
        if callable(setter):
            setter(float(self.command[config.CMD_TARGET_FOOT]))

    def _current_standup_pose(self):
        """当前 standup 起始姿态（非 standup 阶段返回 None）。"""
        if self.controller.stage != "standup":
            return None
        poses = config.STANDUP_POSES
        return poses[self._standup_pose_index % len(poses)]

    def _setup_render(self) -> None:
        head_ok = not self.args.no_head and HeadCamera.available(self.model)
        if not self.args.no_head and not head_ok:
            print(
                "[提示] models 里还没有 cam_head 相机，自动关闭头部画面"
                "（等价 --no-head；等模型重新生成后即可显示第一视角）。"
            )
        self._head_enabled = head_ok
        try:
            self.renderer = DualRenderer(self.model, head=head_ok)
        except CameraNotFoundError as exc:  # pragma: no cover - head_ok 已提前检查
            print(f"[提示] {exc}")
            self.renderer = DualRenderer(self.model, head=False)
            self._head_enabled = False

        if self.args.record:
            self.recorder = VideoRecorder(self.args.record, fps=config.RENDER_FPS)
            print(f"录制：{Path(self.args.record).resolve()}")

        if not self.args.headless:
            cv2 = require_cv2()
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            print(
                "窗口已打开：1-4 切阶段 / W S 调 vx / A D 调 vy / L R 切目标脚 / "
                "U standup / Space 复位 / F 跟随 / Esc 退出"
            )

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self) -> int:
        next_render = 0.0
        while self._running:
            if (
                self.args.seconds is not None
                and self.data.time >= float(self.args.seconds)
            ):
                break

            started = time.perf_counter()

            # 0) 复位/切阶段会把 data.time 归零；渲染节拍必须跟着回退，
            #    否则要等仿真时间爬回旧节拍，期间不渲染也不轮询窗口消息 → 假死。
            next_render = clamp_render_clock(next_render, self.data.time, RENDER_DT)

            # 1) 观测 → 控制器动作 → 位置伺服目标
            state = read_state(self.model, self.data, self.layout)
            self._last_state = state
            stage_config = config.STAGES[self.controller.stage]
            obs = self.obs_builder.build(
                state, self.command, gait=stage_config.gait
            )
            action = np.clip(self.controller.act(obs), -1.0, 1.0).astype(np.float64)
            self.obs_builder.push_action(action)
            self.data.ctrl[:] = action_to_ctrl(action, self.layout)

            # 2) 手动轻推（P 键）：在物理步之前设置外力，持续 0.1 s
            if self._manual_push_steps > 0:
                torso = self.layout.torso_body_id
                self.data.xfrc_applied[torso, :3] = self._manual_push_vec
                self._manual_push_steps -= 1
                if self._manual_push_steps == 0:
                    self.data.xfrc_applied[torso, :3] = 0.0

            # 3) 物理推进（一个控制周期 = DECIMATION 个仿真步）
            for _ in range(int(config.DECIMATION)):
                mujoco.mj_step(self.model, self.data)
            self._sim_steps += 1

            # 3) 渲染（按仿真时间限流到 RENDER_FPS）
            if self._need_render and self.data.time >= next_render:
                self._render()
                next_render = advance_render_clock(
                    next_render, self.data.time, RENDER_DT
                )

            # 4) 键盘/窗口消息：每个控制步都轮询，不依赖渲染
            self._poll_gui()

            # 5) 实时限速（--fast 不限速）
            if not self.args.fast:
                remaining = CONTROL_DT - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
        return 0

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def _render(self) -> None:
        assert self.renderer is not None
        self.follow_cam.update(self.model, self.data, self.layout)
        follow_rgb, head_rgb = self.renderer.render(
            self.model, self.data, self.follow_cam
        )
        self._update_fps()
        frame = compose_frame(
            follow_rgb,
            head_rgb,
            self._hud_lines(),
            head_enabled=self._head_enabled,
        )
        if self.recorder is not None:
            self.recorder.write(frame)
        if not self.args.headless:
            cv2 = require_cv2()
            cv2.imshow(WINDOW_NAME, frame)

    def _poll_gui(self) -> None:
        """每个控制步都轮询键盘/窗口消息，不把输入绑在渲染上。

        - 用 ``cv2.pollKey()``（非阻塞）优先，老版本退回 ``waitKey(1)``；
        - 即使某一帧渲染很慢、或帧率被节流，窗口也一直 pump 消息，按键立即可用；
        - 之前 ``waitKey`` 只在 ``_render()`` 里调用，渲染一停输入就"失联"，
          也是窗口被判定 Not Responding 的直接原因之一。
        """
        if self.args.headless:
            return
        cv2 = require_cv2()
        poll_key = getattr(cv2, "pollKey", None)
        raw = poll_key() if callable(poll_key) else cv2.waitKey(1)
        self._handle_key(int(raw) & 0xFF)

    def _update_fps(self) -> None:
        self._fps_frames += 1
        now = time.perf_counter()
        elapsed = now - self._fps_t0
        if elapsed >= 0.5:
            self._fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_t0 = now

    def _hud_lines(self) -> list[str]:
        state = self._last_state
        command = self.command
        task = config.TASK_NAMES[int(np.argmax(command[0:5]))]
        controller_line = (
            f"controller {self.controller.name}  stage {self.controller.stage}"
        )
        pose = self._current_standup_pose()
        if pose is not None:
            controller_line += f"  pose {pose.name}"
        yaw_error_deg = float(
            np.degrees(
                np.arctan2(
                    np.sin(float(command[config.CMD_YAW]) - state.yaw_rad),
                    np.cos(float(command[config.CMD_YAW]) - state.yaw_rad),
                )
            )
        )
        position_error = float(
            np.linalg.norm(np.asarray(state.base_pos[:2]) - self._target_xy)
        )
        return [
            controller_line,
            (
                f"cmd {task} vx={command[config.CMD_VX]:+.2f} "
                f"vy={command[config.CMD_VY]:+.2f} "
                f"lift_h={command[config.CMD_LIFT_H]:.3f} "
                f"foot={command[config.CMD_TARGET_FOOT]:+.0f} "
                f"yaw={np.degrees(command[config.CMD_YAW]):+.0f}deg"
            ),
            (
                f"base_z {state.base_pos[2]:.3f}m  "
                f"tilt {state.tilt_deg:5.1f}deg  "
                f"yaw_err {yaw_error_deg:+6.1f}deg  pos_err {position_error*100:4.1f}cm"
            ),
            (
                f"foot_z L{state.foot_z[0]:.3f} R{state.foot_z[1]:.3f}  "
                f"body {int(state.body_contact)}  tail {int(state.tail_contact)}"
            ),
            (
                f"fps {self._fps:5.1f}  t {state.time:6.2f}s  "
                f"{'FAST' if self.args.fast else 'realtime'}"
            ),
        ]

    # ------------------------------------------------------------------
    # 键盘
    # ------------------------------------------------------------------
    def _handle_key(self, key: int) -> None:
        if key in (-1, 255):
            return
        if key in (ESC, ord("q"), ord("Q")):
            self._running = False
            return
        if key in TASK_KEYS:
            self._switch_task(TASK_KEYS[key])
            return

        command = self.command
        if key in (ord("w"), ord("W")):
            command[config.CMD_VX] = float(
                np.clip(
                    command[config.CMD_VX] + 0.02, *config.WALK_VX_RANGE
                )
            )
        elif key in (ord("s"), ord("S")):
            command[config.CMD_VX] = float(
                np.clip(
                    command[config.CMD_VX] - 0.02, *config.WALK_VX_RANGE
                )
            )
        elif key in (ord("a"), ord("A")):
            command[config.CMD_VY] = float(
                np.clip(
                    command[config.CMD_VY] + 0.02, *config.WALK_VY_RANGE
                )
            )
        elif key in (ord("d"), ord("D")):
            command[config.CMD_VY] = float(
                np.clip(
                    command[config.CMD_VY] - 0.02, *config.WALK_VY_RANGE
                )
            )
        elif key in (ord("l"), ord("L")):
            command[config.CMD_TARGET_FOOT] = -1.0
            self._sync_target_foot()
        elif key in (ord("r"), ord("R")):
            command[config.CMD_TARGET_FOOT] = 1.0
            self._sync_target_foot()
        elif key in (ord("u"), ord("U")):
            # 只把命令切到 standup；控制器本身不动（standup 阶段仍需按 3 换控制器）。
            self.command = make_command("standup")
        elif key in (ord("z"), ord("Z"), ord("c"), ord("C")):
            # Z/C：turn 阶段的绝对朝向指令 ∓5°
            if self.controller.stage != "turn":
                print("[提示] Z/C 调朝向只在 turn 阶段生效（先按 5 切到 turn）。")
                return
            delta = -5.0 if key in (ord("z"), ord("Z")) else 5.0
            yaw_deg = float(np.degrees(self.command[config.CMD_YAW])) + delta
            self.command[config.CMD_YAW] = float(
                np.radians((yaw_deg + 180.0) % 360.0 - 180.0)
            )
        elif key in (ord("v"), ord("V")):
            # V：standup 阶段在 crouch / side / supine 之间循环，切完立即复位。
            if self.controller.stage != "standup":
                print("[提示] V 只在 standup 阶段生效（先按 3 切到 standup）。")
                return
            self._standup_pose_index = (
                self._standup_pose_index + 1
            ) % len(config.STANDUP_POSES)
            self._reset_episode()
        elif key == ord(" "):
            self._reset_episode()
        elif key in (ord("f"), ord("F")):
            tracking = self.follow_cam.toggle()
            print(f"跟随相机：{'tracking' if tracking else 'free（固定机位）'}")
        elif key in (ord("p"), ord("P")):
            self._trigger_push()

    def _trigger_push(self, force: float = 5.0, steps: int = 5) -> None:
        """手动推一下：水平随机方向、5 N、0.1 s。"""
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        self._manual_push_vec[:] = (force * np.cos(angle), force * np.sin(angle), 0.0)
        self._manual_push_steps = int(steps)
        print(f"轻推：{force:.1f} N × {steps} 步（0.1 s）")

    def _switch_task(self, stage: str) -> None:
        """1-5 键：重新加载该阶段控制器并复位。"""
        if stage == self.controller.stage:
            return
        new_controller = None
        error: Exception | None = None
        try:
            new_controller = load_controller(stage, expected_contract=self.contract)
        except Exception as exc:  # noqa: BLE001 - 没训练好是预期分支
            error = exc
            fallback = SCRIPTED_FALLBACK.get(stage)
            if fallback is not None:
                try:
                    new_controller = load_controller(fallback)
                except Exception as exc2:  # pragma: no cover - 内置名字不会失败
                    error = exc2
        if new_controller is None:
            message = str(error) if error is not None else "未知错误"
            print(f"[提示] 切换到 {stage!r} 失败：{message.splitlines()[0]}")
            return

        old = self.controller
        self.controller = new_controller
        if old is not new_controller:
            old.close()
        self._reset_episode()
        print(f"已切换控制器：{self.controller.name}（stage={self.controller.stage}）")

    def _reset_episode(self) -> None:
        """按当前阶段复位：standup 摆到课程起始姿态，其余回到站姿。"""
        self.data.xfrc_applied[:] = 0.0
        self._manual_push_steps = 0
        if self.controller.stage == "standup":
            pose = self._current_standup_pose()
            assert pose is not None
            apply_standup_pose(self.model, self.data, self.layout, pose)
            print(f"standup 起始姿态：{pose.name}（roll={pose.roll_deg:.0f}°）")
        else:
            reset_to_stand(self.model, self.data, self.layout)
            mujoco.mj_forward(self.model, self.data)
        self._last_state = read_state(self.model, self.data, self.layout)
        # 位置保持目标 = 复位后的基座 xy（与训练 reset 一致）。
        self._target_xy = np.asarray(
            self._last_state.base_pos[:2], dtype=np.float64
        ).copy()
        self.obs_builder.reset(self._target_xy)
        self.controller.reset()
        self.follow_cam.initialized = False
        self.command = self._new_command(self.controller.stage)
        self._sync_target_foot()

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
            print(
                f"录制完成：{self.recorder.path}"
                f"（{self.recorder.frames} 帧，{config.RENDER_FPS} fps）"
            )
            self.recorder = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self._need_render and not self.args.headless:
            try:
                require_cv2().destroyAllWindows()
            except ImportError:  # pragma: no cover - 正常运行不会走到
                pass
        try:
            self.controller.close()
        except Exception:  # pragma: no cover - close 不应影响退出码
            pass
        print(
            f"退出：sim_time={self.data.time:.2f}s，控制步数={self._sim_steps}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_stdio()
    try:
        runner = Runner(args)
    except Exception as exc:  # noqa: BLE001 - 启动失败统一友好提示
        print(f"[错误] 启动失败：{exc}", file=sys.stderr)
        return 1
    try:
        return runner.run()
    except KeyboardInterrupt:  # pragma: no cover - 手动 Ctrl+C
        print("收到 Ctrl+C，退出")
        return 0
    except Exception as exc:  # noqa: BLE001 - 运行期异常统一友好提示
        print(f"[错误] 运行失败：{exc}", file=sys.stderr)
        return 1
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())

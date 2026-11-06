"""小恐龙的 Gymnasium 环境（三阶段共用一套代码）。

一个 episode 的流程::

    reset()  →  选定初始姿态（站立 / 深蹲 / 侧躺 / 仰卧）+ 采样命令
    step(a)  →  a → ctrl（位置伺服）→ mj_step × decimation → 观测/奖励/终止

阶段差异全部由 ``config.STAGES`` 驱动：episode 长度、reset 方式、奖励权重、
是否推进步态时钟。训练端 ``train/`` 只负责 PPO；这里不依赖 stable-baselines3。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import config
from .action import action_to_ctrl
from .curriculum import Curriculum
from .model_io import JointLayout, load
from .observation import ObsBuilder, make_command
from .rewards import (
    compute_reward,
    is_standing,
    lift_stance_ok,
    out_of_bounds,
    potential,
)
from .state import (
    apply_standup_pose,
    read_state,
    reset_to_stand,
    settle,
    wrap_to_pi,
)

try:  # pragma: no cover
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("dino_wbc.env 需要 gymnasium") from exc

try:  # pragma: no cover
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError("dino_wbc.env 需要 mujoco") from exc


class DinoWbcEnv(gym.Env):
    """18 个位置伺服关节 + 浮动基座的 WBC 环境。"""

    metadata = {"render_modes": [], "render_fps": config.RENDER_FPS}

    def __init__(
        self,
        stage: str = "stand",
        *,
        model_path: str | None = None,
        kp: float | None = None,
        kv: float | None = None,
        forcerange: float | dict[str, float] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        if stage not in config.STAGES:
            raise ValueError(f"未知阶段 {stage!r}，可选 {tuple(config.STAGES)}")
        self.stage = stage
        self.stage_config = config.STAGES[stage]
        self.model, self.layout = load(
            model_path, kp=kp, kv=kv, forcerange=forcerange
        )
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.obs_builder = ObsBuilder(self.layout)
        self.curriculum = Curriculum(stage, self.rng)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(config.ACTION_DIM,), dtype=np.float32
        )
        self.max_episode_steps = self.stage_config.episode_steps

        self._command = make_command(stage)
        self._prev_action = np.zeros(config.ACTION_DIM, dtype=np.float64)
        self._step_count = 0
        self._success_streak = 0
        self._phi_prev = 0.0
        self._lift_reached_steps = 0
        self._lift_streak = 0  # 连续达标步数
        self._lift_best_streak = 0
        self._vel_track_sum = 0.0
        self._height_term_sum = 0.0
        self._heading_term_sum = 0.0
        self._position_term_sum = 0.0
        self._yaw_hold_streak = 0
        self._yaw_best_streak = 0
        self._episode_success = False
        self._episode_closed = False
        self._fell = False
        self._last_terms: dict[str, float] = {}
        #: 位置保持目标（复位后的基座 xy，世界系）。
        self._target_xy = np.zeros(2, dtype=np.float64)
        #: 朝向/位置持续违规计数（超过 grace 才终止）。
        self._heading_violation = 0
        self._position_violation = 0
        # ---- 外力轻推状态 ----
        self._push_remaining = 0
        self._next_push_step = 0
        self._push_force = 0.0  # 本次推力的模长（N）
        self._push_vec = np.zeros(3, dtype=np.float64)
        self._pushed = False

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.curriculum.rng = self.rng

        self._reset_pose()
        state = read_state(self.model, self.data, self.layout)
        # 位置目标 = 复位后的基座 xy；朝向目标是命令里的绝对角度（stand 为 0）。
        self._target_xy = np.asarray(state.base_pos[:2], dtype=np.float64).copy()
        self._command = self._sample_command()
        self.obs_builder.reset(self._target_xy)

        self._prev_action[:] = 0.0
        self._step_count = 0
        self._success_streak = 0
        self._phi_prev = potential(state, self.layout)
        self._lift_reached_steps = 0
        self._lift_streak = 0
        self._lift_best_streak = 0
        self._vel_track_sum = 0.0
        self._height_term_sum = 0.0
        self._heading_term_sum = 0.0
        self._position_term_sum = 0.0
        self._yaw_hold_streak = 0
        self._yaw_best_streak = 0
        self._heading_violation = 0
        self._position_violation = 0
        self._episode_success = False
        self._episode_closed = False
        self._fell = False
        self._reset_push()
        # standup 的成功必须"从低处起来"：防止浅蹲/站立初始姿态被误判为起身成功
        self._started_below_stand = bool(
            state.base_pos[2] < self.layout.stand_base_z - config.SUCCESS_Z_TOL
        )

        obs = self.obs_builder.build(
            state, self._command, gait=self.stage_config.gait
        )
        info = self._make_info(state, success=False, fallen=False)
        info["command"] = self._command.copy()
        return obs, info

    def _reset_pose(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0.0

        if self.stage_config.reset_mode == "stand":
            reset_to_stand(self.model, self.data, self.layout)
            self.data.qpos[self.layout.qposadr] += self.rng.uniform(
                -config.RESET_Q_NOISE, config.RESET_Q_NOISE, self.layout.num_joints
            )
            self.data.qpos[2] += self.rng.uniform(
                -config.RESET_BASE_Z_NOISE, config.RESET_BASE_Z_NOISE
            )
            self.data.qvel[:] = self.rng.uniform(
                -config.RESET_VEL_NOISE, config.RESET_VEL_NOISE, self.model.nv
            )
            mujoco.mj_forward(self.model, self.data)
            settle(self.model, self.data, self.layout)
            return

        # standup：按课程档位摆一个躺倒/深蹲姿态，先让物理落到地面。
        pose = self.curriculum.sample_standup_pose()
        apply_standup_pose(self.model, self.data, self.layout, pose)

    def _sample_command(self) -> np.ndarray:
        if self.stage == "lift":
            return make_command(
                "lift",
                lift_h=self.curriculum.sample_lift_h(),
                target_foot=self.curriculum.sample_lift_target(),
            )
        if self.stage == "walk":
            vx, vy = self.curriculum.sample_velocity()
            return make_command("walk", vx=vx, vy=vy)
        if self.stage == "turn":
            return make_command("turn", yaw=self.curriculum.sample_yaw())
        return make_command(self.stage, yaw=0.0)

    # ------------------------------------------------------------------
    # 外力轻推
    # ------------------------------------------------------------------
    def _reset_push(self) -> None:
        self._push_remaining = 0
        self._push_force = 0.0
        self._push_vec[:] = 0.0
        self._pushed = False
        self.data.xfrc_applied[:] = 0.0
        push = self.stage_config.push
        if push is None or not push.enabled:
            self._next_push_step = 1 << 30
            return
        self._next_push_step = int(self.rng.integers(*push.interval_steps))

    def _update_push(self) -> None:
        """在物理步之前设置/清除 ``xfrc_applied``。"""
        push = self.stage_config.push
        if push is None or not push.enabled:
            return
        torso = self.layout.torso_body_id
        if self._push_remaining > 0:
            self.data.xfrc_applied[torso, :3] = self._push_vec
            self._push_remaining -= 1
            if self._push_remaining == 0:
                self.data.xfrc_applied[torso, :3] = 0.0
            return
        if self._step_count >= self._next_push_step:
            force = float(self.curriculum.sample_push_force())
            angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
            self._push_force = force
            self._push_vec[:] = (force * np.cos(angle), force * np.sin(angle), 0.0)
            self._push_remaining = max(int(push.duration_steps) - 1, 0)
            self._next_push_step = self._step_count + int(
                self.rng.integers(*push.interval_steps)
            )
            self._pushed = True
            self.data.xfrc_applied[torso, :3] = self._push_vec

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action):
        raw = np.clip(
            np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0
        )
        if raw.shape[0] != config.ACTION_DIM:
            raise ValueError(f"动作维度应为 {config.ACTION_DIM}，实际 {raw.shape[0]}")

        ctrl = action_to_ctrl(raw, self.layout)
        self.data.ctrl[:] = ctrl
        self._pushed = False
        self._update_push()
        for _ in range(config.DECIMATION):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        state = read_state(self.model, self.data, self.layout)
        finite = bool(
            np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        )

        fallen = False
        if not finite:
            fallen = True
        elif self.stage_config.reset_mode == "standup":
            # 躺倒阶段不按站姿判摔倒；滚出界仍算失败。
            fallen = out_of_bounds(state)
        else:
            cfg = self.stage_config
            fall_z = (
                self.layout.stand_base_z - cfg.fall_z_delta
                if cfg.fall_z_delta is not None
                else config.FALL_BASE_Z
            )
            fall_tilt = (
                cfg.fall_tilt_deg
                if cfg.fall_tilt_deg is not None
                else config.FALL_TILT_DEG
            )
            # 绝对朝向/位置偏差：连续违规超过 grace 步才终止（避免瞬时尖峰误杀）。
            heading_error_deg = abs(
                np.degrees(
                    wrap_to_pi(float(self._command[config.CMD_YAW]) - state.yaw_rad)
                )
            )
            position_error = float(
                np.linalg.norm(np.asarray(state.base_pos[:2]) - self._target_xy)
            )
            if (
                cfg.max_heading_error_deg is not None
                and heading_error_deg > cfg.max_heading_error_deg
            ):
                self._heading_violation += 1
            else:
                self._heading_violation = 0
            if (
                cfg.max_position_error_m is not None
                and position_error > cfg.max_position_error_m
            ):
                self._position_violation += 1
            else:
                self._position_violation = 0
            grace = max(int(cfg.heading_grace_steps), 1)
            fallen = bool(
                state.base_pos[2] < fall_z
                or state.tilt_deg > fall_tilt
                or out_of_bounds(state)
                or (cfg.terminate_on_body_contact and state.body_contact)
                or (cfg.terminate_on_tail_contact and state.tail_contact)
                or self._heading_violation >= grace
                or self._position_violation >= grace
            )
        self._fell = self._fell or fallen

        # standup：以"连续站立 0.5 s"为成功；躺倒阶段不判跌倒。
        success = False
        if self.stage_config.reset_mode == "standup":
            if self._started_below_stand and is_standing(state, self.layout):
                self._success_streak += 1
            else:
                self._success_streak = 0
            success = self._success_streak >= config.SUCCESS_HOLD_STEPS
            if not finite:
                fallen = True
        else:
            self._success_streak = 0

        phi = potential(state, self.layout) if self.stage == "standup" else 0.0
        # 先推进步态时钟再算奖励，保证奖励里的摆动足相位与观测一致
        obs = self.obs_builder.build(
            state, self._command, gait=self.stage_config.gait
        )
        phase_now = self.obs_builder.phase if self.stage_config.gait else 0.0
        reward, terms = compute_reward(
            self.stage,
            state,
            raw,
            self._prev_action,
            self._command,
            self.layout,
            phase=phase_now,
            phi=phi,
            phi_prev=self._phi_prev if self.stage == "standup" else None,
            success=success,
            fallen=fallen,
            target_xy=self._target_xy,
        )
        self._phi_prev = phi
        self._last_terms = terms

        # 课程用的"能力达标"统计
        self._track_curriculum_metrics(state, terms, fallen)

        terminated = bool(fallen or success)
        truncated = bool(
            not terminated and self._step_count >= self.max_episode_steps
        )

        self.obs_builder.push_action(raw)
        self._prev_action = raw

        info = self._make_info(state, success=success, fallen=fallen)
        info["reward_terms"] = terms
        info["TimeLimit.truncated"] = truncated

        if terminated or truncated:
            episode_success = self._episode_outcome(success)
            info["episode_success"] = episode_success
            info["curriculum"] = self.curriculum.state().__dict__
            if not self._episode_closed:
                self.curriculum.on_episode(episode_success)
                self._episode_closed = True

        return obs, float(reward), terminated, truncated, info

    def _track_curriculum_metrics(
        self, state, terms: dict[str, float], fallen: bool
    ) -> None:
        """累计"这一步算不算达标"，episode 结束时再判定成败。

        抬脚改为**连续保持**（``_lift_streak``）：v1 用"累计 25 步"导致
        跌倒过程中脚瞬时冲到 3–4 cm 也被算成功，课程因此被骗到 4 cm。
        """
        if fallen:
            self._lift_streak = 0
            self._yaw_hold_streak = 0
            return
        if self.stage == "lift":
            target = 0 if self._command[config.CMD_TARGET_FOOT] < 0 else 1
            wanted = max(float(self._command[config.CMD_LIFT_H]), 1e-3)
            support = 1 - target
            reached = (
                state.foot_z[target] >= config.LIFT_HEIGHT_RATIO * wanted
                and bool(state.contact[support])
                and lift_stance_ok(state, self.layout)
            )
            if reached:
                self._lift_reached_steps += 1
                self._lift_streak += 1
                self._lift_best_streak = max(self._lift_best_streak, self._lift_streak)
            else:
                self._lift_streak = 0
        elif self.stage == "walk":
            self._vel_track_sum += terms.get("vel_track", 0.0)
        elif self.stage == "stand":
            self._height_term_sum += terms.get("height", 0.0)
            self._heading_term_sum += terms.get("heading", 0.0)
            self._position_term_sum += terms.get("position", 0.0)
        elif self.stage == "turn":
            heading_error_deg = abs(
                np.degrees(
                    wrap_to_pi(float(self._command[config.CMD_YAW]) - state.yaw_rad)
                )
            )
            position_error = float(
                np.linalg.norm(np.asarray(state.base_pos[:2]) - self._target_xy)
            )
            ok = (
                heading_error_deg <= config.TURN_HOLD_HEADING_DEG
                and position_error <= config.TURN_HOLD_POSITION
            )
            if ok:
                self._yaw_hold_streak += 1
                self._yaw_best_streak = max(self._yaw_best_streak, self._yaw_hold_streak)
            else:
                self._yaw_hold_streak = 0

    def _episode_outcome(self, success_now: bool) -> bool:
        """把阶段指标折算成课程意义上的"成功"。"""
        steps = max(self._step_count, 1)
        if self.stage == "lift":
            self._episode_success = (
                (not self._fell)
                and self._lift_best_streak >= config.LIFT_HOLD_STEPS
            )
        elif self.stage == "walk":
            self._episode_success = (
                not self._fell
            ) and self._vel_track_sum / steps > 0.60
        elif self.stage == "stand":
            meaningfully_standing = (
                self._height_term_sum / steps > 0.80
                and self._heading_term_sum / steps > 0.80
                and self._position_term_sum / steps > 0.80
            )
            self._episode_success = (not self._fell) and meaningfully_standing
        elif self.stage == "turn":
            self._episode_success = (
                (not self._fell)
                and self._yaw_best_streak >= config.TURN_HOLD_STEPS
            )
        else:
            self._episode_success = bool(self._episode_success or success_now)
        return bool(self._episode_success)

    def _make_info(self, state, *, success: bool, fallen: bool) -> dict[str, Any]:
        target = 0 if self._command[config.CMD_TARGET_FOOT] < 0 else 1
        heading_error_deg = float(
            np.degrees(
                wrap_to_pi(float(self._command[config.CMD_YAW]) - state.yaw_rad)
            )
        )
        position_error = float(
            np.linalg.norm(np.asarray(state.base_pos[:2]) - self._target_xy)
        )
        return {
            "stage": self.stage,
            "success": bool(success),
            "fallen": bool(fallen),
            "base_z": float(state.base_pos[2]),
            "tilt_deg": float(state.tilt_deg),
            "yaw_deg": float(np.degrees(state.yaw_rad)),
            "yaw_cmd_deg": float(np.degrees(self._command[config.CMD_YAW])),
            "heading_error_deg": heading_error_deg,
            "position_error": position_error,
            "contact": state.contact.copy(),
            "foot_z": state.foot_z.copy(),
            "command": self._command.copy(),
            "step": self._step_count,
            "curriculum": self.curriculum.state().__dict__,
            "body_contact": bool(state.body_contact),
            "tail_contact": bool(state.tail_contact),
            "target_foot_z": float(state.foot_z[target]),
            "support_contact": bool(state.contact[1 - target]),
            "lift_streak": int(self._lift_streak),
            "lift_best_streak": int(self._lift_best_streak),
            "yaw_hold_streak": int(self._yaw_hold_streak),
            "yaw_best_streak": int(self._yaw_best_streak),
            "pushed": bool(self._pushed),
            "push_force": float(self._push_force),
        }

    # ------------------------------------------------------------------
    # 其它
    # ------------------------------------------------------------------
    def close(self) -> None:
        self.data = None

    def set_command(self, command: np.ndarray) -> None:
        """运行端/评测端直接改命令（训练时由 reset 采样）。"""
        self._command = np.asarray(command, dtype=np.float64).copy()

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    @property
    def layout_info(self) -> JointLayout:
        return self.layout

    def render(self):  # pragma: no cover - 运行端自己做渲染
        raise NotImplementedError("请用 dino_wbc.render / run/run.py 渲染")

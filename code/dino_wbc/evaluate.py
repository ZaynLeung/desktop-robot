"""控制器评测：在环境里跑 N 个 episode，输出成功率/跌倒率/跟踪误差等指标。

运行端和训练端都可以用：不依赖 stable-baselines3，只要求控制器实现
:class:`dino_wbc.controllers.base.Controller` 接口。
"""

from __future__ import annotations

import numpy as np

from . import config
from .env import DinoWbcEnv


def evaluate(
    controller,
    *,
    episodes: int = 10,
    model_path: str | None = None,
    seed: int = 0,
    max_steps: int | None = None,
    standup_pose: str = "all",
) -> dict:
    """跑 ``episodes`` 个 episode，返回汇总指标与逐 episode 记录。

    :param standup_pose: 仅 standup 阶段有效；``"all"`` 时在 crouch / side / supine
        之间轮换（默认，能看出是哪一档没学会），也可固定成某一档的名字。
    """
    stage = getattr(controller, "stage", "lift")
    env = DinoWbcEnv(stage, seed=seed, model_path=model_path)
    limit = max_steps or config.STAGES[stage].episode_steps

    pose_names = [pose.name for pose in config.STANDUP_POSES]
    if stage == "standup" and standup_pose not in ("all", *pose_names):
        raise ValueError(
            f"未知 standup 起始姿态 {standup_pose!r}，可选 'all' 或 {pose_names}"
        )

    records: list[dict] = []
    for episode in range(episodes):
        if stage == "standup":
            if standup_pose == "all":
                env.curriculum.pose_index = episode % len(config.STANDUP_POSES)
            else:
                env.curriculum.pose_index = pose_names.index(standup_pose)
        controller.reset()
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0.0
        vel_track_sum = 0.0
        steps = 0
        success = False
        fallen = False
        while steps < limit:
            action = controller.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            vel_track_sum += info.get("reward_terms", {}).get("vel_track", 0.0)
            steps += 1
            success = bool(info.get("success"))
            fallen = bool(info.get("fallen"))
            if terminated or truncated:
                break
        records.append(
            {
                "episode": episode,
                "pose": (
                    config.STANDUP_POSES[env.curriculum.pose_index].name
                    if stage == "standup"
                    else None
                ),
                "success": bool(info.get("episode_success", success)),
                "fallen": fallen,
                "reward": episode_reward,
                "steps": steps,
                "base_z": float(info.get("base_z", 0.0)),
                "tilt_deg": float(info.get("tilt_deg", 0.0)),
                "mean_vel_track": vel_track_sum / max(steps, 1),
            }
        )

    env.close()
    successes = [record["success"] for record in records]
    falls = [record["fallen"] for record in records]
    summary = {
        "stage": stage,
        "episodes": episodes,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "fall_rate": float(np.mean(falls)) if falls else 0.0,
        "mean_reward": float(np.mean([r["reward"] for r in records])),
        "mean_base_z": float(np.mean([r["base_z"] for r in records])),
        "mean_tilt_deg": float(np.mean([r["tilt_deg"] for r in records])),
        "mean_vel_track": float(np.mean([r["mean_vel_track"] for r in records])),
    }
    return {"summary": summary, "episodes": records}

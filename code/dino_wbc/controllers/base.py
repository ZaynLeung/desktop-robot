"""统一控制器接口。

运行端（``code/run/run.py``）只认这一个接口：给 80 维观测、返回 ``[-1, 1]^18``
的归一化动作。脚本控制器和策略控制器都实现它，运行端不关心动作从哪里来。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .. import config


class Controller(ABC):
    """策略/脚本控制器的统一接口。

    子类至少要实现 :meth:`act`，并设置 :attr:`name` 与 :attr:`stage`。
    其它方法都有默认实现（无状态控制器不需要覆盖）。
    """

    #: 展示名，如 ``"lift@v1"``。
    name: str = "controller"
    #: 阶段：``stand`` / ``lift`` / ``standup`` / ``walk``；运行端据此构造 command。
    stage: str = "stand"
    #: 输入观测维度（80 维）。
    obs_dim: int = config.OBS_DIM
    #: 输出动作维度（18 维）。
    act_dim: int = config.ACTION_DIM

    def reset(self) -> None:
        """重置内部状态（计时器、缓存等）。无状态控制器什么都不用做。"""

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        """观测 → 归一化动作。

        返回形状 ``(act_dim,)``、dtype ``float32``、范围 ``[-1, 1]`` 的数组；
        越界会被运行端 / 环境统一裁剪。
        """

    def close(self) -> None:
        """释放资源（例如卸载策略网络）。默认无操作。"""

    def describe(self) -> dict[str, object]:
        """给 HUD / 日志用的元信息。"""
        return {
            "name": self.name,
            "stage": self.stage,
            "obs_dim": int(self.obs_dim),
            "act_dim": int(self.act_dim),
        }

    def _check_obs(self, obs: np.ndarray) -> np.ndarray:
        """把观测规整成一维 float32，并校验维度。"""
        values = np.asarray(obs, dtype=np.float32).reshape(-1)
        if values.shape[0] != int(self.obs_dim):
            raise ValueError(
                f"观测维度应为 {int(self.obs_dim)}，实际 {values.shape[0]}"
            )
        return values

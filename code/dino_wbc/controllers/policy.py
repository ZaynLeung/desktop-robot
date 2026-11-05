"""策略控制器：把 ``policy_io`` 的 ActorNet 包成运行端统一接口。

运行端只做三件事：``obs → act() → 归一化动作``。这里固定 CPU、``float32``、
``batch=1``、``torch.no_grad()``，保证在玩家机器上跑得动、也不会悄悄改变行为。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config
from ..policy_io import load_policy, require_torch
from .base import Controller


class PolicyController(Controller):
    """加载 checkpoint 的确定性策略控制器。

    ``stage`` 来自 checkpoint 的 ``meta["stage"]``（缺省 ``"lift"``），运行端据此
    构造 8 维 command；``contract_hash`` 不匹配会抛
    :class:`~dino_wbc.policy_io.PolicyContractError`。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        expected_contract: str | None = None,
        name: str | None = None,
    ) -> None:
        torch = require_torch()
        self.path = Path(path)
        net, bundle = load_policy(self.path, expected_contract=expected_contract)
        # 强制 CPU / float32：训练导出的权重可能是 float64 或残留梯度。
        self.net = net.to("cpu").float().eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

        self.bundle = bundle
        self.contract_hash = str(bundle["contract_hash"])
        self.obs_dim = int(bundle["obs_dim"])
        self.act_dim = int(bundle["act_dim"])

        meta = dict(bundle.get("meta") or {})
        stage = str(meta.get("stage", "lift"))
        if stage not in config.STAGES:
            raise ValueError(
                f"checkpoint 里的 stage={stage!r} 不在 {tuple(config.STAGES)}；"
                "训练导出时请把 --stage 写进 meta"
            )
        self.stage = stage
        self.meta = meta
        self.name = name or f"{stage}@{self.path.stem}"

        # 复用的输入缓冲：避免每帧分配。
        self._obs_buffer = torch.zeros(1, self.obs_dim, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Controller 接口
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """策略本身无内部状态（观测里已含 prev_action 与相位）。"""

    def act(self, obs: np.ndarray) -> np.ndarray:
        torch = require_torch()
        values = self._check_obs(obs)
        with torch.no_grad():
            self._obs_buffer[0, :] = torch.from_numpy(values)
            out = self.net(self._obs_buffer)
        action = out.detach().cpu().numpy().reshape(-1)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def describe(self) -> dict[str, object]:
        info = super().describe()
        net_info = dict(self.bundle.get("net") or {})
        info.update(
            {
                "policy": "actor-net",
                "checkpoint": str(self.path),
                "contract_hash": self.contract_hash,
                "hidden": list(net_info.get("hidden", [])),
                "activation": net_info.get("activation", "elu"),
                "device": "cpu",
                "meta": dict(self.meta),
            }
        )
        return info

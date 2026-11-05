"""训练端的 SB3 回调：把课程进度、奖励分项写进 TensorBoard，并导出运行端格式的策略。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from dino_wbc import config
from dino_wbc.policy_io import export_from_sb3


def _scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


class CurriculumLogCallback(BaseCallback):
    """记录各向量环境里最新的课程进度与奖励分项均值。"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._latest_curriculum: dict[str, float] = {}
        self._term_sums: dict[str, float] = defaultdict(float)
        self._term_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            curriculum = info.get("curriculum")
            if isinstance(curriculum, dict):
                for key, value in curriculum.items():
                    scalar = _scalar(value)
                    if scalar is not None:
                        self._latest_curriculum[key] = scalar
            terms = info.get("reward_terms")
            if isinstance(terms, dict):
                for key, value in terms.items():
                    scalar = _scalar(value)
                    if scalar is not None:
                        self._term_sums[key] += scalar
                self._term_count += 1
        return True

    def _on_rollout_end(self) -> None:
        for key, value in self._latest_curriculum.items():
            self.logger.record(f"curriculum/{key}", value)
        if self._term_count:
            for key, total in self._term_sums.items():
                self.logger.record(f"reward_terms/{key}", total / self._term_count)
        self._term_sums.clear()
        self._term_count = 0


class ExportBestPolicyCallback(EvalCallback):
    """在 eval 指标刷新时，把确定性策略导出成 ``policy.pt``（运行端格式）。

    同时保留 SB3 自己的 ``best_model.zip``，方便用 ``--init-from`` 继续训练。
    """

    def __init__(
        self,
        *args,
        export_path: str | Path,
        contract_hash: str,
        stage: str,
        joint_names: tuple[str, ...] = config.JOINT_NAMES,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.export_path = Path(export_path)
        self.contract_hash = contract_hash
        self.stage = stage
        self.joint_names = joint_names
        self._last_exported = float("-inf")

    def _on_step(self) -> bool:
        keep = super()._on_step()
        if self.best_mean_reward > self._last_exported:
            export_from_sb3(
                self.model,
                self.export_path,
                contract_hash=self.contract_hash,
                joint_names=self.joint_names,
                meta={
                    "stage": self.stage,
                    "eval_mean_reward": float(self.best_mean_reward),
                    "num_timesteps": int(self.num_timesteps),
                },
            )
            self._last_exported = self.best_mean_reward
            if self.verbose:
                print(f"[export] 已更新 {self.export_path}（eval={self.best_mean_reward:.3f}）")
        return keep

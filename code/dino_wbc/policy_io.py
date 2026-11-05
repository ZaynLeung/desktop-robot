"""策略 checkpoint 的存取（训练端导出 / 运行端加载共用）。

一个 checkpoint 里同时保存「网络权重」和「模型契约」：

* ``format``     固定 ``dino_wbc.policy.v1``，用来拒绝旧/异构文件；
* ``obs_dim`` / ``act_dim`` / ``joint_names`` 保证观测与关节顺序没变；
* ``contract_hash`` 是 :func:`dino_wbc.model_io.contract_hash` 的指纹，模型尺寸/
  关节限位改过就拒绝加载，避免"静默地用错动作"；
* ``net`` 存建网络需要的 ``hidden`` / ``activation``；
* ``meta`` 放训练端附加信息（``stage``、训练步数、奖励权重等），运行端只读。

所有 torch 相关导入都在模块顶层用 ``try/except`` 兜住：没装 torch 时模块仍可
import（运行端其它模块会连锁 import 到这里），真正调用 save/load 时再抛出带安装
命令的 :class:`ImportError`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from . import config

#: checkpoint 格式标识；改结构时递增版本号。
FORMAT = "dino_wbc.policy.v1"

#: 契约不匹配时统一的 CLI 提示（用户能直接照做）。
_CONTRACT_HINT = "模型改过，请重新训练或换 checkpoint"
_TORCH_INSTALL_HINT = (
    "需要 PyTorch（训练端导出 / 运行端加载策略）：\n"
    r"  E:\conda_envs\desktop-robot\python.exe -m pip install torch stable-baselines3"
)

try:  # pragma: no cover - 缺 torch 时模块仍可 import
    import torch
    from torch import nn

    _TORCH_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _exc


class PolicyContractError(RuntimeError):
    """checkpoint 与当前模型 / 配置不兼容。"""


def require_torch():
    """延迟取 torch 模块；未安装时给出可执行的安装提示。"""
    if torch is None:  # pragma: no cover - 只在未装 torch 的环境触发
        raise ImportError(_TORCH_INSTALL_HINT) from _TORCH_IMPORT_ERROR
    return torch


def _activation_module(name: str):
    """名字 → 激活层实例（只支持这批常用激活）。"""
    modules = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    key = str(name).strip().lower()
    if key not in modules:
        raise ValueError(f"不支持的激活函数 {name!r}，可选 {tuple(modules)}")
    return modules[key]()


# ----------------------------------------------------------------------
# 网络
# ----------------------------------------------------------------------
if nn is not None:

    class ActorNet(nn.Module):
        """确定性 Actor：``obs → 18 维动作均值``。

        结构：``Linear(hidden[0]) → act → Linear(hidden[1]) → act → ... → Linear(18)``。
        **最后一层不加 tanh**（环境/运行端统一裁剪到 ``[-1, 1]``，也保留大动作
        的梯度信息；SB3 导出时两边的 logits 完全一致）。
        """

        def __init__(
            self,
            obs_dim: int = config.OBS_DIM,
            act_dim: int = config.ACTION_DIM,
            hidden: Sequence[int] = (128, 128),
            activation: str = "elu",
        ) -> None:
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.act_dim = int(act_dim)
            self.hidden = [int(width) for width in hidden]
            self.activation = str(activation).strip().lower()

            layers: list[nn.Module] = []
            last = self.obs_dim
            for width in self.hidden:
                layers.append(nn.Linear(last, width))
                layers.append(_activation_module(self.activation))
                last = width
            layers.append(nn.Linear(last, self.act_dim))  # 不加 tanh
            self.mlp = nn.Sequential(*layers)

        def forward(self, obs):  # noqa: D401 - torch 约定名
            return self.mlp(obs)

else:  # pragma: no cover - 占位符：保证模块可 import、调用时报清晰错误

    class ActorNet:  # type: ignore[no-redef]
        """torch 未安装时的占位类；实例化会抛出带安装提示的 ImportError。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


# ----------------------------------------------------------------------
# 保存 / 加载
# ----------------------------------------------------------------------
def save_policy(
    path: str | Path,
    model: "ActorNet",
    *,
    contract_hash: str,
    joint_names: tuple[str, ...],
    hidden: list[int],
    activation: str,
    meta: dict | None = None,
) -> Path:
    """把 ActorNet 及其契约信息保存成 ``dino_wbc.policy.v1`` checkpoint。"""
    require_torch()
    if not isinstance(model, nn.Module):
        raise TypeError(f"save_policy 期望 torch.nn.Module，实际 {type(model)!r}")

    target = Path(path)
    if target.parent and str(target.parent):
        target.parent.mkdir(parents=True, exist_ok=True)

    obs_dim = int(getattr(model, "obs_dim", config.OBS_DIM))
    act_dim = int(getattr(model, "act_dim", config.ACTION_DIM))
    if (obs_dim, act_dim) != (config.OBS_DIM, config.ACTION_DIM):
        raise ValueError(
            f"网络维度 ({obs_dim}, {act_dim}) 与 config "
            f"({config.OBS_DIM}, {config.ACTION_DIM}) 不一致"
        )

    bundle = {
        "format": FORMAT,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "joint_names": tuple(str(name) for name in joint_names),
        "contract_hash": str(contract_hash),
        "net": {
            "hidden": [int(width) for width in hidden],
            "activation": str(activation),
        },
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "meta": dict(meta or {}),
    }
    torch.save(bundle, target)
    return target


def _validate_bundle(
    bundle: Any,
    *,
    path: Path,
    expected_contract: str | None,
) -> None:
    """逐项校验 checkpoint 与当前配置的兼容性。"""
    if not isinstance(bundle, Mapping) or bundle.get("format") != FORMAT:
        raise PolicyContractError(
            f"checkpoint 格式不匹配：{path}\n"
            f"  期望 format={FORMAT!r}，实际 {bundle.get('format') if isinstance(bundle, Mapping) else type(bundle)!r}\n"
            f"{_CONTRACT_HINT}。"
        )

    problems: list[str] = []
    obs_dim = bundle.get("obs_dim")
    act_dim = bundle.get("act_dim")
    if obs_dim != config.OBS_DIM:
        problems.append(f"obs_dim：checkpoint={obs_dim}，当前 config={config.OBS_DIM}")
    if act_dim != config.ACTION_DIM:
        problems.append(
            f"act_dim：checkpoint={act_dim}，当前 config={config.ACTION_DIM}"
        )

    names = bundle.get("joint_names")
    names = tuple(names) if names is not None else ()
    if names != tuple(config.JOINT_NAMES):
        problems.append(
            f"joint_names：checkpoint={names}，当前 config={tuple(config.JOINT_NAMES)}"
        )

    if expected_contract is not None:
        actual = bundle.get("contract_hash")
        if actual != expected_contract:
            problems.append(
                f"contract_hash：checkpoint={actual}，当前模型={expected_contract}"
            )

    if problems:
        detail = "\n".join(f"  - {item}" for item in problems)
        raise PolicyContractError(
            f"checkpoint 与当前模型不兼容：{path}\n{detail}\n{_CONTRACT_HINT}。"
        )


def load_policy(
    path: str | Path,
    *,
    expected_contract: str | None = None,
) -> tuple["ActorNet", dict]:
    """加载 checkpoint → ``(ActorNet, bundle)``，并校验契约。

    :param expected_contract: 当前模型的 :func:`dino_wbc.model_io.contract_hash`；
        传入 ``None`` 时只校验格式/维度/关节顺序，不比对哈希。
    """
    require_torch()
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"找不到策略 checkpoint：{target}\n"
            "先训练该阶段（见 code/train/），或用 --controller scripted_stand 冒烟。"
        )

    try:
        # 先按 weights_only=True 读（只含张量/基础类型，更安全）；旧格式再退化。
        try:
            bundle = torch.load(target, map_location="cpu", weights_only=True)
        except Exception:
            bundle = torch.load(target, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - 损坏文件属于异常路径
        raise PolicyContractError(
            f"无法读取 checkpoint：{target}\n  {exc}\n{_CONTRACT_HINT}。"
        ) from exc

    _validate_bundle(bundle, path=target, expected_contract=expected_contract)

    net_info = dict(bundle.get("net") or {})
    hidden = [int(width) for width in net_info.get("hidden", [])]
    activation = str(net_info.get("activation", "elu"))
    net = ActorNet(
        int(bundle["obs_dim"]),
        int(bundle["act_dim"]),
        hidden,
        activation,
    )
    try:
        net.load_state_dict(bundle["state_dict"], strict=True)
    except Exception as exc:
        raise PolicyContractError(
            f"网络结构与 checkpoint 不一致：{target}\n  {exc}\n{_CONTRACT_HINT}。"
        ) from exc
    net.eval().float()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net, dict(bundle)


# ----------------------------------------------------------------------
# 从 SB3 PPO 导出
# ----------------------------------------------------------------------
def _infer_activation(policy_net) -> str:
    """从 SB3 的 ``policy_net`` 里认出激活函数名（认不出就按 SB3 默认 elu）。"""
    mapping = {
        nn.ELU: "elu",
        nn.ReLU: "relu",
        nn.Tanh: "tanh",
        nn.GELU: "gelu",
        nn.SiLU: "silu",
    }
    for module in policy_net.modules():
        for cls, name in mapping.items():
            if isinstance(module, cls):
                return name
    return "elu"


def export_from_sb3(
    sb3_model,
    path: str | Path,
    *,
    contract_hash: str,
    joint_names: tuple[str, ...],
    meta: dict | None = None,
) -> Path:
    """把 SB3 PPO 的确定性 Actor 权重导出成 :class:`ActorNet` checkpoint。

    SB3 的结构是 ``policy.mlp_extractor.policy_net``（``Linear→act→Linear→act``）再接
    ``policy.action_net``（``Linear(→act_dim)``）。这里**不假设层数**：按
    ``policy_net`` 里 ``nn.Linear`` 的注册顺序反推 ``hidden``，再逐层拷贝权重。

    :param sb3_model: PPO 实例（也接受直接传 ``policy``）；
    :param meta: 建议至少带上 ``{"stage": "lift"}``，运行端据此构造命令。
    """
    require_torch()

    policy = getattr(sb3_model, "policy", sb3_model)
    mlp_extractor = getattr(policy, "mlp_extractor", None)
    policy_net = getattr(mlp_extractor, "policy_net", None)
    action_net = getattr(policy, "action_net", None)
    if policy_net is None or action_net is None:
        # 不是 SB3 对象时，给出「装包/换对象」两条提示。
        try:
            import stable_baselines3  # noqa: F401
        except ImportError as exc:  # pragma: no cover - 未装 SB3 的环境
            raise ImportError(
                "export_from_sb3 需要 stable-baselines3（或一个带 "
                "policy.mlp_extractor.policy_net / policy.action_net 的对象）：\n"
                r"  E:\conda_envs\desktop-robot\python.exe -m pip install stable-baselines3"
            ) from exc
        raise TypeError(
            "export_from_sb3 期望 SB3 PPO 模型（需要 policy.mlp_extractor.policy_net "
            f"与 policy.action_net），实际 {type(sb3_model)!r}"
        )

    # 按注册顺序收集 policy_net 里的 Linear（SB3 是 Sequential，顺序即深度）。
    linear_modules: list[tuple[str, Any]] = [
        (name, module)
        for name, module in policy_net.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if not linear_modules:
        raise TypeError("policy_net 里没有找到 nn.Linear，无法导出")
    hidden = [int(module.out_features) for _, module in linear_modules]
    obs_dim = int(linear_modules[0][1].in_features)
    act_dim = int(action_net.out_features)
    if obs_dim != config.OBS_DIM or act_dim != config.ACTION_DIM:
        raise ValueError(
            "SB3 网络的输入/输出维度与 config 不一致："
            f"obs_dim={obs_dim}（期望 {config.OBS_DIM}），"
            f"act_dim={act_dim}（期望 {config.ACTION_DIM}）"
        )

    activation = _infer_activation(policy_net)
    net = ActorNet(obs_dim, act_dim, hidden, activation)

    # 逐层拷贝：policy_net 的第 i 个 Linear → ActorNet.mlp 的第 2i 层；
    # 最后一层（无激活）→ mlp 的最后一个 Linear。
    policy_state = policy_net.state_dict()
    action_state = action_net.state_dict()
    new_state: dict[str, Any] = {}
    for index, (module_name, _module) in enumerate(linear_modules):
        prefix = f"{module_name}." if module_name else ""
        new_state[f"mlp.{2 * index}.weight"] = policy_state[prefix + "weight"]
        new_state[f"mlp.{2 * index}.bias"] = policy_state[prefix + "bias"]
    last = 2 * len(linear_modules)
    new_state[f"mlp.{last}.weight"] = action_state["weight"]
    new_state[f"mlp.{last}.bias"] = action_state["bias"]
    net.load_state_dict(new_state, strict=True)
    net.eval().float()

    exported_meta = dict(meta or {})
    exported_meta.setdefault("source", "stable_baselines3")
    return save_policy(
        path,
        net,
        contract_hash=contract_hash,
        joint_names=tuple(joint_names),
        hidden=hidden,
        activation=activation,
        meta=exported_meta,
    )

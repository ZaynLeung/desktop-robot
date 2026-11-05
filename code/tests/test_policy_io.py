"""``policy_io`` 的存取/契约校验/导出测试。

缺 torch 时整个文件跳过（运行端在无 torch 的机器上仍可用脚本控制器）。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dino_wbc import config  # noqa: E402  (必须在 importorskip 之后)
from dino_wbc.policy_io import (  # noqa: E402
    FORMAT,
    ActorNet,
    PolicyContractError,
    export_from_sb3,
    load_policy,
    save_policy,
)

#: 测试里用的假契约哈希。
CONTRACT = "deadbeef01234567"


def _make_net(hidden: list[int] | None = None) -> ActorNet:
    """建一个权重随机的 ActorNet（同样随机 hidden，方便测不同层数）。"""
    net = ActorNet(config.OBS_DIM, config.ACTION_DIM, hidden or [32, 16], "elu")
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    return net


def test_save_load_roundtrip(tmp_path):
    """存进去再读出来：结构、元信息、权重、输出全部一致。"""
    net = _make_net([32, 16])
    path = save_policy(
        tmp_path / "policy.pt",
        net,
        contract_hash=CONTRACT,
        joint_names=config.JOINT_NAMES,
        hidden=[32, 16],
        activation="elu",
        meta={"stage": "lift", "steps": 12345, "tag": "unit-test"},
    )
    assert path.exists()

    loaded, bundle = load_policy(path, expected_contract=CONTRACT)
    assert bundle["format"] == FORMAT
    assert bundle["obs_dim"] == config.OBS_DIM
    assert bundle["act_dim"] == config.ACTION_DIM
    assert tuple(bundle["joint_names"]) == tuple(config.JOINT_NAMES)
    assert bundle["contract_hash"] == CONTRACT
    assert bundle["net"] == {"hidden": [32, 16], "activation": "elu"}
    assert bundle["meta"]["stage"] == "lift"

    original_state = net.state_dict()
    loaded_state = loaded.state_dict()
    assert set(original_state) == set(loaded_state)
    for key, value in original_state.items():
        np.testing.assert_allclose(
            loaded_state[key].detach().cpu().numpy(),
            value.detach().cpu().numpy(),
            atol=1e-7,
        )

    obs = torch.randn(1, config.OBS_DIM)
    with torch.no_grad():
        np.testing.assert_allclose(
            loaded(obs).numpy(), net(obs).numpy(), atol=1e-6
        )


def test_actor_net_has_no_tanh_saturation(tmp_path):
    """最后一层不加 tanh：输出可以明显超过 [-1, 1]，由运行端负责裁剪。"""
    net = ActorNet(config.OBS_DIM, config.ACTION_DIM, [8], "elu")
    with torch.no_grad():
        net.mlp[-1].bias.fill_(5.0)
    out = net(torch.zeros(1, config.OBS_DIM)).detach().numpy()
    assert out.shape == (1, config.ACTION_DIM)
    assert float(out.max()) > 1.5


def test_contract_mismatch_raises(tmp_path):
    """模型契约改了 → PolicyContractError，消息里有可执行的提示。"""
    net = _make_net()
    path = save_policy(
        tmp_path / "policy.pt",
        net,
        contract_hash="hash-A",
        joint_names=config.JOINT_NAMES,
        hidden=[32, 16],
        activation="elu",
        meta={"stage": "stand"},
    )
    with pytest.raises(PolicyContractError) as err:
        load_policy(path, expected_contract="hash-B")
    message = str(err.value)
    assert "模型改过" in message
    assert "重新训练" in message


def test_joint_names_mismatch_raises(tmp_path):
    """关节顺序变了，即使不传 expected_contract 也要拒绝加载。"""
    net = _make_net()
    wrong_names = tuple(reversed(config.JOINT_NAMES))
    path = save_policy(
        tmp_path / "policy.pt",
        net,
        contract_hash=CONTRACT,
        joint_names=wrong_names,
        hidden=[32, 16],
        activation="elu",
    )
    with pytest.raises(PolicyContractError):
        load_policy(path)


def test_bad_format_rejected(tmp_path):
    """旧/异构 checkpoint 被格式位拒绝。"""
    path = tmp_path / "bad.pt"
    torch.save({"format": "someone.else.v9"}, path)
    with pytest.raises(PolicyContractError):
        load_policy(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy(tmp_path / "nope.pt")


def test_export_from_sb3_ducktyped(tmp_path):
    """不装 SB3 也能测导出：构造同结构的假 PPO，逐层核对权重。"""

    class FakePolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp_extractor = torch.nn.Module()
            self.mlp_extractor.policy_net = torch.nn.Sequential(
                torch.nn.Linear(config.OBS_DIM, 24),
                torch.nn.ELU(),
                torch.nn.Linear(24, 12),
                torch.nn.ELU(),
            )
            self.action_net = torch.nn.Linear(12, config.ACTION_DIM)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy = FakePolicy()

    model = FakeModel()
    path = export_from_sb3(
        model,
        tmp_path / "from_sb3.pt",
        contract_hash=CONTRACT,
        joint_names=config.JOINT_NAMES,
        meta={"stage": "walk"},
    )
    net, bundle = load_policy(path, expected_contract=CONTRACT)
    assert bundle["net"]["hidden"] == [24, 12]
    assert bundle["net"]["activation"] == "elu"
    assert bundle["meta"]["stage"] == "walk"
    assert bundle["meta"]["source"] == "stable_baselines3"

    obs = torch.randn(1, config.OBS_DIM)
    with torch.no_grad():
        expected = model.policy.action_net(model.policy.mlp_extractor.policy_net(obs))
        actual = net(obs)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-6)


def test_export_from_sb3_rejects_non_sb3_object(tmp_path):
    """传错对象 → TypeError（存在 SB3 时）/ ImportError（没装 SB3 时）。"""

    class NotAPolicy:
        pass

    with pytest.raises((TypeError, ImportError)):
        export_from_sb3(
            NotAPolicy(),
            tmp_path / "nope.pt",
            contract_hash=CONTRACT,
            joint_names=config.JOINT_NAMES,
        )

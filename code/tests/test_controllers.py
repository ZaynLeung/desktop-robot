"""控制器层测试：脚本控制器、策略控制器、registry 工厂。

策略控制器部分缺 torch 时单独 skip，脚本控制器/registry 的用例照常运行。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dino_wbc import config
from dino_wbc.controllers import (
    Controller,
    HoldStandController,
    LiftFootController,
    load_controller,
)


def _zero_obs() -> np.ndarray:
    return np.zeros(config.OBS_DIM, dtype=np.float32)


# ----------------------------------------------------------------------
# 接口 / 脚本控制器
# ----------------------------------------------------------------------
def test_controller_is_abstract():
    """Controller 是抽象类，不能直接实例化。"""
    with pytest.raises(TypeError):
        Controller()  # type: ignore[abstract]


def test_hold_stand_controller():
    controller = HoldStandController()
    assert isinstance(controller, Controller)
    assert controller.name == "scripted_stand"
    assert controller.stage == "stand"
    assert controller.obs_dim == config.OBS_DIM
    assert controller.act_dim == config.ACTION_DIM

    action = controller.act(_zero_obs())
    assert action.shape == (config.ACTION_DIM,)
    assert action.dtype == np.float32
    np.testing.assert_allclose(action, 0.0)

    controller.reset()
    controller.close()
    info = controller.describe()
    assert info["name"] == "scripted_stand"
    assert info["stage"] == "stand"


def test_hold_stand_rejects_wrong_obs():
    with pytest.raises(ValueError):
        HoldStandController().act(np.zeros(config.OBS_DIM - 1, dtype=np.float32))


def test_lift_foot_controller_shape_and_side():
    controller = LiftFootController(target_foot=-1.0)
    assert controller.stage == "lift"
    assert controller.side == "L"

    action = controller.act(_zero_obs())
    assert action.shape == (config.ACTION_DIM,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    # 只有一个目标脚在动，且只走 hip_pitch / knee / ankle 三个关节。
    assert set(config.JOINT_NAMES[index] for index in np.flatnonzero(action)) <= {
        "hip_pitch_L",
        "knee_L",
        "ankle_L",
    }

    controller.set_target_foot(1.0)
    assert controller.side == "R"
    action = controller.act(_zero_obs())
    assert set(config.JOINT_NAMES[index] for index in np.flatnonzero(action)) <= {
        "hip_pitch_R",
        "knee_R",
        "ankle_R",
    }


def test_lift_foot_controller_amplitude_within_scale():
    """跑满一个周期：动作始终在 [-1,1]，且屈髋为正、屈膝为负。"""
    controller = LiftFootController(target_foot=-1.0)
    hip = config.JOINT_NAMES.index("hip_pitch_L")
    knee = config.JOINT_NAMES.index("knee_L")
    ankle = config.JOINT_NAMES.index("ankle_L")

    actions = np.array(
        [controller.act(_zero_obs()) for _ in range(150)], dtype=np.float64
    )
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)
    assert actions[:, hip].max() > 0.0
    assert actions[:, knee].min() < -0.1
    assert actions[:, ankle].max() > 0.0
    # 波形幅度不超过构造时给的 amplitude。
    assert actions[:, hip].max() <= controller.amplitude + 1e-6
    assert abs(actions[:, knee].min()) <= controller.amplitude + 1e-6
    # 复位后相位归零（第一个动作是 0 = 站姿）。
    controller.reset()
    np.testing.assert_allclose(controller.act(_zero_obs()), 0.0, atol=1e-7)


def test_lift_foot_rejects_bad_arguments():
    with pytest.raises(ValueError):
        LiftFootController(target_foot=0.0)
    with pytest.raises(ValueError):
        LiftFootController(amplitude=1.5)


# ----------------------------------------------------------------------
# registry.json
# ----------------------------------------------------------------------
def test_registry_json_skeleton(code_dir):
    """初始 registry 必须覆盖四个阶段，且指向 code/checkpoints/<stage>。"""
    path = code_dir / "registry.json"
    assert path.exists(), "code/registry.json 缺失"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for stage in config.TASK_NAMES:
        assert stage in payload, f"registry.json 缺少阶段 {stage!r}"
        entry = payload[stage]
        assert entry["stage"] == stage
        assert entry["checkpoint"].endswith(f"{stage}/policy.pt")
        assert not str(entry["checkpoint"]).startswith(("/", "\\"))


def test_registry_scripted_names():
    assert load_controller("scripted_stand").stage == "stand"
    assert load_controller("scripted_lift").stage == "lift"


def test_registry_missing_checkpoint_mentions_training(tmp_path, monkeypatch):
    """checkpoint 不存在 → 报错信息里要带训练命令提示。"""
    import dino_wbc.controllers.registry as registry_module

    registry_file = tmp_path / "registry.json"
    registry_file.write_text(
        json.dumps(
            {"stand": {"stage": "stand", "checkpoint": "checkpoints/stand/policy.pt"}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(registry_module, "REGISTRY_PATH", registry_file)

    with pytest.raises(FileNotFoundError) as err:
        load_controller("stand")
    message = str(err.value)
    assert "checkpoint" in message
    assert "训练" in message
    assert "train.py" in message


def test_registry_unknown_name_lists_options():
    with pytest.raises(FileNotFoundError) as err:
        load_controller("definitely_not_a_controller")
    message = str(err.value)
    assert "scripted_stand" in message
    assert "stand" in message


def test_registry_missing_file_mentions_training(tmp_path, monkeypatch):
    import dino_wbc.controllers.registry as registry_module

    monkeypatch.setattr(registry_module, "REGISTRY_PATH", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError) as err:
        load_controller("walk")
    assert "训练" in str(err.value)


def test_registry_missing_pt_path(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        load_controller(str(tmp_path / "not_trained.pt"))
    assert "checkpoint" in str(err.value)


# ----------------------------------------------------------------------
# 策略控制器（需要 torch）
# ----------------------------------------------------------------------
def test_policy_controller_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from dino_wbc.controllers.policy import PolicyController
    from dino_wbc.policy_io import ActorNet, save_policy

    net = ActorNet(config.OBS_DIM, config.ACTION_DIM, [16], "elu")
    # 让输出远大于 1，验证运行端会裁剪到 [-1, 1]。
    with torch.no_grad():
        net.mlp[-1].bias.fill_(10.0)
    path = save_policy(
        tmp_path / "policy.pt",
        net,
        contract_hash="c0ffee",
        joint_names=config.JOINT_NAMES,
        hidden=[16],
        activation="elu",
        meta={"stage": "lift"},
    )

    controller = PolicyController(path, expected_contract="c0ffee")
    assert controller.stage == "lift"
    assert controller.name == "lift@policy"
    assert controller.obs_dim == config.OBS_DIM
    assert controller.act_dim == config.ACTION_DIM
    # 强制 CPU / eval 模式。
    assert all(parameter.device.type == "cpu" for parameter in controller.net.parameters())
    assert not controller.net.training

    action = controller.act(_zero_obs())
    assert action.shape == (config.ACTION_DIM,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    np.testing.assert_allclose(action, 1.0, atol=1e-5)

    info = controller.describe()
    assert info["policy"] == "actor-net"
    assert info["device"] == "cpu"
    controller.close()


def test_policy_controller_contract_mismatch(tmp_path):
    pytest.importorskip("torch")
    from dino_wbc.controllers.policy import PolicyController
    from dino_wbc.policy_io import ActorNet, PolicyContractError, save_policy

    net = ActorNet(config.OBS_DIM, config.ACTION_DIM, [8], "elu")
    path = save_policy(
        tmp_path / "policy.pt",
        net,
        contract_hash="hash-A",
        joint_names=config.JOINT_NAMES,
        hidden=[8],
        activation="elu",
        meta={"stage": "stand"},
    )
    with pytest.raises(PolicyContractError) as err:
        PolicyController(path, expected_contract="hash-B")
    assert "模型改过" in str(err.value)


def test_policy_controller_unknown_stage(tmp_path):
    pytest.importorskip("torch")
    from dino_wbc.controllers.policy import PolicyController
    from dino_wbc.policy_io import ActorNet, save_policy

    net = ActorNet(config.OBS_DIM, config.ACTION_DIM, [8], "elu")
    path = save_policy(
        tmp_path / "policy.pt",
        net,
        contract_hash="hash-A",
        joint_names=config.JOINT_NAMES,
        hidden=[8],
        activation="elu",
        meta={"stage": "dance"},
    )
    with pytest.raises(ValueError):
        PolicyController(path, expected_contract="hash-A")

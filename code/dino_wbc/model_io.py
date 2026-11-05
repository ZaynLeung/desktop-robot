"""MuJoCo 模型加载、关节布局与「模型契约」。

训练端和运行端都从这里拿模型，保证两边看到的关节顺序、限位、执行器完全一致。
``contract_hash`` 会写进 checkpoint，运行端加载时比对，模型改过就拒绝加载，
避免"静默地用错动作"。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from . import config

try:  # pragma: no cover - 缺包时给出可读错误
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dino_wbc 需要 mujoco，请先执行：\n"
        r"  E:\conda_envs\desktop-robot\python.exe -m pip install -r code/requirements-wbc.txt"
    ) from exc


#: ``code/`` 目录；仓库根是它的上一级。
CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "dino_scene.xml"

#: 足部碰撞件命名前缀；左右由后缀 ``_L`` / ``_R`` 区分。
_FOOT_GEOM_PREFIXES = ("foot_geom_", "toe_geom_")


class ModelContractError(RuntimeError):
    """模型里的关节/执行器与 :mod:`dino_wbc.config` 不一致。"""


@dataclass(frozen=True)
class JointLayout:
    """从 ``MjModel`` 解析出来的、控制代码需要的全部索引。"""

    names: tuple[str, ...]
    qposadr: np.ndarray  # (N,)
    dofadr: np.ndarray  # (N,)
    ranges: np.ndarray  # (N, 2)，站姿相对
    act_ids: np.ndarray  # (N,)，每个关节对应的 actuator 下标
    joint_ids: np.ndarray  # (N,)
    torso_body_id: int
    head_body_id: int
    foot_geom_ids: Mapping[str, tuple[int, ...]]
    foot_body_ids: Mapping[str, int]
    #: 地板 geom id（``dino_scene.xml`` 里为 ``floor``；单独加载 ``dino.xml`` 时为 -1）。
    floor_geom_id: int
    keyframe_name: str
    stand_qpos: np.ndarray  # (7 + N,)
    stand_base_z: float
    stand_up: np.ndarray  # (3,) 标称站姿下躯干局部 z 轴的世界方向

    @property
    def num_joints(self) -> int:
        return len(self.names)


def resolve_model_path(path: str | Path | None = None) -> Path:
    """把命令行/默认路径解析成绝对路径，并给出友好错误。"""
    target = Path(path) if path is not None else DEFAULT_MODEL
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if not target.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{target}\n"
            "先在仓库根目录运行：python -m desktop_robot build-dino"
        )
    return target


def _name2id(model, objtype, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id < 0:
        raise ModelContractError(f"模型里找不到 {name!r}")
    return obj_id


def build_layout(
    model,
    *,
    joint_names: Iterable[str] = config.JOINT_NAMES,
    stand_key: str = config.STAND_KEYFRAME,
) -> JointLayout:
    """解析关节/执行器/足部几何，并校验顺序与 :mod:`dino_wbc.config` 一致。"""
    names = tuple(joint_names)
    expected = tuple(config.JOINT_NAMES)
    if names != expected:
        raise ModelContractError(
            "关节顺序与 config.JOINT_NAMES 不一致：\n"
            f"  期望 {expected}\n  实际 {names}"
        )

    hinge_joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if tuple(hinge_joints) != expected:
        raise ModelContractError(
            "模型的铰链顺序与 config.JOINT_NAMES 不一致（models/*.xml 过期？）：\n"
            f"  期望 {expected}\n  实际 {tuple(hinge_joints)}\n"
            "重新生成：python -m desktop_robot build-dino"
        )

    qposadr = np.empty(len(names), dtype=np.int64)
    dofadr = np.empty(len(names), dtype=np.int64)
    ranges = np.empty((len(names), 2), dtype=np.float64)
    act_ids = np.full(len(names), -1, dtype=np.int64)
    joint_ids = np.empty(len(names), dtype=np.int64)

    for index, name in enumerate(names):
        jid = _name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_ids[index] = jid
        qposadr[index] = model.jnt_qposadr[jid]
        dofadr[index] = model.jnt_dofadr[jid]
        low, high = model.jnt_range[jid]
        ranges[index] = (low, high)

    # 执行器：优先按名字 act_<joint>，退化时按 actuator_trnid 反查。
    for index, name in enumerate(names):
        act_name = f"act_{name}"
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if aid < 0:
            candidates = np.flatnonzero(model.actuator_trnid[:, 0] == joint_ids[index])
            if len(candidates) != 1:
                raise ModelContractError(
                    f"关节 {name} 找不到唯一的执行器（期望名为 {act_name}）"
                )
            aid = int(candidates[0])
        act_ids[index] = aid
    if len(set(act_ids.tolist())) != len(names):
        raise ModelContractError("执行器到关节的映射不是一一对应")

    foot_geom_ids: dict[str, list[int]] = {"L": [], "R": []}
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not gname.startswith(_FOOT_GEOM_PREFIXES):
            continue
        if gname.startswith(("foot_geom_L", "toe_geom_L")):
            foot_geom_ids["L"].append(gid)
        elif gname.startswith(("foot_geom_R", "toe_geom_R")):
            foot_geom_ids["R"].append(gid)
    if not foot_geom_ids["L"] or not foot_geom_ids["R"]:
        raise ModelContractError("找不到足部碰撞几何（foot_geom_* / toe_geom_*）")

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, stand_key)
    if key_id < 0:
        raise ModelContractError(f"模型里找不到关键帧 {stand_key!r}")
    stand_qpos = np.array(model.key_qpos[key_id], dtype=np.float64)
    stand_mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(stand_mat, np.asarray(stand_qpos[3:7], dtype=np.float64))
    stand_up = stand_mat.reshape(3, 3)[:, 2].copy()

    return JointLayout(
        names=names,
        qposadr=qposadr,
        dofadr=dofadr,
        ranges=ranges,
        act_ids=act_ids,
        joint_ids=joint_ids,
        torso_body_id=_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso"),
        head_body_id=_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head"),
        foot_geom_ids={side: tuple(ids) for side, ids in foot_geom_ids.items()},
        foot_body_ids={
            side: _name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot_{side}")
            for side in ("L", "R")
        },
        floor_geom_id=int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        ),
        keyframe_name=stand_key,
        stand_qpos=stand_qpos,
        stand_base_z=float(stand_qpos[2]),
        stand_up=stand_up,
    )


def _apply_actuator_overrides(
    model,
    layout: JointLayout,
    *,
    kp: float | None,
    kv: float | None,
    forcerange: float | Mapping[str, float] | None,
) -> None:
    """在内存里覆盖位置执行器的 PD 增益/力矩上限（不改提交的 XML）。

    这是计划里写明的「调参杠杆」：默认 ``None`` 完全沿用模型值；做消融或
    发现 kp=25 偏软时，训练端和运行端传入**同一组**覆盖值即可。
    """
    if kp is None and kv is None and forcerange is None:
        return
    for index, name in enumerate(layout.names):
        aid = int(layout.act_ids[index])
        if kp is not None:
            model.actuator_gainprm[aid, 0] = kp
            model.actuator_biasprm[aid, 1] = -kp
        if kv is not None:
            model.actuator_biasprm[aid, 2] = -kv
        if forcerange is not None:
            limit = (
                float(forcerange[name])
                if isinstance(forcerange, Mapping)
                else float(forcerange)
            )
            model.actuator_forcerange[aid] = (-limit, limit)


def load_model(path: str | Path | None = None):
    """加载 ``MjModel``；``layout`` 单独用 :func:`build_layout` 取得。"""
    target = resolve_model_path(path)
    return mujoco.MjModel.from_xml_path(str(target))


def load(
    path: str | Path | None = None,
    *,
    kp: float | None = None,
    kv: float | None = None,
    forcerange: float | Mapping[str, float] | None = None,
) -> tuple[object, JointLayout]:
    """一次拿到 ``(model, layout)``，并校验契约。"""
    model = load_model(path)
    layout = build_layout(model)
    _apply_actuator_overrides(model, layout, kp=kp, kv=kv, forcerange=forcerange)
    return model, layout


def contract_hash(
    model,
    layout: JointLayout,
    *,
    obs_dim: int = config.OBS_DIM,
    action_scale: tuple[float, ...] = config.ACTION_SCALE,
) -> str:
    """模型 + 观测/动作布局的指纹，写进 checkpoint 用于拒绝不匹配的加载。"""
    hasher = hashlib.sha256()
    hasher.update(b"dino_wbc.contract.v1|")
    hasher.update("|".join(layout.names).encode())
    for low, high in layout.ranges:
        hasher.update(f"|{low:.6f},{high:.6f}".encode())
    hasher.update(f"|obs={obs_dim}".encode())
    scale_text = ",".join(f"{value:.4f}" for value in action_scale)
    hasher.update(("|scale=" + scale_text).encode())
    hasher.update(f"|nq={model.nq},nv={model.nv},nu={model.nu}".encode())
    return hasher.hexdigest()[:16]


def foot_geom_sides(layout: JointLayout) -> dict[int, str]:
    """geom id → "L"/"R"，用于接触检测。"""
    return {
        gid: side for side, ids in layout.foot_geom_ids.items() for gid in ids
    }

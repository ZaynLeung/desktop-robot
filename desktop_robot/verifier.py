"""生成物校验。

``verify`` 分两层：

1. **静态校验**（永远可跑，只需要标准库）：把生成出来的 XML 解析回来，
   逐项对照 :class:`~desktop_robot.config.RobotSpec`，能抓住生成器的绝大多数错误。
2. **动态校验**（可选，需要 MuJoCo）：真正 ``MjModel.from_xml_path`` 加载、
   比较 ``nq`` / ``nv`` / ``nu``，再按 ``stand`` 关键帧跑一段看基座高度是否稳定。

这样在没有 MuJoCo 的机器上（比如本项目的初始开发环境）也能给出有意义的
校验结果，装了 MuJoCo 之后再加 ``--dynamic`` 做真机级确认。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import xml.etree.ElementTree as ET

from .builder import MODELS_DIR, half
from .config import DEFAULT_SPEC, RobotSpec
from .kinematics import foot_sole_z, solve_pelvis_z


@dataclass
class Report:
    """校验结果。``errors`` 非空即视为失败。"""

    info: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _check_duplicates(report: Report, names: list[str], kind: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            report.errors.append(f"{kind} 名字重复: {name}")
        seen.add(name)


def verify_static(
    spec: RobotSpec = DEFAULT_SPEC, models_dir: Path | str = MODELS_DIR
) -> Report:
    """解析 ``models/`` 下的 XML 并对照参数源做静态校验。"""
    report = Report()
    models = Path(models_dir)
    robot_path = models / "biped.xml"
    scene_path = models / "scene.xml"

    for path in (robot_path, scene_path):
        if not path.exists():
            report.errors.append(f"缺少模型文件: {path}（先运行 python -m desktop_robot build）")
    if not report.ok:
        return report

    try:
        robot_root = ET.parse(robot_path).getroot()
        scene_root = ET.parse(scene_path).getroot()
    except ET.ParseError as exc:  # pragma: no cover - 生成器正确时不会触发
        report.errors.append(f"XML 解析失败: {exc}")
        return report

    # ---- 机器人模型 ----
    if robot_root.tag != "mujoco":
        report.errors.append(f"biped.xml 根元素应为 mujoco，实际为 {robot_root.tag}")
    if robot_root.get("model") != spec.name:
        report.errors.append(f"model 名不符: {robot_root.get('model')} != {spec.name}")

    compilers = robot_root.findall("compiler")
    if len(compilers) != 1:
        report.errors.append(f"compiler 元素应恰好 1 个，实际 {len(compilers)}")
    elif compilers[0].get("angle") != "radian":
        report.errors.append("compiler/angle 必须为 radian（本工程所有角度都是弧度）")

    for tag in ("worldbody", "actuator", "keyframe"):
        if robot_root.find(tag) is None:
            report.errors.append(f"biped.xml 缺少 <{tag}>")

    bodies = robot_root.findall(".//body")
    if len(bodies) != spec.n_bodies:
        report.errors.append(f"body 数量应为 {spec.n_bodies}，实际 {len(bodies)}")
    _check_duplicates(report, [b.get("name", "") for b in bodies], "body")

    # ---- 自由度 ----
    freejoints = robot_root.findall(".//freejoint")
    if len(freejoints) != 1 or freejoints[0].get("name") != "root":
        report.errors.append("应存在且仅存在一个名为 root 的 freejoint")

    hinges = robot_root.findall(".//joint")
    actual_joints = [j.get("name", "") for j in hinges]
    if actual_joints != spec.joint_names:
        report.errors.append(
            "关节名/顺序与参数源不一致:\n"
            f"  期望 {spec.joint_names}\n  实际 {actual_joints}"
        )
    _check_duplicates(report, actual_joints, "joint")
    if len(hinges) != len(spec.actuated()):
        report.errors.append(f"关节数量应为 {len(spec.actuated())}，实际 {len(hinges)}")

    for joint in hinges:
        name = joint.get("name", "?")
        rng = joint.get("range")
        if not rng:
            report.errors.append(f"关节 {name} 缺少 range")
            continue
        lo, hi = (float(v) for v in rng.split())
        if not lo < hi:
            report.errors.append(f"关节 {name} 的 range 非法: [{lo}, {hi}]")
        axis = [float(v) for v in (joint.get("axis") or "0 0 1").split()]
        if math.isclose(math.sqrt(sum(a * a for a in axis)), 0.0):
            report.errors.append(f"关节 {name} 的 axis 是零向量")

    # ---- 执行器 ----
    actuators = robot_root.find("actuator")
    act_elems = list(actuators) if actuators is not None else []
    if len(act_elems) != spec.nu:
        report.errors.append(f"执行器数量应为 {spec.nu}，实际 {len(act_elems)}")
    _check_duplicates(report, [a.get("name", "") for a in act_elems], "actuator")
    joint_names = set(actual_joints)
    for act in act_elems:
        target = act.get("joint")
        if target not in joint_names:
            report.errors.append(f"执行器 {act.get('name')} 指向不存在的关节 {target}")

    # ---- 关键帧 ----
    keys = robot_root.findall("keyframe/key")
    if len(keys) != 1 or keys[0].get("name") != "stand":
        report.errors.append("应存在且仅存在一个名为 stand 的关键帧")
    else:
        qpos = keys[0].get("qpos", "").split()
        ctrl = keys[0].get("ctrl", "").split()
        if len(qpos) != spec.nq:
            report.errors.append(f"stand 关键帧 qpos 长度应为 {spec.nq}，实际 {len(qpos)}")
        if len(ctrl) != spec.nu:
            report.errors.append(f"stand 关键帧 ctrl 长度应为 {spec.nu}，实际 {len(ctrl)}")
        if len(qpos) >= 3:
            z = float(qpos[2])
            expected = solve_pelvis_z(spec)
            if abs(z - expected) > 1e-4:
                report.errors.append(
                    f"stand 骨盆高度 {z:.6f} 与运动学解 {expected:.6f} 不符"
                )

    # ---- 姿态自洽性：站立时足底应贴地 ----
    hip_pitch, knee, ankle = spec.stand
    for side in ("L", "R"):
        sole = foot_sole_z(spec, side, [0.0, hip_pitch, knee, ankle])
        if abs(sole + solve_pelvis_z(spec)) > 1e-6:
            report.errors.append(f"{side} 侧足底在站立姿态下未与地面齐平")

    # ---- 场景模型 ----
    if scene_root.find("include") is None or scene_root.find("include").get("file") != "biped.xml":
        report.errors.append("scene.xml 应当 <include file=\"biped.xml\"/>")
    if scene_root.find("compiler") is not None:
        report.errors.append("scene.xml 不应自带 <compiler>（会和 include 的冲突）")
    floor = scene_root.find(".//geom[@name='floor']")
    if floor is None:
        report.errors.append("scene.xml 缺少 floor 地面")
    elif (floor.get("contype"), floor.get("conaffinity")) != ("1", "1"):
        report.errors.append("floor 的 contype/conaffinity 应为 1/1，否则脚踩不到地面")

    # ---- 汇总信息 ----
    report.info["文件"] = f"{robot_path.name}, {scene_path.name}"
    report.info["规模"] = f"bodies={len(bodies)}, joints={len(hinges)}, actuators={len(act_elems)}"
    report.info["期望规模"] = f"nq={spec.nq}, nv={spec.nv}, nu={spec.nu}"
    report.info["总质量"] = f"≈ {spec.total_mass():.3f} kg"
    report.info["站立骨盆高"] = f"{solve_pelvis_z(spec):.4f} m"

    # 足部是唯一参与接触的几何，这里顺带核对一下
    feet = [g for g in robot_root.findall(".//geom") if g.get("name", "").startswith("foot_geom_")]
    if len(feet) != 2:
        report.errors.append(f"应有 2 个足部 geom，实际 {len(feet)}")
    for geom in feet:
        if (geom.get("contype"), geom.get("conaffinity")) != ("1", "1"):
            report.errors.append(f"{geom.get('name')} 的 contype/conaffinity 应为 1/1")

    # 碰撞掩码：脚 1/1（撞地板和另一只脚），其余连杆 2/1（只撞地板）
    all_robot_geoms = robot_root.findall(".//geom")
    for geom in all_robot_geoms:
        name = geom.get("name", "")
        mask = (geom.get("contype"), geom.get("conaffinity"))
        if name.startswith("foot_geom_"):
            if mask != ("1", "1"):
                report.errors.append(f"{name} 的碰撞掩码应为 1/1，实际为 {mask}")
        elif mask != ("2", "1"):
            report.errors.append(
                f"{name} 的碰撞掩码应为 2/1（只和地板碰撞、连杆间不自碰），实际为 {mask}"
            )

    # 交叉核对 box 半尺寸与参数源一致
    pelvis_geom = robot_root.find(".//geom[@name='pelvis_geom']")
    if pelvis_geom is not None:
        expected = [f for f in half(spec.pelvis_size)]
        actual = [float(v) for v in pelvis_geom.get("size", "").split()]
        if any(abs(a - b) > 1e-9 for a, b in zip(actual, expected)):
            report.errors.append(f"pelvis 半尺寸不一致: {actual} != {expected}")

    return report

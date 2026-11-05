"""把 :class:`~desktop_robot.config.RobotSpec` 渲染成 MuJoCo 的 MJCF 文件。

设计要点
--------
* 用 ``xml.etree.ElementTree`` 构造再序列化，天然保证 XML 合法。
* 输出是**确定性的**：同样的 ``RobotSpec`` 一定生成字节一致的 XML，
  这样 ``models/*.xml`` 可以干净地进入 git，重跑 build 不会产生噪音 diff。
* 不使用 ``<default>`` 继承：所有属性显式写在元素上。少一层隐式继承，
  模型文件可以直接当文档读，也不会出现"场景里的地板意外继承机器人默认值"。
* ``models/scene.xml`` 用 MJCF 标准的 ``<include>`` 引用机器人模型，
  机器人本体保持独立可加载。
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from .config import DEFAULT_SPEC, JointSpec, RobotSpec, Vec3, joint_name
from .kinematics import solve_pelvis_z, stand_qpos

#: 默认的模型输出目录（仓库根下的 ``models/``）。
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

#: 浮点输出的固定小数位。固定位宽 + 去掉尾随零 → 输出稳定且可读。
DECIMALS = 6

_HEADER = """<?xml version="1.0" encoding="utf-8"?>
<!-- 由 desktop_robot.builder 自动生成，请勿手工修改。       -->
<!-- 参数源：desktop_robot/config.py 的 DEFAULT_SPEC。      -->
<!-- 重新生成：python -m desktop_robot build                -->
"""


def fnum(value: float) -> str:
    """把浮点数格式化成稳定、紧凑的字符串。"""
    v = float(value)
    if abs(v) < 1e-12:
        v = 0.0
    text = f"{v:.{DECIMALS}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0", "-") else text


def vec(values) -> str:
    """把一组数格式化成 MJCF 的空格分隔字符串。"""
    return " ".join(fnum(v) for v in values)


def half(size: Vec3) -> Vec3:
    """全尺寸 → box 的半尺寸。"""
    return (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)


def _geom_attributes(
    name: str,
    size: Vec3,
    pos: Vec3,
    spec: RobotSpec,
    rgba: str,
    contype: int,
    conaffinity: int,
    friction: str = "0.9 0.02 0.001",
) -> dict[str, str]:
    """构造一个 box geom 的属性表。

    碰撞用位掩码区分三类几何（MuJoCo 规则：两个 geom 碰撞当且仅当
    ``contype1 & conaffinity2`` 或 ``contype2 & conaffinity1`` 非零）：

    ===========  =======  ===========  ====================================
    类别         contype  conaffinity  与之碰撞
    ===========  =======  ===========  ====================================
    地板          1        1           脚、躯干
    脚            1        1           地板、另一只脚
    其他连杆      2        1           只和地板
    ===========  =======  ===========  ====================================

    于是：脚踩得住地、摔倒时躯干不会穿模，但连杆之间不会自碰撞
    （简单模型的常规做法，也避免步态训练时接触点爆炸）。
    """
    return {
        "name": name,
        "type": "box",
        "size": vec(half(size)),
        "pos": vec(pos),
        "density": fnum(spec.density),
        "rgba": rgba,
        "friction": friction,
        "contype": str(contype),
        "conaffinity": str(conaffinity),
    }


def _joint_attributes(joint: JointSpec, side: str | None, spec: RobotSpec) -> dict[str, str]:
    """构造一个受限旋转关节的属性表。"""
    return {
        "name": joint_name(joint, side),
        "type": "hinge",
        "axis": vec(joint.axis_for(side)),
        "range": vec(joint.range),
        "limited": "true",
        "armature": fnum(joint.armature),
        "damping": fnum(joint.damping),
        "frictionloss": fnum(joint.frictionloss),
        "pos": vec((0.0, 0.0, 0.0)),
    }


def _actuator_attributes(joint: JointSpec, side: str | None) -> dict[str, str]:
    """构造一个位置执行器的属性表（``inheritrange`` 让 ctrl 范围跟随关节范围）。"""
    name = joint_name(joint, side)
    return {
        "name": f"act_{name}",
        "joint": name,
        "kp": fnum(joint.kp),
        "kv": fnum(joint.kv),
        "forcerange": f"{-joint.forcerange:g} {joint.forcerange:g}",
        "forcelimited": "true",
        "inheritrange": "1",
        "ctrllimited": "true",
    }


def build_robot(spec: RobotSpec = DEFAULT_SPEC) -> ET.Element:
    """构造机器人本体的 MJCF 根元素（不含地面/灯光）。"""
    root = ET.Element("mujoco", {"model": spec.name})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})

    body = ET.SubElement(root, "worldbody")

    # ---- 骨盆：浮动基座 ----
    pelvis = ET.SubElement(body, "body", {"name": "pelvis", "pos": "0 0 0"})
    ET.SubElement(pelvis, "freejoint", {"name": "root"})
    ET.SubElement(
        pelvis,
        "geom",
        _geom_attributes(
            "pelvis_geom",
            spec.pelvis_size,
            (0.0, 0.0, 0.0),
            spec,
            "0.55 0.58 0.64 1",
            contype=2,
            conaffinity=1,
        ),
    )

    # ---- 腰部 / 躯干：骨盆上方，绕 z 轴偏航 ----
    torso = ET.SubElement(
        pelvis, "body", {"name": "torso", "pos": vec((0.0, 0.0, spec.waist_z))}
    )
    ET.SubElement(torso, "joint", _joint_attributes(spec.waist_joint, None, spec))
    ET.SubElement(
        torso,
        "geom",
        _geom_attributes(
            "torso_geom",
            spec.torso_size,
            (0.0, 0.0, spec.torso_center_z),
            spec,
            "0.35 0.62 0.78 1",
            contype=2,
            conaffinity=1,
        ),
    )

    # ---- 双腿：L 在 +y，R 在 -y ----
    for side in ("L", "R"):
        sign = 1.0 if side == "L" else -1.0

        thigh = ET.SubElement(
            pelvis,
            "body",
            {"name": f"thigh_{side}", "pos": vec((0.0, sign * spec.hip_y, spec.hip_z))},
        )
        ET.SubElement(thigh, "joint", _joint_attributes(spec.hip_roll, side, spec))
        ET.SubElement(thigh, "joint", _joint_attributes(spec.hip_pitch, side, spec))
        ET.SubElement(
            thigh,
            "geom",
            _geom_attributes(
                f"thigh_geom_{side}",
                spec.thigh_size,
                (0.0, 0.0, -spec.thigh_len / 2.0),
                spec,
                "0.72 0.75 0.80 1",
                contype=2,
                conaffinity=1,
            ),
        )

        shank = ET.SubElement(
            thigh,
            "body",
            {"name": f"shank_{side}", "pos": vec((0.0, 0.0, -spec.thigh_len))},
        )
        ET.SubElement(shank, "joint", _joint_attributes(spec.knee, side, spec))
        ET.SubElement(
            shank,
            "geom",
            _geom_attributes(
                f"shank_geom_{side}",
                spec.shank_size,
                (0.0, 0.0, -spec.shank_len / 2.0),
                spec,
                "0.72 0.75 0.80 1",
                contype=2,
                conaffinity=1,
            ),
        )

        foot = ET.SubElement(
            shank,
            "body",
            {"name": f"foot_{side}", "pos": vec((0.0, 0.0, -spec.shank_len))},
        )
        ET.SubElement(foot, "joint", _joint_attributes(spec.ankle, side, spec))
        ET.SubElement(
            foot,
            "geom",
            _geom_attributes(
                f"foot_geom_{side}",
                spec.foot_size,
                spec.foot_center,
                spec,
                "0.20 0.24 0.30 1",
                contype=1,
                conaffinity=1,
                friction="1.2 0.02 0.001",
            ),
        )

    # ---- 执行器：九个位置源 ----
    actuator = ET.SubElement(root, "actuator")
    for joint, side in spec.actuated():
        ET.SubElement(actuator, "position", _actuator_attributes(joint, side))

    # ---- 关键帧：标称站立姿态 ----
    keyframe = ET.SubElement(root, "keyframe")
    stand = spec.stand_q()
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "stand",
            "qpos": vec(stand_qpos(spec)),
            "ctrl": vec([stand[name] for name in spec.joint_names]),
        },
    )
    return root


def build_scene(spec: RobotSpec = DEFAULT_SPEC) -> ET.Element:
    """构造双足机器人的场景：本体通过 ``<include file="biped.xml"/>`` 引入。"""
    return build_scene_for("biped.xml", f"{spec.name}_scene")


def build_scene_for(
    include_file: str,
    model_name: str,
    *,
    statistic_center: str = "0 0 0.16",
    statistic_extent: str = "0.4",
) -> ET.Element:
    """构造通用场景：``include_file`` 引入机器人本体，外面套地面/灯光/相机。

    双足机型和恐龙机型共用这一段；差别只在被 include 的文件名与
    ``<statistic>`` 的取景范围（恐龙带尾巴，模型在 x 方向更长）。
    """
    root = ET.Element("mujoco", {"model": model_name})
    ET.SubElement(root, "include", {"file": include_file})
    # 求解器设置放在场景里：机器人本体保持一个"纯模型"，可以单独加载。
    ET.SubElement(
        root,
        "option",
        {
            "timestep": "0.002",
            "integrator": "implicitfast",
            "cone": "elliptic",
            "impratio": "10",
            "gravity": "0 0 -9.81",
        },
    )
    ET.SubElement(root, "statistic", {"center": statistic_center, "extent": statistic_extent})

    # 纯内建纹理（builtin），不引入任何外部图片文件，仓库仍然没有二进制资产
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "sky",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.30 0.40 0.55",
            "rgb2": "0.05 0.06 0.09",
            "width": "256",
            "height": "256",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "grid",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.24 0.26 0.30",
            "rgb2": "0.30 0.33 0.38",
            "mark": "edge",
            "markrgb": "0.36 0.39 0.44",
            "width": "256",
            "height": "256",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "grid", "texture": "grid", "texrepeat": "16 16", "reflectance": "0.03"},
    )

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        {
            "azimuth": "135",
            "elevation": "-20",
            # 离屏渲染缓冲：默认 640x480 太小，录帧/做 RL 观测都不够用
            "offwidth": "1280",
            "offheight": "960",
        },
    )
    ET.SubElement(visual, "quality", {"shadowsize": "8192"})
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.7 0.7 0.7", "ambient": "0.25 0.25 0.25", "specular": "0.1 0.1 0.1"},
    )

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        {
            "name": "key_light",
            "pos": "0.8 -0.8 1.5",
            "dir": "-0.4 0.5 -0.75",
            "directional": "true",
            "diffuse": "0.8 0.8 0.8",
            "specular": "0.2 0.2 0.2",
        },
    )
    # 补光不投影：两盏方向光同时投影会让地板出现明显的阴影摩尔纹
    ET.SubElement(
        world,
        "light",
        {
            "name": "fill_light",
            "pos": "-0.8 0.6 1.0",
            "dir": "0.5 -0.4 -0.6",
            "directional": "true",
            "diffuse": "0.25 0.28 0.32",
            "specular": "0 0 0",
            "castshadow": "false",
        },
    )
    ET.SubElement(
        world,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "2 2 0.1",
            "pos": "0 0 0",
            "material": "grid",
            "rgba": "0.30 0.33 0.38 1",
            "friction": "1.0 0.02 0.001",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    # 两个固定机位：正视图用于报告出图，45° 视图用于看整体姿态
    ET.SubElement(
        world,
        "camera",
        {"name": "front", "pos": "0.60 0 0.30", "xyaxes": "0 1 0 -0.2425 0 0.9701"},
    )
    ET.SubElement(
        world,
        "camera",
        {"name": "iso", "pos": "0.45 -0.45 0.30", "xyaxes": "0.7069 0.7069 0 -0.1621 0.1621 0.9734"},
    )
    return root


def to_text(root: ET.Element) -> str:
    """把元素树序列化成带表头注释的稳定文本。"""
    ET.indent(root, space="  ")
    return _HEADER + ET.tostring(root, encoding="unicode") + "\n"


def render_robot_xml(spec: RobotSpec = DEFAULT_SPEC) -> str:
    """渲染机器人模型文本。"""
    return to_text(build_robot(spec))


def render_scene_xml(spec: RobotSpec = DEFAULT_SPEC) -> str:
    """渲染场景模型文本。"""
    return to_text(build_scene(spec))


def write_models(
    spec: RobotSpec = DEFAULT_SPEC, out_dir: Path | str = MODELS_DIR
) -> list[Path]:
    """写出 ``biped.xml`` 与 ``scene.xml``，返回写出的路径。

    内容与现有文件相同的时候不重写，避免无意义的 mtime 变化。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for filename, text in (
        ("biped.xml", render_robot_xml(spec)),
        ("scene.xml", render_scene_xml(spec)),
    ):
        path = out / filename
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)
    return written


def summary(spec: RobotSpec = DEFAULT_SPEC) -> str:
    """给 CLI 用的一行摘要。"""
    height = solve_pelvis_z(spec) + spec.waist_z + spec.torso_center_z + spec.torso_size[2] / 2
    return (
        f"{spec.name}: {spec.n_bodies} bodies, nq={spec.nq}, nv={spec.nv}, nu={spec.nu}, "
        f"mass≈{spec.total_mass():.3f} kg, height≈{height:.3f} m"
    )

"""从 ``desktop_robot.dino`` 的参数源直接导出 CAD 占位装配。

产物：
* ``desktop_dino_cad.step``：整包 STEP（含彩色关节辅助轴 ref_*，用于查看/量尺寸）
* ``assembly.step``：**扁平装配体**，一根连杆一个 component、位置就是站姿世界位姿，
  不含任何 ref_* 辅助轴（SolidWorks 打开→保存全部，即可得到原生 .SLDASM + .SLDPRT）
* ``parts/<link>.step``：**每根连杆一个单件**（多实体），局部坐标、原点在各自关节，
  用于在 SolidWorks 里单独编辑后替换
* ``desktop_dino_parts.csv``：零件/几何清单，便于核对尺寸
* ``desktop_dino_assembly.json``：装配树、关节轴与相对位姿

说明：
* 源模型参数单位是米；CAD/STEP 默认导出成毫米，便于机械设计。
* STEP 对「真正的坐标系/基准轴」支持并不统一，``desktop_dino_cad.step`` 里用可见的
  辅助轴几何代替；``assembly.step`` / ``parts/`` 这两种“可编辑”产物则刻意不带辅助轴，
  避免 SolidWorks 把 90 个 ref_* 也变成零件。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import cadquery as cq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from desktop_robot.dino import DEFAULT_DINO_SPEC, build_dino


DEFAULT_OUT_DIR = REPO_ROOT / "build" / "cad" / "dino"
MM_PER_M = 1000.0


@dataclass(frozen=True)
class GeomRecord:
    body_name: str
    geom_name: str
    geom_type: str
    size_m: tuple[float, ...]
    local_pos_m: tuple[float, float, float]
    local_quat_wxyz: tuple[float, float, float, float]
    rgba: tuple[float, float, float, float]
    density: float


@dataclass(frozen=True)
class JointRecord:
    body_name: str
    joint_name: str
    joint_type: str
    local_pos_m: tuple[float, float, float]
    axis_local: tuple[float, float, float]
    range_rad: tuple[float, float] | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export desktop dino to CAD STEP assembly.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for STEP/CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--step-name",
        default="desktop_dino_cad.step",
        help="STEP assembly filename.",
    )
    parser.add_argument(
        "--parts-name",
        default="desktop_dino_parts.csv",
        help="Parts CSV filename.",
    )
    parser.add_argument(
        "--assembly-name",
        default="desktop_dino_assembly.json",
        help="Assembly JSON filename.",
    )
    parser.add_argument(
        "--joint-axis-length-mm",
        type=float,
        default=20.0,
        help="Length of helper axes created at joint anchors.",
    )
    parser.add_argument(
        "--joint-axis-radius-mm",
        type=float,
        default=0.8,
        help="Radius of helper axis cylinders.",
    )
    parser.add_argument(
        "--joint-axis-color",
        default="0.95,0.78,0.12,1.0",
        help="RGBA color for joint-axis highlight cylinders.",
    )
    return parser.parse_args()


def _parse_vec(text: str | None, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    return tuple(float(part) for part in text.split())


def _parse_rgba(text: str | None) -> tuple[float, float, float, float]:
    rgba = _parse_vec(text, default=(0.7, 0.7, 0.7, 1.0))
    if len(rgba) == 3:
        return (rgba[0], rgba[1], rgba[2], 1.0)
    return (rgba[0], rgba[1], rgba[2], rgba[3])


def _vec_mm(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(value * MM_PER_M for value in values)


def _normalize(axis: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(value / norm for value in axis)


def _quat_to_axis_angle_deg(
    quat_wxyz: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], float]:
    w, x, y, z = quat_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return (0.0, 0.0, 1.0), 0.0
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s <= 1e-9 or abs(angle) <= 1e-9:
        return (0.0, 0.0, 1.0), 0.0
    axis = (x / s, y / s, z / s)
    return _normalize(axis), math.degrees(angle)


def _loc_from_pos_quat(
    pos_m: tuple[float, float, float], quat_wxyz: tuple[float, float, float, float]
) -> cq.Location:
    pos_mm = _vec_mm(pos_m)
    axis, angle_deg = _quat_to_axis_angle_deg(quat_wxyz)
    if abs(angle_deg) <= 1e-9:
        return cq.Location(pos_mm)
    return cq.Location(pos_mm, axis, angle_deg)


def _cad_color(rgba: tuple[float, float, float, float]) -> cq.Color:
    return cq.Color(*rgba)


def _make_box(size_m: tuple[float, float, float]) -> cq.Shape:
    x_mm, y_mm, z_mm = _vec_mm(size_m)
    return cq.Solid.makeBox(x_mm, y_mm, z_mm, cq.Vector(-x_mm / 2.0, -y_mm / 2.0, -z_mm / 2.0))


def _make_cylinder(radius_m: float, half_length_m: float) -> cq.Shape:
    radius_mm = radius_m * MM_PER_M
    length_mm = 2.0 * half_length_m * MM_PER_M
    return cq.Solid.makeCylinder(
        radius_mm,
        length_mm,
        pnt=cq.Vector(0.0, 0.0, -length_mm / 2.0),
        dir=cq.Vector(0.0, 0.0, 1.0),
    )


def _make_capsule(radius_m: float, half_length_m: float) -> cq.Shape:
    radius_mm = radius_m * MM_PER_M
    half_length_mm = half_length_m * MM_PER_M
    cylinder = cq.Solid.makeCylinder(
        radius_mm,
        2.0 * half_length_mm,
        pnt=cq.Vector(0.0, 0.0, -half_length_mm),
        dir=cq.Vector(0.0, 0.0, 1.0),
    )
    sphere_a = cq.Solid.makeSphere(radius_mm, cq.Vector(0.0, 0.0, half_length_mm))
    sphere_b = cq.Solid.makeSphere(radius_mm, cq.Vector(0.0, 0.0, -half_length_mm))
    return cylinder.fuse(sphere_a).fuse(sphere_b)


def _make_sphere(radius_m: float) -> cq.Shape:
    return cq.Solid.makeSphere(radius_m * MM_PER_M)


def _make_geom_shape(geom_type: str, size_m: tuple[float, ...]) -> cq.Shape:
    if geom_type == "box":
        return _make_box((size_m[0], size_m[1], size_m[2]))
    if geom_type == "cylinder":
        return _make_cylinder(size_m[0], size_m[1])
    if geom_type == "capsule":
        return _make_capsule(size_m[0], size_m[1])
    if geom_type == "sphere":
        return _make_sphere(size_m[0])
    raise ValueError(f"暂不支持的 geom type: {geom_type}")


def _make_axis_cylinder(
    axis_local: tuple[float, float, float],
    *,
    length_mm: float,
    radius_mm: float,
    centered: bool,
) -> cq.Shape:
    axis = cq.Vector(*_normalize(axis_local))
    start = axis.multiply(-length_mm / 2.0) if centered else cq.Vector(0.0, 0.0, 0.0)
    return cq.Solid.makeCylinder(radius_mm, length_mm, pnt=start, dir=axis)


def _range_from_joint(joint_elem: ET.Element) -> tuple[float, float] | None:
    text = joint_elem.get("range")
    if not text:
        return None
    values = tuple(float(part) for part in text.split())
    return (values[0], values[1])


def _extract_geom_record(body_name: str, geom_elem: ET.Element) -> GeomRecord:
    geom_type = geom_elem.get("type", "box")
    size = _parse_vec(geom_elem.get("size"), default=(0.0, 0.0, 0.0))
    if geom_type == "box":
        size_m = (2.0 * size[0], 2.0 * size[1], 2.0 * size[2])
    elif geom_type in {"cylinder", "capsule"}:
        size_m = (size[0], size[1])
    elif geom_type == "sphere":
        size_m = (size[0],)
    else:
        raise ValueError(f"暂不支持的 geom type: {geom_type}")
    return GeomRecord(
        body_name=body_name,
        geom_name=geom_elem.get("name", "unnamed_geom"),
        geom_type=geom_type,
        size_m=size_m,
        local_pos_m=_parse_vec(geom_elem.get("pos"), default=(0.0, 0.0, 0.0)),
        local_quat_wxyz=_parse_vec(geom_elem.get("quat"), default=(1.0, 0.0, 0.0, 0.0)),
        rgba=_parse_rgba(geom_elem.get("rgba")),
        density=float(geom_elem.get("density", "0")),
    )


def _extract_joint_record(body_name: str, joint_elem: ET.Element) -> JointRecord:
    return JointRecord(
        body_name=body_name,
        joint_name=joint_elem.get("name", "unnamed_joint"),
        joint_type=joint_elem.get("type", "hinge"),
        local_pos_m=_parse_vec(joint_elem.get("pos"), default=(0.0, 0.0, 0.0)),
        axis_local=_parse_vec(joint_elem.get("axis"), default=(0.0, 0.0, 1.0)),
        range_rad=_range_from_joint(joint_elem),
    )


def _geom_record_to_csv_row(record: GeomRecord) -> dict[str, Any]:
    return {
        "body_name": record.body_name,
        "geom_name": record.geom_name,
        "geom_type": record.geom_type,
        "size_m": json.dumps(record.size_m),
        "size_mm": json.dumps(_vec_mm(record.size_m)),
        "local_pos_m": json.dumps(record.local_pos_m),
        "local_pos_mm": json.dumps(_vec_mm(record.local_pos_m)),
        "local_quat_wxyz": json.dumps(record.local_quat_wxyz),
        "rgba": json.dumps(record.rgba),
        "density": record.density,
        "role": "visual_only" if abs(record.density) < 1e-9 else "solid",
    }


def _joint_record_to_json(record: JointRecord) -> dict[str, Any]:
    return {
        "name": record.joint_name,
        "type": record.joint_type,
        "local_pos_m": list(record.local_pos_m),
        "local_pos_mm": list(_vec_mm(record.local_pos_m)),
        "axis_local": list(record.axis_local),
        "range_rad": list(record.range_rad) if record.range_rad else None,
    }


def _color_from_csv_text(text: str) -> tuple[float, float, float, float]:
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 4:
        raise ValueError("joint-axis-color 应为四个逗号分隔的 RGBA 值")
    return (values[0], values[1], values[2], values[3])


def _build_body_assembly(
    body_elem: ET.Element,
    *,
    joint_axis_length_mm: float,
    joint_axis_radius_mm: float,
    joint_axis_rgba: tuple[float, float, float, float],
    geom_rows: list[GeomRecord],
) -> tuple[cq.Assembly, dict[str, Any]]:
    body_name = body_elem.get("name", "unnamed_body")
    body_assy = cq.Assembly(name=body_name)
    body_json: dict[str, Any] = {
        "name": body_name,
        "local_pos_m": list(_parse_vec(body_elem.get("pos"), default=(0.0, 0.0, 0.0))),
        "local_pos_mm": list(_vec_mm(_parse_vec(body_elem.get("pos"), default=(0.0, 0.0, 0.0)))),
        "local_quat_wxyz": list(_parse_vec(body_elem.get("quat"), default=(1.0, 0.0, 0.0, 0.0))),
        "geoms": [],
        "joints": [],
        "children": [],
    }

    for geom_elem in body_elem.findall("geom"):
        record = _extract_geom_record(body_name, geom_elem)
        geom_rows.append(record)
        shape = _make_geom_shape(record.geom_type, record.size_m)
        body_assy.add(
            shape,
            name=record.geom_name,
            loc=_loc_from_pos_quat(record.local_pos_m, record.local_quat_wxyz),
            color=_cad_color(record.rgba),
        )
        body_json["geoms"].append(
            {
                "name": record.geom_name,
                "type": record.geom_type,
                "size_m": list(record.size_m),
                "size_mm": list(_vec_mm(record.size_m)),
                "local_pos_m": list(record.local_pos_m),
                "local_pos_mm": list(_vec_mm(record.local_pos_m)),
                "local_quat_wxyz": list(record.local_quat_wxyz),
                "rgba": list(record.rgba),
                "density": record.density,
            }
        )

    axis_colors = {
        "x": (0.88, 0.18, 0.18, 1.0),
        "y": (0.18, 0.72, 0.28, 1.0),
        "z": (0.16, 0.33, 0.88, 1.0),
    }
    for joint_elem in body_elem.findall("joint"):
        joint = _extract_joint_record(body_name, joint_elem)
        body_json["joints"].append(_joint_record_to_json(joint))
        joint_loc = cq.Location(_vec_mm(joint.local_pos_m))
        triad_axes = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }
        for axis_name, axis_local in triad_axes.items():
            body_assy.add(
                _make_axis_cylinder(
                    axis_local,
                    length_mm=joint_axis_length_mm,
                    radius_mm=joint_axis_radius_mm,
                    centered=False,
                ),
                name=f"ref_{joint.joint_name}_{axis_name}",
                loc=joint_loc,
                color=_cad_color(axis_colors[axis_name]),
            )
        body_assy.add(
            _make_axis_cylinder(
                joint.axis_local,
                length_mm=joint_axis_length_mm * 1.1,
                radius_mm=joint_axis_radius_mm * 1.35,
                centered=True,
            ),
            name=f"ref_{joint.joint_name}_axis",
            loc=joint_loc,
            color=_cad_color(joint_axis_rgba),
        )
        body_assy.add(
            cq.Solid.makeSphere(joint_axis_radius_mm * 1.8),
            name=f"ref_{joint.joint_name}_anchor",
            loc=joint_loc,
            color=_cad_color((0.92, 0.92, 0.92, 1.0)),
        )

    for child_body in body_elem.findall("body"):
        child_assy, child_json = _build_body_assembly(
            child_body,
            joint_axis_length_mm=joint_axis_length_mm,
            joint_axis_radius_mm=joint_axis_radius_mm,
            joint_axis_rgba=joint_axis_rgba,
            geom_rows=geom_rows,
        )
        child_pos = _parse_vec(child_body.get("pos"), default=(0.0, 0.0, 0.0))
        child_quat = _parse_vec(child_body.get("quat"), default=(1.0, 0.0, 0.0, 0.0))
        body_assy.add(
            child_assy,
            name=child_json["name"],
            loc=_loc_from_pos_quat(child_pos, child_quat),
        )
        body_json["children"].append(child_json)

    return body_assy, body_json


def _body_compound(
    body_elem: ET.Element,
) -> tuple[cq.Compound, tuple[float, float, float, float]]:
    """把一个 body 的所有 geom 合成「单件」（body 局部坐标），供 ``parts/<link>.step``。

    返回 ``(复合体, 主色)``；主色取第一个有质量的 geom（电机包络是零密度，会被跳过）。
    """
    body_name = body_elem.get("name", "unnamed_body")
    shapes: list[cq.Shape] = []
    color: tuple[float, float, float, float] = (0.30, 0.55, 0.25, 1.0)
    for geom_elem in body_elem.findall("geom"):
        record = _extract_geom_record(body_name, geom_elem)
        shape = _make_geom_shape(record.geom_type, record.size_m)
        shapes.append(
            shape.moved(_loc_from_pos_quat(record.local_pos_m, record.local_quat_wxyz))
        )
        if record.density > 0.0:
            color = record.rgba
    if not shapes:
        return cq.Compound.makeCompound([]), color
    return cq.Compound.makeCompound(shapes), color


def _collect_world_bodies(
    body_elem: ET.Element,
    parent_loc: cq.Location,
    out: list[tuple[ET.Element, cq.Location]],
) -> None:
    """深度优先收集 ``(body, 世界位置)``：把每级的 pos/quat 累乘到父坐标系上。"""
    local = _loc_from_pos_quat(
        _parse_vec(body_elem.get("pos"), default=(0.0, 0.0, 0.0)),
        _parse_vec(body_elem.get("quat"), default=(1.0, 0.0, 0.0, 0.0)),
    )
    world = parent_loc * local
    out.append((body_elem, world))
    for child in body_elem.findall("body"):
        _collect_world_bodies(child, world, out)


def _write_split_outputs(
    root_body: ET.Element, out_dir: Path, spec_name: str
) -> tuple[Path, Path, int]:
    """写「一杆一个零件」的 ``parts/<link>.step`` 和已装配就位的 ``assembly.step``。

    每个 body 一个复合实体（不再拆成 geom 级），装配体是**扁平**的：一个连杆一个
    component、位置就是整机站姿下的世界位姿——SolidWorks 打开后零件就在正确位置，
    保存一次即得原生 .SLDASM + 可编辑 .SLDPRT。不带任何 ``ref_*`` 辅助轴。
    """
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    assembly = cq.Assembly(name=f"{spec_name}_assembly")
    bodies: list[tuple[ET.Element, cq.Location]] = []
    _collect_world_bodies(root_body, cq.Location(), bodies)
    for body_elem, world in bodies:
        name = body_elem.get("name", "unnamed_body")
        compound, color = _body_compound(body_elem)
        cq.exporters.export(compound, str(parts_dir / f"{name}.step"))
        assembly.add(compound, name=name, loc=world, color=_cad_color(color))
    assembly_path = out_dir / "assembly.step"
    assembly.save(str(assembly_path), exportType="STEP", mode="default")
    return assembly_path, parts_dir, len(bodies)


def _write_parts_csv(path: Path, records: list[GeomRecord]) -> None:
    rows = [_geom_record_to_csv_row(record) for record in records]
    fieldnames = [
        "body_name",
        "geom_name",
        "geom_type",
        "size_m",
        "size_mm",
        "local_pos_m",
        "local_pos_mm",
        "local_quat_wxyz",
        "rgba",
        "density",
        "role",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_assembly_json(path: Path, spec_name: str, root_body: dict[str, Any]) -> None:
    payload = {
        "model": spec_name,
        "units": {
            "source": "m",
            "cad_export": "mm",
            "mm_per_m": MM_PER_M,
        },
        "notes": {
            "helper_axes": "关节锚点处的彩色圆柱用于在 SolidWorks 里充当可见基准轴。",
            "source": "直接基于 desktop_robot.dino.build_dino(DEFAULT_DINO_SPEC) 提取。",
        },
        "root_body": root_body,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")


def main() -> int:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_axis_rgba = _color_from_csv_text(args.joint_axis_color)

    root = build_dino(DEFAULT_DINO_SPEC)
    torso = root.find("worldbody/body")
    if torso is None:
        raise SystemExit("找不到 torso body，无法导出 CAD 装配。")

    geom_rows: list[GeomRecord] = []
    torso_assy, torso_json = _build_body_assembly(
        torso,
        joint_axis_length_mm=args.joint_axis_length_mm,
        joint_axis_radius_mm=args.joint_axis_radius_mm,
        joint_axis_rgba=joint_axis_rgba,
        geom_rows=geom_rows,
    )

    root_assy = cq.Assembly(name=f"{DEFAULT_DINO_SPEC.name}_cad")
    torso_pos = _parse_vec(torso.get("pos"), default=(0.0, 0.0, 0.0))
    torso_quat = _parse_vec(torso.get("quat"), default=(1.0, 0.0, 0.0, 0.0))
    root_assy.add(
        torso_assy,
        name=torso_json["name"],
        loc=_loc_from_pos_quat(torso_pos, torso_quat),
    )

    step_path = out_dir / args.step_name
    parts_csv_path = out_dir / args.parts_name
    assembly_json_path = out_dir / args.assembly_name

    root_assy.save(str(step_path), exportType="STEP", mode="default")
    _write_parts_csv(parts_csv_path, geom_rows)
    _write_assembly_json(assembly_json_path, DEFAULT_DINO_SPEC.name, torso_json)

    assembly_path, parts_dir, body_count = _write_split_outputs(
        torso, out_dir, DEFAULT_DINO_SPEC.name
    )

    print(f"STEP : {step_path}")
    print(f"PARTS: {parts_csv_path}")
    print(f"TREE : {assembly_json_path}")
    print(f"ASM  : {assembly_path}  (扁平装配，{body_count} 个连杆组件)")
    print(f"LINKS: {parts_dir}\\*.step  ({body_count} 个单件，原点=各自关节)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

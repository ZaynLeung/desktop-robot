"""导出可用于正/逆运动学的「连杆-关节表」：每行一个关节，带 4×4 齐次变换矩阵。

产物（默认 ``build/kinematics/``）：
* ``dino_kinematics.xlsx``：README / joints / links / frames / check 五个 sheet
* ``dino_kinematics_joints.csv``：与 ``joints`` sheet 同构，给脚本/MATLAB 直接读
* ``dino_kinematics.json``：同样的数据 + 链式结构，给 Python 用

约定（同时写进 README sheet）：
* 一行一个关节，``T_child(q) = T_parent · T0 · Rot(axis, q) · T1``；
* ``q`` 就是 MuJoCo 的 qpos（``q=0`` 对应标称站姿，物理角 = ``q + q_stand``）；
* 矩阵是列向量约定 ``T = [R p; 0 1]``，长度单位 m、角度 rad；
* 同一 body 上有多个关节时（颈 pitch/yaw）插入虚拟连杆，保证每行只有一个自由度。

用法::

    python .agents/misc/py/export_dino_kinematics.py
    python .agents/misc/py/export_dino_kinematics.py --out-dir build/kinematics --samples 8
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from desktop_robot.builder import MODELS_DIR  # noqa: E402
from desktop_robot.dino import (  # noqa: E402
    DEFAULT_DINO_SPEC,
    DinoSpec,
    build_dino,
    solve_base_z,
)

DEFAULT_OUT_DIR = REPO_ROOT / "build" / "kinematics"

MATRIX_COLUMNS = [f"r{row}{col}" for row in range(1, 5) for col in range(1, 5)]


# ----------------------------------------------------------------------
# 小工具：4×4 变换
# ----------------------------------------------------------------------
def _rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("axis 不能是零向量")
    x, y, z = x / norm, y / norm, z / norm
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def _translation(pos: tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = pos
    return T


def _quat_matrix(quat: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _vec(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    return tuple(float(part) for part in text.split())


def _body_transform(body: ET.Element) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _quat_matrix(_vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0)))
    T[:3, 3] = _vec(body.get("pos"), (0.0, 0.0, 0.0))
    return T


# ----------------------------------------------------------------------
# 表结构
# ----------------------------------------------------------------------
@dataclass
class JointRow:
    joint: str
    type: str
    parent: str
    child: str
    axis: tuple[float, float, float] | None
    t0: np.ndarray
    t1: np.ndarray
    q_min: float | None = None
    q_max: float | None = None
    q_stand: float = 0.0
    actuator: str = ""
    kp: float = 0.0
    forcerange: float | None = None
    body: str = ""

    def fk(self, q: float) -> np.ndarray:
        T = self.t0
        if self.axis is not None:
            R = np.eye(4)
            R[:3, :3] = _rotation(self.axis, q)
            T = T @ R
        return T @ self.t1


@dataclass
class Table:
    rows: list[JointRow] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)


def _build_table(spec: DinoSpec) -> Table:
    root = build_dino(spec)
    torso = root.find("worldbody/body")
    if torso is None:
        raise SystemExit("找不到 torso body")
    table = Table()

    actuators = {
        (elem.get("joint") or ""): elem for elem in root.findall("actuator/position")
    }

    def add_link(
        name: str,
        parent: str,
        pos: tuple[float, ...] | None,
        quat: tuple[float, ...] | None,
        joints: list[str],
        kind: str,
    ) -> None:
        table.links.append(
            {
                "link": name,
                "kind": kind,
                "parent": parent,
                "pos": pos,
                "quat": quat,
                "joints": joints,
            }
        )

    def walk(body_elem: ET.Element, parent_link: str, is_root: bool = False) -> None:
        name = body_elem.get("name", "unnamed")
        T_ref = _body_transform(body_elem)
        joints = body_elem.findall("joint")
        body_parent = parent_link
        prev_link = parent_link
        for index, joint_elem in enumerate(joints):
            joint_name = joint_elem.get("name", "")
            last = index == len(joints) - 1
            child = name if last else f"{name}__{joint_name}"
            anchor = _vec(joint_elem.get("pos"), (0.0, 0.0, 0.0))
            axis = _vec(joint_elem.get("axis"), (0.0, 0.0, 1.0))
            T_anchor = _translation(anchor)
            t0 = (T_ref @ T_anchor) if index == 0 else T_anchor
            act = actuators.get(joint_name)
            q_min = q_max = None
            rng = joint_elem.get("range")
            if rng:
                lo, hi = (float(v) for v in rng.split())
                q_min, q_max = lo, hi
            table.rows.append(
                JointRow(
                    joint=joint_name,
                    type=joint_elem.get("type", "hinge"),
                    parent=prev_link,
                    child=child,
                    axis=axis,
                    t0=t0,
                    t1=np.linalg.inv(T_anchor),
                    q_min=q_min,
                    q_max=q_max,
                    q_stand=spec.stand_q().get(joint_name, 0.0),
                    actuator=act.get("name", "") if act is not None else "",
                    kp=float(act.get("kp", "0")) if act is not None else 0.0,
                    forcerange=float(act.get("forcerange", "0").split()[1])
                    if act is not None
                    else None,
                    body=name,
                )
            )
            if not last:
                # 同一 body 上的第 1..n-1 个关节：插一个虚拟连杆，位姿由关节行给出
                add_link(child, body_parent, None, None, [joint_name], "virtual")
                body_parent = child
            prev_link = child
        add_link(
            name,
            body_parent,
            _vec(body_elem.get("pos"), (0.0, 0.0, 0.0)),
            _vec(body_elem.get("quat"), (1.0, 0.0, 0.0, 0.0)),
            [j.get("name", "") for j in joints],
            "body",
        )
        if not joints and not is_root:
            table.rows.append(
                JointRow(
                    joint="",
                    type="fixed",
                    parent=parent_link,
                    child=name,
                    axis=None,
                    t0=T_ref,
                    t1=np.eye(4),
                    body=name,
                )
            )
        for child_body in body_elem.findall("body"):
            walk(child_body, name)

    walk(torso, "world", is_root=True)

    # 基座：world → torso（站姿）
    base_row = JointRow(
        joint="root",
        type="free",
        parent="world",
        child="torso",
        axis=None,
        t0=_translation((0.0, 0.0, solve_base_z(spec)))
        @ np.block(
            [
                [_rotation((0.0, -1.0, 0.0), spec.base_pitch), np.zeros((3, 1))],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        ),
        t1=np.eye(4),
    )
    table.rows.insert(0, base_row)

    # 末端 frame：足底中心、头部相机
    for side in ("L", "R"):
        half_h = spec.foot_size[2] / 2.0
        table.frames.append(
            {
                "name": f"foot_sole_{side}",
                "parent": f"foot_{side}",
                "T": _translation(
                    (spec.foot_center[0], 0.0, spec.foot_center[2] - half_h)
                ),
                "note": "足底接触中心（脚掌底面），z 向朝上",
            }
        )
    head = root.find(".//body[@name='head']")
    if head is not None:
        camera = head.find("camera")
        if camera is not None:
            xy = _vec(camera.get("xyaxes"), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
            x = np.asarray(xy[0:3], dtype=float)
            y = np.asarray(xy[3:6], dtype=float)
            T = np.eye(4)
            T[:3, 0] = x
            T[:3, 1] = y
            T[:3, 2] = np.cross(x, y)
            T[:3, 3] = _vec(camera.get("pos"), (0.0, 0.0, 0.0))
            table.frames.append(
                {
                    "name": "cam_head",
                    "parent": "head",
                    "T": T,
                    "note": "头部相机；沿自身 −z 看",
                }
            )
    return table


# ----------------------------------------------------------------------
# 正运动学（用表算）
# ----------------------------------------------------------------------
def _fk(table: Table, q: dict[str, float]) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {"world": np.eye(4)}
    for row in table.rows:
        if row.type == "free":
            poses[row.child] = poses[row.parent] @ row.t0
        elif row.type == "fixed":
            poses[row.child] = poses[row.parent] @ row.t0
        else:
            poses[row.child] = poses[row.parent] @ row.fk(q.get(row.joint, 0.0))
    return poses


# ----------------------------------------------------------------------
# 交叉验证（表算 FK vs MuJoCo）
# ----------------------------------------------------------------------
def _validate(table: Table, spec: DinoSpec, samples: int, seed: int = 0) -> list[dict]:
    import mujoco  # noqa: PLC0415

    xml = ET.tostring(build_dino(spec), encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    joint_names = [row.joint for row in table.rows if row.type == "hinge"]
    limits = {
        row.joint: (row.q_min or 0.0, row.q_max or 0.0)
        for row in table.rows
        if row.type == "hinge"
    }
    results: list[dict] = []
    for sample in range(samples + 1):
        if sample == 0:
            q = {name: 0.0 for name in joint_names}
        else:
            q = {
                name: float(rng.uniform(lo, hi)) if hi > lo else 0.0
                for name, (lo, hi) in limits.items()
            }
        data.qpos[:] = model.qpos0
        for name, value in q.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = value
        mujoco.mj_forward(model, data)

        poses = _fk(table, q)
        T_torso_table = np.linalg.inv(poses["torso"])
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        T_torso = np.eye(4)
        T_torso[:3, :3] = data.xmat[torso_id].reshape(3, 3)
        T_torso[:3, 3] = data.xpos[torso_id]
        worst = 0.0
        for body_id in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name not in poses:
                continue
            T_mj = np.eye(4)
            T_mj[:3, :3] = data.xmat[body_id].reshape(3, 3)
            T_mj[:3, 3] = data.xpos[body_id]
            T_mj = np.linalg.inv(T_torso) @ T_mj
            T_table = T_torso_table @ poses[name]
            worst = max(worst, float(np.abs(T_table - T_mj).max()))
        results.append({"sample": sample, "max_abs_error": worst})
    return results


# ----------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------
def _write_csv(path: Path, table: Table) -> None:
    fieldnames = [
        "joint",
        "type",
        "parent",
        "child",
        "axis_x",
        "axis_y",
        "axis_z",
        "q_min",
        "q_max",
        "q_stand",
        "q_min_deg",
        "q_max_deg",
        "q_stand_deg",
        "actuator",
        "kp",
        "forcerange",
    ]
    fieldnames += [f"T0_{name}" for name in MATRIX_COLUMNS]
    fieldnames += [f"T1_{name}" for name in MATRIX_COLUMNS]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in table.rows:
            record: dict[str, object] = {
                "joint": row.joint,
                "type": row.type,
                "parent": row.parent,
                "child": row.child,
                "axis_x": row.axis[0] if row.axis else "",
                "axis_y": row.axis[1] if row.axis else "",
                "axis_z": row.axis[2] if row.axis else "",
                "q_min": row.q_min if row.q_min is not None else "",
                "q_max": row.q_max if row.q_max is not None else "",
                "q_stand": row.q_stand,
                "q_min_deg": math.degrees(row.q_min) if row.q_min is not None else "",
                "q_max_deg": math.degrees(row.q_max) if row.q_max is not None else "",
                "q_stand_deg": math.degrees(row.q_stand),
                "actuator": row.actuator,
                "kp": row.kp,
                "forcerange": row.forcerange if row.forcerange is not None else "",
            }
            for index, key in enumerate(MATRIX_COLUMNS):
                record[f"T0_{key}"] = round(float(row.t0.ravel()[index]), 12)
                record[f"T1_{key}"] = round(float(row.t1.ravel()[index]), 12)
            writer.writerow(record)


def _write_json(path: Path, table: Table, check: list[dict]) -> None:
    payload = {
        "convention": {
            "fk": "T_child(q) = T_parent @ T0 @ Rot(axis, q) @ T1",
            "q": "MuJoCo qpos；q=0 即标称站姿，物理角 = q + q_stand",
            "matrix": "T = [R p; 0 1]，列向量，单位 m / rad",
        },
        "rows": [
            {
                "joint": row.joint,
                "type": row.type,
                "parent": row.parent,
                "child": row.child,
                "axis": list(row.axis) if row.axis else None,
                "T0": row.t0.tolist(),
                "T1": row.t1.tolist(),
                "q_min": row.q_min,
                "q_max": row.q_max,
                "q_stand": row.q_stand,
                "actuator": row.actuator,
            }
            for row in table.rows
        ],
        "frames": [
            {"name": frame["name"], "parent": frame["parent"], "T": frame["T"].tolist()}
            for frame in table.frames
        ],
        "links": table.links,
        "check": check,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sheet_data(
    table: Table, check: list[dict], spec: DinoSpec
) -> list[dict[str, object]]:
    """把表整理成「sheet 名 + 行」，XLSX 的两种写法共用同一份数据。"""
    header = [
        "joint",
        "type",
        "parent",
        "child",
        "axis_x",
        "axis_y",
        "axis_z",
        "q_min",
        "q_max",
        "q_stand",
        "q_min_deg",
        "q_max_deg",
        "q_stand_deg",
        "actuator",
        "kp",
        "forcerange",
    ]
    header += [f"T0_{name}" for name in MATRIX_COLUMNS]
    header += [f"T1_{name}" for name in MATRIX_COLUMNS]

    joint_rows: list[list[object]] = [header]
    for row in table.rows:
        values: list[object] = [
            row.joint,
            row.type,
            row.parent,
            row.child,
            round(row.axis[0], 9) if row.axis else None,
            round(row.axis[1], 9) if row.axis else None,
            round(row.axis[2], 9) if row.axis else None,
            round(row.q_min, 9) if row.q_min is not None else None,
            round(row.q_max, 9) if row.q_max is not None else None,
            round(row.q_stand, 9),
            round(math.degrees(row.q_min), 4) if row.q_min is not None else None,
            round(math.degrees(row.q_max), 4) if row.q_max is not None else None,
            round(math.degrees(row.q_stand), 4),
            row.actuator,
            round(row.kp, 6),
            row.forcerange if row.forcerange is not None else None,
        ]
        values += [round(float(v), 12) for v in row.t0.ravel()]
        values += [round(float(v), 12) for v in row.t1.ravel()]
        joint_rows.append(values)

    link_rows: list[list[object]] = [
        ["link", "kind", "parent", "pos_x", "pos_y", "pos_z", "joints"]
    ]
    for link in table.links:
        pos = link["pos"]
        link_rows.append(
            [
                link["link"],
                link["kind"],
                link["parent"],
                round(pos[0], 6) if pos else None,
                round(pos[1], 6) if pos else None,
                round(pos[2], 6) if pos else None,
                ", ".join(link["joints"]),
            ]
        )

    frame_rows: list[list[object]] = [
        ["frame", "parent", "note"] + [f"T_{name}" for name in MATRIX_COLUMNS]
    ]
    for frame in table.frames:
        frame_rows.append(
            [frame["name"], frame["parent"], frame["note"]]
            + [round(float(v), 12) for v in frame["T"].ravel()]
        )

    check_rows: list[list[object]] = [["sample", "max_abs_error", "ok"]]
    for item in check:
        check_rows.append(
            [
                item["sample"],
                item["max_abs_error"],
                "PASS" if item["max_abs_error"] < 1e-9 else "FAIL",
            ]
        )

    readme_rows: list[list[object]] = [
        ["恐龙机器人 · 运动学表", ""],
        ["生成脚本", ".agents/misc/py/export_dino_kinematics.py"],
        ["源模型", "desktop_robot/dino.py → models/dino.xml"],
        ["", ""],
        ["一行的含义", "一个关节（或一个固定连接）"],
        ["FK 公式", "T_child(q) = T_parent · T0 · Rot(axis, q) · T1"],
        ["矩阵约定", "T = [R p; 0 1]，列向量；单位 m、rad"],
        ["q 的含义", "MuJoCo qpos；q=0 就是标称站姿；物理角 = q + q_stand"],
        ["T0/T1 的来历", "T0 = 父连杆→关节锚点（含 body 的 pos/quat），T1 = 锚点反向平移"],
        [
            "多关节 body",
            "颈 pitch/yaw 同属 head body → 插一个虚拟连杆 head__neck_pitch（links 里 kind=virtual），"
            "它的位姿完全由 neck_pitch 那行决定，不是零件",
        ],
        ["基座", "root 行是 world→torso 的站姿位姿（6 自由度浮动基座）"],
        ["足底 frame", "frames sheet 给出 foot_sole_L/R，位于脚掌底面中心"],
        ["校验", "check sheet 是「表算 FK」与 MuJoCo 实际位姿的逐连杆最大误差"],
        ["", ""],
        ["关节数", str(sum(1 for row in table.rows if row.type == "hinge"))],
        ["连杆数", str(len(table.links))],
        ["基座高", f"{solve_base_z(spec):.6f} m"],
    ]

    return [
        {"name": "README", "rows": readme_rows},
        {"name": "joints", "rows": joint_rows},
        {"name": "links", "rows": link_rows},
        {"name": "frames", "rows": frame_rows},
        {"name": "check", "rows": check_rows},
    ]


def _write_xlsx_openpyxl(path: Path, sheets: list[dict[str, object]]) -> None:
    from openpyxl import Workbook  # noqa: PLC0415
    from openpyxl.styles import Alignment, Font, PatternFill  # noqa: PLC0415

    wb = Workbook()
    first = True
    for sheet in sheets:
        ws = wb.active if first else wb.create_sheet()
        first = False
        ws.title = str(sheet["name"])
        for line in sheet["rows"]:  # type: ignore[union-attr]
            ws.append(list(line))
        if ws.title != "README":
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="305496")
                cell.alignment = Alignment(horizontal="center")
            ws.freeze_panes = "A2"
        for column in range(1, ws.max_column + 1):
            width = 10
            for row in ws.iter_rows(min_col=column, max_col=column):
                value = row[0].value
                if value is not None:
                    width = max(width, min(len(str(value)) + 2, 40))
            ws.column_dimensions[ws.cell(row=1, column=column).column_letter].width = width
    wb.save(path)


def _column_name(index: int) -> str:
    """0 → A，25 → Z，26 → AA。"""
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _write_xlsx_minimal(path: Path, sheets: list[dict[str, object]]) -> None:
    """没有 openpyxl 时的兜底：直接按 OOXML 写一个最小可用工作簿。"""
    from xml.sax.saxutils import escape  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    def sheet_xml(rows: list[list[object]]) -> str:
        widths: list[int] = []
        for row in rows:
            for index, value in enumerate(row):
                if index >= len(widths):
                    widths.append(10)
                text = "" if value is None else str(value)
                widths[index] = max(widths[index], min(len(text) + 2, 40))
        parts = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
            '<sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            "</sheetView></sheetViews>",
            "<cols>",
        ]
        for index, width in enumerate(widths, start=1):
            parts.append(f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>')
        parts.append("</cols><sheetData>")
        for row_index, row in enumerate(rows, start=1):
            cells = []
            for column_index, value in enumerate(row, start=1):
                if value is None or value == "":
                    continue
                reference = f"{_column_name(column_index - 1)}{row_index}"
                style = ' s="1"' if row_index == 1 else ""
                if isinstance(value, bool):
                    text = "1" if value else "0"
                    cells.append(f'<c r="{reference}" t="b"{style}><v>{text}</v></c>')
                elif isinstance(value, (int, float)):
                    cells.append(f'<c r="{reference}"{style}><v>{value!r}</v></c>')
                else:
                    cells.append(
                        f'<c r="{reference}" t="inlineStr"{style}><is><t xml:space="preserve">'
                        f"{escape(str(value))}</t></is></c>"
                    )
            parts.append(f'<row r="{row_index}">' + "".join(cells) + "</row>")
        parts.append("</sheetData></worksheet>")
        return "".join(parts)

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for index in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = []
    relationships = []
    for index, sheet in enumerate(sheets, start=1):
        workbook_sheets.append(
            f'<sheet name="{escape(str(sheet["name"]))}" sheetId="{index}" r:id="rId{index}"/>'
        )
        relationships.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    styles_id = len(sheets) + 1
    relationships.append(
        f'<Relationship Id="rId{styles_id}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )

    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )

    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>" + "".join(workbook_sheets) + "</sheets></workbook>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>',
        )
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(relationships)
            + "</Relationships>",
        )
        archive.writestr("xl/styles.xml", styles)
        for index, sheet in enumerate(sheets, start=1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml", sheet_xml(sheet["rows"])  # type: ignore[arg-type]
            )


def _write_xlsx(path: Path, table: Table, check: list[dict], spec: DinoSpec) -> str:
    sheets = _sheet_data(table, check, spec)
    try:
        _write_xlsx_openpyxl(path, sheets)
        return "openpyxl"
    except ModuleNotFoundError:
        _write_xlsx_minimal(path, sheets)
        return "builtin"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", type=int, default=5, help="随机校验样本数（含 1 组站姿）")
    args = parser.parse_args()

    spec = DEFAULT_DINO_SPEC
    table = _build_table(spec)
    check = _validate(table, spec, samples=args.samples)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "dino_kinematics.xlsx"
    csv_path = out_dir / "dino_kinematics_joints.csv"
    json_path = out_dir / "dino_kinematics.json"

    _write_csv(csv_path, table)
    _write_json(json_path, table, check)
    writer = _write_xlsx(xlsx_path, table, check, spec)

    worst = max(item["max_abs_error"] for item in check)
    print(f"XLSX : {xlsx_path}  (writer={writer})")
    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")
    print(
        f"关节 {sum(1 for row in table.rows if row.type == 'hinge')} 个、"
        f"连杆 {len(table.links)} 个；表算 FK vs MuJoCo 最大误差 {worst:.2e}"
    )
    if worst > 1e-9:
        print("[警告] 校验误差偏大，检查 T0/T1 或关节顺序", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

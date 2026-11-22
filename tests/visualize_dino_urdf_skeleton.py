"""用 URDF 可视化小机器龙的骨架。

默认读取 ``build/urdf/dino/desktop_dino_torso.urdf``，绘制零位姿下的关节骨架。
这是一个便于手动查看的脚本，不参与 pytest 自动测试。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URDF = ROOT / "build" / "urdf" / "dino" / "desktop_dino_torso.urdf"


@dataclass(frozen=True)
class JointInfo:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def _parse_vec3(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    return np.asarray([float(part) for part in text.split()], dtype=float)


def _rot_x(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rot_y(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rot_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rpy_to_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy.tolist()
    # URDF 的 origin rpy 等价于 Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_rotation(rpy)
    transform[:3, 3] = xyz
    return transform


def load_urdf_joints(urdf_path: Path) -> tuple[str, dict[str, list[JointInfo]]]:
    tree = ET.parse(urdf_path)
    robot = tree.getroot()
    if robot.tag != "robot":
        raise ValueError(f"{urdf_path} 不是合法 URDF robot 根节点")

    child_links: set[str] = set()
    joints_by_parent: dict[str, list[JointInfo]] = {}
    links: set[str] = set()

    for link_elem in robot.findall("link"):
        name = link_elem.get("name")
        if name:
            links.add(name)

    for joint_elem in robot.findall("joint"):
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue
        parent_link = parent_elem.get("link")
        child_link = child_elem.get("link")
        if not parent_link or not child_link:
            continue

        origin_elem = joint_elem.find("origin")
        xyz = _parse_vec3(origin_elem.get("xyz") if origin_elem is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_vec3(origin_elem.get("rpy") if origin_elem is not None else None, (0.0, 0.0, 0.0))
        axis_elem = joint_elem.find("axis")
        axis = _parse_vec3(axis_elem.get("xyz") if axis_elem is not None else None, (0.0, 0.0, 1.0))

        joint = JointInfo(
            name=joint_elem.get("name", "unnamed_joint"),
            joint_type=joint_elem.get("type", "fixed"),
            parent_link=parent_link,
            child_link=child_link,
            xyz=xyz,
            rpy=rpy,
            axis=axis,
        )
        joints_by_parent.setdefault(parent_link, []).append(joint)
        child_links.add(child_link)

    root_candidates = sorted(link for link in links if link not in child_links)
    if not root_candidates:
        raise ValueError(f"{urdf_path} 里没有找到根 link")
    return root_candidates[0], joints_by_parent


def forward_kinematics_zero(
    root_link: str, joints_by_parent: dict[str, list[JointInfo]]
) -> tuple[dict[str, np.ndarray], list[tuple[np.ndarray, np.ndarray, JointInfo]]]:
    link_frames: dict[str, np.ndarray] = {root_link: np.eye(4, dtype=float)}
    edges: list[tuple[np.ndarray, np.ndarray, JointInfo]] = []

    def walk(parent_link: str) -> None:
        parent_tf = link_frames[parent_link]
        parent_pos = parent_tf[:3, 3].copy()
        for joint in joints_by_parent.get(parent_link, []):
            joint_tf = parent_tf @ make_transform(joint.xyz, joint.rpy)
            link_frames[joint.child_link] = joint_tf
            child_pos = joint_tf[:3, 3].copy()
            edges.append((parent_pos, child_pos, joint))
            walk(joint.child_link)

    walk(root_link)
    return link_frames, edges


def _all_points(link_frames: dict[str, np.ndarray], edges: Iterable[tuple[np.ndarray, np.ndarray, JointInfo]]) -> np.ndarray:
    points = [frame[:3, 3] for frame in link_frames.values()]
    for parent_pos, child_pos, _joint in edges:
        points.extend([parent_pos, child_pos])
    return np.asarray(points, dtype=float)


def plot_skeleton(
    link_frames: dict[str, np.ndarray],
    edges: list[tuple[np.ndarray, np.ndarray, JointInfo]],
    *,
    show_axes: bool,
    axis_length: float,
    annotate: bool,
    title: str,
    save_path: Path | None,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    points = _all_points(link_frames, edges)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(np.maximum(maxs - mins, 1e-3)) * 1.2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))

    for parent_pos, child_pos, joint in edges:
        line = np.stack([parent_pos, child_pos], axis=0)
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color="royalblue", linewidth=2.0, marker="o")
        if annotate:
            ax.text(
                child_pos[0],
                child_pos[1],
                child_pos[2],
                joint.name,
                fontsize=8,
                color="black",
            )

        if show_axes and joint.joint_type in {"revolute", "continuous", "prismatic"}:
            axis_local = joint.axis.astype(float)
            norm = np.linalg.norm(axis_local)
            if norm > 1e-9:
                axis_world = link_frames[joint.child_link][:3, :3] @ (axis_local / norm)
                anchor = child_pos
                tip = anchor + axis_world * axis_length
                ax.plot(
                    [anchor[0], tip[0]],
                    [anchor[1], tip[1]],
                    [anchor[2], tip[2]],
                    color="darkorange",
                    linewidth=2.0,
                )

    root_link = next(iter(link_frames))
    root_pos = link_frames[root_link][:3, 3]
    ax.scatter(root_pos[0], root_pos[1], root_pos[2], color="crimson", s=70, label=f"root: {root_link}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize desktop dino skeleton from a URDF file.")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="URDF file to load. Default: build/urdf/dino/desktop_dino_torso.urdf",
    )
    parser.add_argument(
        "--no-axes",
        action="store_true",
        help="Hide joint axis direction lines.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.04,
        help="Length of the displayed joint-axis line in meters.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide joint name labels.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save the rendered skeleton figure to an image file instead of opening a window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    urdf_path = args.urdf.resolve()
    if not urdf_path.exists():
        raise SystemExit(f"URDF 不存在：{urdf_path}")

    root_link, joints_by_parent = load_urdf_joints(urdf_path)
    link_frames, edges = forward_kinematics_zero(root_link, joints_by_parent)
    plot_skeleton(
        link_frames,
        edges,
        show_axes=not args.no_axes,
        axis_length=args.axis_length,
        annotate=not args.no_labels,
        title=f"Desktop Dino Skeleton ({urdf_path.name})",
        save_path=args.save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

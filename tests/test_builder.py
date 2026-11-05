"""生成器测试：XML 结构、确定性、可重复生成。"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from desktop_robot.builder import (
    build_robot,
    build_scene,
    fnum,
    render_robot_xml,
    render_scene_xml,
    summary,
    vec,
    write_models,
)
from desktop_robot.config import DEFAULT_SPEC
from desktop_robot.kinematics import solve_pelvis_z


@pytest.fixture(scope="module")
def robot_root() -> ET.Element:
    return build_robot(DEFAULT_SPEC)


def test_root_and_compiler(robot_root):
    assert robot_root.tag == "mujoco"
    assert robot_root.get("model") == DEFAULT_SPEC.name
    compilers = robot_root.findall("compiler")
    assert len(compilers) == 1
    assert compilers[0].get("angle") == "radian"


def test_body_tree_shape(robot_root):
    bodies = robot_root.findall(".//body")
    assert [b.get("name") for b in bodies] == [
        "pelvis",
        "torso",
        "thigh_L",
        "shank_L",
        "foot_L",
        "thigh_R",
        "shank_R",
        "foot_R",
    ]


def test_single_free_joint_on_pelvis(robot_root):
    freejoints = robot_root.findall(".//freejoint")
    assert len(freejoints) == 1
    assert freejoints[0].get("name") == "root"
    pelvis = robot_root.find(".//body[@name='pelvis']")
    assert pelvis is not None and pelvis.find("freejoint") is not None


def test_joint_names_and_order(robot_root):
    names = [j.get("name") for j in robot_root.findall(".//joint")]
    assert names == DEFAULT_SPEC.joint_names


def test_hip_body_holds_two_joints_in_order(robot_root):
    thigh = robot_root.find(".//body[@name='thigh_L']")
    assert thigh is not None
    assert [j.get("name") for j in thigh.findall("joint")] == ["hip_roll_L", "hip_pitch_L"]


def test_actuators_target_every_joint_once(robot_root):
    joints = {j.get("name") for j in robot_root.findall(".//joint")}
    actuators = robot_root.findall("./actuator/position")
    assert len(actuators) == DEFAULT_SPEC.nu
    assert {a.get("joint") for a in actuators} == joints
    for actuator in actuators:
        assert actuator.get("inheritrange") == "1"
        assert actuator.get("forcelimited") == "true"


def test_keyframe_matches_spec(robot_root):
    keys = robot_root.findall("./keyframe/key")
    assert len(keys) == 1
    key = keys[0]
    assert key.get("name") == "stand"

    qpos = [float(v) for v in key.get("qpos").split()]
    ctrl = [float(v) for v in key.get("ctrl").split()]
    assert len(qpos) == DEFAULT_SPEC.nq
    assert len(ctrl) == DEFAULT_SPEC.nu
    assert qpos[2] == pytest.approx(solve_pelvis_z(DEFAULT_SPEC), abs=1e-6)
    assert ctrl == [DEFAULT_SPEC.stand_q()[n] for n in DEFAULT_SPEC.joint_names]


def test_geoms_are_boxes_with_expected_sizes(robot_root):
    geoms = robot_root.findall(".//geom")
    assert len(geoms) == DEFAULT_SPEC.n_bodies
    assert all(g.get("type") == "box" for g in geoms)
    pelvis = robot_root.find(".//geom[@name='pelvis_geom']")
    expected = [f / 2 for f in DEFAULT_SPEC.pelvis_size]
    assert [float(v) for v in pelvis.get("size").split()] == pytest.approx(expected)


def test_collision_masks(robot_root):
    for geom in robot_root.findall(".//geom"):
        name = geom.get("name")
        expected = ("1", "1") if name.startswith("foot_geom_") else ("2", "1")
        assert (geom.get("contype"), geom.get("conaffinity")) == expected, name


def test_scene_includes_robot_and_has_ground():
    scene = build_scene(DEFAULT_SPEC)
    assert scene.find("include").get("file") == "biped.xml"
    # 场景不能自带 compiler，否则和 include 进来的重复
    assert scene.find("compiler") is None
    assert scene.find("option") is not None
    assert scene.find(".//geom[@name='floor']") is not None
    assert scene.find(".//camera[@name='front']") is not None


def test_rendering_is_deterministic():
    assert render_robot_xml(DEFAULT_SPEC) == render_robot_xml(DEFAULT_SPEC)
    assert render_scene_xml(DEFAULT_SPEC) == render_scene_xml(DEFAULT_SPEC)


def test_rendered_text_is_valid_xml():
    for text in (render_robot_xml(DEFAULT_SPEC), render_scene_xml(DEFAULT_SPEC)):
        assert text.startswith("<?xml")
        assert text.endswith("\n")
        ET.fromstring(text)


def test_fnum_is_stable():
    assert fnum(0.1 + 0.2) == "0.3"
    assert fnum(-0.0) == "0"
    assert fnum(1.0) == "1"
    assert fnum(0.06) == "0.06"
    assert vec((0.0, -0.02, 1.5)) == "0 -0.02 1.5"


def test_write_models_creates_both_files(tmp_models):
    paths = write_models(DEFAULT_SPEC, tmp_models)
    assert [p.name for p in paths] == ["biped.xml", "scene.xml"]
    for path in paths:
        assert path.read_text(encoding="utf-8").startswith("<?xml")

    # 重复写出不应改变内容
    before = {p: p.read_text(encoding="utf-8") for p in paths}
    write_models(DEFAULT_SPEC, tmp_models)
    for path in paths:
        assert path.read_text(encoding="utf-8") == before[path]


def test_summary_mentions_dof_and_mass():
    text = summary(DEFAULT_SPEC)
    assert "nq=16" in text and "nu=9" in text and "kg" in text

"""校验器的测试：既要对真实模型通过，也要真的能抓到错误。"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from desktop_robot.builder import MODELS_DIR, write_models
from desktop_robot.config import DEFAULT_SPEC
from desktop_robot.verifier import verify_static


def test_real_models_pass(tmp_models):
    write_models(DEFAULT_SPEC, tmp_models)
    report = verify_static(DEFAULT_SPEC, tmp_models)
    assert report.ok, report.errors
    assert "总质量" in report.info


def test_committed_models_pass():
    report = verify_static(DEFAULT_SPEC, MODELS_DIR)
    assert report.ok, report.errors


def _tamper(workdir, mutate):
    """复制一份模型，用 ``mutate(root)`` 改坏它，返回模型目录。"""
    write_models(DEFAULT_SPEC, workdir)
    path = workdir / "biped.xml"
    root = ET.parse(path).getroot()
    mutate(root)
    path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return workdir


def test_detects_missing_files(tmp_models):
    report = verify_static(DEFAULT_SPEC, tmp_models)
    assert not report.ok
    assert any("缺少模型文件" in e for e in report.errors)


def test_detects_missing_actuator(tmp_models):
    def mutate(root):
        actuator = root.find("actuator")
        actuator.remove(list(actuator)[-1])

    report = verify_static(DEFAULT_SPEC, _tamper(tmp_models, mutate))
    assert not report.ok
    assert any("执行器数量" in e for e in report.errors)


def test_detects_wrong_joint_order(tmp_models):
    def mutate(root):
        root.findall(".//joint")[1].set("name", "not_the_hip")

    report = verify_static(DEFAULT_SPEC, _tamper(tmp_models, mutate))
    assert not report.ok
    assert any("关节名/顺序" in e for e in report.errors)


def test_detects_broken_keyframe_length(tmp_models):
    def mutate(root):
        root.find("./keyframe/key").set("ctrl", "0 0 0")

    report = verify_static(DEFAULT_SPEC, _tamper(tmp_models, mutate))
    assert not report.ok
    assert any("ctrl 长度" in e for e in report.errors)


def test_detects_wrong_pelvis_height_in_keyframe(tmp_models):
    def mutate(root):
        key = root.find("./keyframe/key")
        qpos = key.get("qpos").split()
        qpos[2] = "0.5"
        key.set("qpos", " ".join(qpos))

    report = verify_static(DEFAULT_SPEC, _tamper(tmp_models, mutate))
    assert not report.ok
    assert any("骨盆高度" in e for e in report.errors)


def test_detects_bad_collision_mask(tmp_models):
    def mutate(root):
        root.find(".//geom[@name='torso_geom']").set("contype", "1")

    report = verify_static(DEFAULT_SPEC, _tamper(tmp_models, mutate))
    assert not report.ok
    assert any("碰撞掩码" in e for e in report.errors)


def test_detects_scene_with_own_compiler(tmp_models):
    models = _tamper(tmp_models, lambda root: None)
    scene_path = models / "scene.xml"
    scene = ET.parse(scene_path).getroot()
    ET.SubElement(scene, "compiler", {"angle": "radian"})
    scene_path.write_text(ET.tostring(scene, encoding="unicode"), encoding="utf-8")

    report = verify_static(DEFAULT_SPEC, models)
    assert not report.ok
    assert any("不应自带" in e for e in report.errors)

"""把 MuJoCo 的 dino MJCF 导出成 URDF。

默认输入 ``models/dino.xml``，优先直接调用 ``mjcf2urdf`` 的 Python API 完成转换。
``dino_scene.xml`` 含地面/灯光/相机，更适合可视化；导出 URDF 时应优先用纯本体 ``dino.xml``。
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import subprocess
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "models" / "dino.xml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "urdf" / "dino"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert models/dino.xml to URDF via mjcf2urdf.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input MJCF file (default: models/dino.xml).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for converted URDF files.",
    )
    parser.add_argument(
        "--build-first",
        action="store_true",
        help="Run `python -m desktop_robot build-dino` before conversion.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _convert_with_python_api(input_path: Path, out_dir: Path) -> None:
    try:
        from mjcf2urdf import convert_mjcf_to_urdf
    except ImportError as exc:
        raise SystemExit(
            "找不到 `mjcf2urdf` Python 包。请先在当前环境安装：\n"
            f'  "{sys.executable}" -m pip install mjcf2urdf six'
        ) from exc

    convert_mjcf_to_urdf(str(input_path), str(out_dir))


def _guess_urdf_path(out_dir: Path, input_path: Path) -> Path | None:
    direct_match = out_dir / f"{input_path.stem}.urdf"
    if direct_match.exists():
        return direct_match
    candidates = sorted(out_dir.glob("*.urdf"))
    return candidates[0] if candidates else None


def _copy_collisions_to_visuals(urdf_path: Path) -> None:
    """把导出 URDF 里的 collision 复制成 visual，避免纯碰撞模型在查看器里空白。"""
    text = urdf_path.read_text(encoding="utf-8")
    text = text.replace('<?xml version="0.0" ?>', '<?xml version="1.0" ?>', 1)
    root = ET.fromstring(text)
    changed = False
    for link in root.findall("link"):
        collisions = list(link.findall("collision"))
        visuals = list(link.findall("visual"))
        if visuals or not collisions:
            continue
        for collision in collisions:
            visual = ET.Element("visual")
            for child in list(collision):
                visual.append(copy.deepcopy(child))
            material = ET.SubElement(visual, "material", {"name": "auto_gray"})
            ET.SubElement(material, "color", {"rgba": "0.65 0.68 0.72 1.0"})
            link.append(visual)
        changed = True
    if changed:
        ET.indent(root, space="  ")
        urdf_path.write_text(
            ET.tostring(root, encoding="unicode", xml_declaration=True),
            encoding="utf-8",
            newline="\n",
        )


def main() -> int:
    args = _parse_args()
    input_path = args.input.resolve()
    out_dir = args.out.resolve()

    if args.build_first:
        _run([sys.executable, "-m", "desktop_robot", "build-dino"])

    if not input_path.exists():
        raise SystemExit(f"输入 MJCF 不存在: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    _convert_with_python_api(input_path, out_dir)

    urdf_path = _guess_urdf_path(out_dir, input_path)
    if urdf_path is not None:
        _copy_collisions_to_visuals(urdf_path)
    print(f"MJCF: {input_path}")
    print(f"OUT : {out_dir}")
    if urdf_path is not None:
        print(f"URDF: {urdf_path}")
    else:
        print("已完成转换，但没有在输出目录里直接找到 .urdf 文件，请检查转换产物。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""用 ``models/dino_scene.xml`` 的机位重出 ``docs/`` 里的恐龙预览图。

产物：
* ``docs/preview_dino_iso.png``：iso 机位 960×720
* ``docs/preview_dino_ref.png``：ref 机位（接近参考图的 3/4 视角）960×720
* ``docs/dino_vs_reference.png``：左设计图（``build/ref_dino.png``）+ 右 ref 机位渲染

用法::

    python .agents/misc/py/render_dino_previews.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENE = REPO_ROOT / "models" / "dino_scene.xml"
DOCS = REPO_ROOT / "docs"
REFERENCE = REPO_ROOT / "build" / "ref_dino.png"

PANEL = 900  # 对照图单侧边长


def _render(model, data, camera: str, width: int, height: int) -> Image.Image:
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        return Image.fromarray(renderer.render())
    finally:
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--docs", type=Path, default=DOCS)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for name in ("iso", "ref"):
        image = _render(model, data, name, width=960, height=720)
        out = args.docs / f"preview_dino_{name}.png"
        image.save(out)
        print(f"wrote {out}")

    if REFERENCE.exists():
        left = Image.open(REFERENCE).convert("RGB").resize((PANEL, PANEL), Image.LANCZOS)
        right = _render(model, data, "ref", width=PANEL, height=PANEL)
        canvas = Image.new("RGB", (PANEL * 2 + 10, PANEL), "white")
        canvas.paste(left, (0, 0))
        canvas.paste(right, (PANEL + 10, 0))
        canvas.save(args.docs / "dino_vs_reference.png")
        print(f"wrote {args.docs / 'dino_vs_reference.png'}")


if __name__ == "__main__":
    main()

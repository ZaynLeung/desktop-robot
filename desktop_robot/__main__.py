"""命令行入口：``python -m desktop_robot {build,verify,view,simulate}``。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .builder import MODELS_DIR, summary, write_models
from .config import DEFAULT_SPEC
from .verifier import verify_static


def _configure_stdio() -> None:
    """让 Windows 控制台能正常输出中文（默认 cp1252 会直接抛 UnicodeEncodeError）。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):  # pragma: no cover - 重定向到管道时可能失败
                pass


def _print_report(report) -> None:
    for key, value in report.info.items():
        print(f"  {key}: {value}")
    for warning in report.warnings:
        print(f"  [警告] {warning}", file=sys.stderr)
    for error in report.errors:
        print(f"  [错误] {error}", file=sys.stderr)


def cmd_build(args: argparse.Namespace) -> int:
    out = Path(args.out)
    paths = write_models(DEFAULT_SPEC, out)
    print(f"已生成 {summary(DEFAULT_SPEC)}")
    for path in paths:
        print(f"  → {path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    report = verify_static(DEFAULT_SPEC, args.models)
    print(f"静态校验: {'通过' if report.ok else '失败'}")
    _print_report(report)
    if not report.ok:
        return 1

    if not args.dynamic:
        print("  （未指定 --dynamic，跳过 MuJoCo 动态校验）")
        return 0

    from . import sim  # noqa: PLC0415

    if not sim.has_mujoco():
        print("跳过动态校验：当前解释器没有安装 mujoco", file=sys.stderr)
        return 0

    print("动态校验（MuJoCo）:")
    result = sim.dynamic_check(Path(args.models) / "scene.xml", steps=args.steps)
    for key in ("nq", "nv", "nu", "nbody", "z_start", "z_end", "z_drift"):
        print(f"  {key}: {result[key]:.4f}")
    problems = result.get("problems") or []
    for problem in problems:
        print(f"  [错误] {problem}", file=sys.stderr)
    print(f"  → {'通过' if not problems else '失败'}")
    return 0 if not problems else 1


def cmd_view(args: argparse.Namespace) -> int:
    from . import sim  # noqa: PLC0415

    target = Path(args.models) / f"{args.model}.xml"
    if not target.exists():
        print(f"找不到模型 {target}，请先运行 python -m desktop_robot build", file=sys.stderr)
        return 1
    sim.view(target)
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from . import sim  # noqa: PLC0415

    target = Path(args.models) / f"{args.model}.xml"
    if not target.exists():
        print(f"找不到模型 {target}，请先运行 python -m desktop_robot build", file=sys.stderr)
        return 1
    stats = sim.run_headless(target, steps=args.steps)
    print("无头仿真结果:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m desktop_robot",
        description="桌面双足机器人的参数化 MuJoCo 模型工具",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="根据 config.py 生成 models/*.xml")
    p_build.add_argument("--out", default=str(MODELS_DIR), help="输出目录（默认 models/）")
    p_build.set_defaults(func=cmd_build)

    p_verify = sub.add_parser("verify", help="校验生成物与参数源是否一致")
    p_verify.add_argument("--models", default=str(MODELS_DIR), help="模型目录")
    p_verify.add_argument("--dynamic", action="store_true", help="额外用 MuJoCo 加载并试跑")
    p_verify.add_argument("--steps", type=int, default=1000, help="动态校验步数")
    p_verify.set_defaults(func=cmd_verify)

    p_view = sub.add_parser("view", help="打开 MuJoCo 查看器")
    p_view.add_argument("--models", default=str(MODELS_DIR), help="模型目录")
    p_view.add_argument(
        "--model", default="scene", choices=("scene", "biped"), help="要打开的模型文件"
    )
    p_view.set_defaults(func=cmd_view)

    p_sim = sub.add_parser("simulate", help="无头仿真，保持站姿并输出基座高度")
    p_sim.add_argument("--models", default=str(MODELS_DIR), help="模型目录")
    p_sim.add_argument(
        "--model", default="scene", choices=("scene", "biped"), help="要仿真的模型文件"
    )
    p_sim.add_argument("--steps", type=int, default=2000, help="仿真步数")
    p_sim.set_defaults(func=cmd_simulate)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

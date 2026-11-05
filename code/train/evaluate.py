"""评测入口：``python code/train/evaluate.py --controller lift --episodes 10``"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from dino_wbc import model_io  # noqa: E402
from dino_wbc.evaluate import evaluate  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="evaluate.py", description="评测一个控制器")
    parser.add_argument("--controller", default="scripted_stand")
    parser.add_argument("--checkpoint", default=None, help="覆盖 registry 里的 checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--pose",
        default="all",
        help="standup 起始姿态：all / crouch / side / supine",
    )
    parser.add_argument(
        "--lift-h",
        type=float,
        default=None,
        help="仅 lift：覆盖抬脚高度（m），例如 0.02 / 0.03 / 0.04",
    )
    parser.add_argument("--json", default=None, help="把完整结果写到该 JSON 文件")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model, layout = model_io.load(args.model)
    contract = model_io.contract_hash(model, layout)

    from dino_wbc.controllers.registry import load_controller

    controller = load_controller(
        args.controller,
        expected_contract=contract,
        checkpoint=args.checkpoint,
    )
    result = evaluate(
        controller,
        episodes=args.episodes,
        model_path=args.model,
        seed=args.seed,
        max_steps=args.max_steps,
        standup_pose=args.pose,
        lift_h=args.lift_h,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    for record in result["episodes"]:
        pose = f" pose={record['pose']}" if record.get("pose") else ""
        print(
            f"  ep{record['episode']:02d}{pose} success={record['success']} "
            f"fallen={record['fallen']} reward={record['reward']:+.2f} "
            f"steps={record['steps']} z={record['base_z']:.3f} "
            f"tilt={record['tilt_deg']:.1f} "
            f"foot_z={record['target_foot_z_max']:.3f} "
            f"body={record['body_contact_rate']:.2f}"
        )
    if args.json:
        Path(args.json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[json] {args.json}")
    controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

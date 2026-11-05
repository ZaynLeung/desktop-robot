"""训练入口：``python code/train/train.py --stage <stage> ...``

只负责 PPO 训练与产物落盘；运行端在 ``code/run/run.py``，两者互不 import。
训练产物：

* ``<out>/policy.pt``     运行端加载的确定性策略（eval 刷新时更新）
* ``<out>/model.zip``     SB3 模型，用于 ``--init-from`` 继续训练
* ``<out>/best_model.zip``
* ``<out>/meta.json``

``--check`` 只建环境做一次 reset/step 自检，不训练、不依赖 stable-baselines3。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# 允许从仓库任意目录执行：把 code/ 放进 sys.path
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402

from dino_wbc import config, model_io  # noqa: E402
from dino_wbc.env import DinoWbcEnv  # noqa: E402


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):  # pragma: no cover
                pass


def linear_schedule(initial: float):
    def _schedule(progress_remaining: float) -> float:
        return initial * progress_remaining

    return _schedule


def _parse_arch(text: str) -> list[int]:
    """``"256,128"`` → ``[256, 128]``；空串报错。"""
    hidden = [int(part) for part in str(text).replace("x", ",").split(",") if part.strip()]
    if not hidden:
        raise ValueError("--net-arch 不能为空，例如 256,128")
    return hidden


def make_env(stage: str, seed: int, model_path: str | None):
    """返回一个可被 SubprocVecEnv 序列化的环境工厂。"""

    def _init() -> DinoWbcEnv:
        return DinoWbcEnv(stage, seed=seed, model_path=model_path)

    return _init


def _build_venv(env_fns, mode: str, dummy_cls, subproc_cls):
    """建向量环境；子进程不可用时（受限沙箱里 multiprocessing 管道会被拒绝）回退单进程。

    MuJoCo 每个环境都要占 CPU，``subproc`` 才是正式训练该用的；``dummy`` 只是
    保证在任何权限环境下都能把流程跑通。
    """
    if mode == "dummy" or len(env_fns) == 1:
        return dummy_cls(env_fns)
    if mode == "subproc":
        return subproc_cls(env_fns, start_method="spawn")
    try:
        return subproc_cls(env_fns, start_method="spawn")
    except (PermissionError, OSError) as exc:
        print(
            f"[warn] 子进程向量环境创建失败（{exc}）；回退 DummyVecEnv 单进程。\n"
            "       受限沙箱会禁止 multiprocessing 管道，正式训练请在普通终端里跑。"
        )
        return dummy_cls(env_fns)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="小恐龙 WBC 的 PPO 训练入口（训练代码与运行代码分离）",
    )
    parser.add_argument("--stage", required=True, choices=tuple(config.STAGES))
    parser.add_argument("--steps", type=int, default=4_000_000, help="总环境步数")
    parser.add_argument("--envs", type=int, default=16, help="并行环境数")
    parser.add_argument(
        "--vec-env",
        choices=("auto", "subproc", "dummy"),
        default="auto",
        help="向量环境实现；auto 在子进程不可用（受限沙箱/无管道权限）时回退 dummy",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--model", default=None, help="MuJoCo 模型路径（默认 models/dino_scene.xml）")
    parser.add_argument("--out", default=None, help="产物目录（默认 code/checkpoints/<stage>_<时间戳>）")
    parser.add_argument("--logdir", default=None, help="TensorBoard 目录（默认 code/runs/...）")
    parser.add_argument("--init-from", default=None, help="用已有 SB3 .zip 热启动")
    parser.add_argument("--n-steps", type=int, default=2048, help="每个环境每次 rollout 的步数")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--net-arch",
        default="256,128",
        help="策略/价值网络隐层，逗号分隔（默认 256,128；容量不足时可试 512,256）",
    )
    parser.add_argument("--ent-coef", type=float, default=0.005, help="PPO 熵系数")
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=-1.0,
        help="初始动作标准差 ln 值（越小起步越温和；默认 -1.0≈0.37，"
        "实测比默认 0 的 ep_len/回报好一个量级）",
    )
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--check", action="store_true", help="只做环境自检后退出")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="训练结束后不自动更新 code/registry.json",
    )
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args(argv)


def _resolve_device(requested: str) -> str:
    import torch  # 延迟导入，--check 不需要

    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "请求了 --device cuda 但 torch.cuda.is_available() 为 False；"
            "检查是否安装了 cu128 轮子/显卡驱动"
        )
    else:
        device = requested
    print(f"[torch] {torch.__version__}  device={device}")
    if torch.cuda.is_available():
        print(f"[cuda ] {torch.cuda.get_device_name(0)}  sm={torch.cuda.get_device_capability(0)}")
    return device


def run_check(args: argparse.Namespace) -> int:
    """不训练的最小自检：reset → 随机动作 → 打印观测/奖励。"""
    env = DinoWbcEnv(args.stage, seed=args.seed, model_path=args.model)
    obs, info = env.reset(seed=args.seed)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"阶段          : {args.stage}")
    print(f"观测/动作维度 : {obs.shape} / {action.shape}")
    print(f"finite        : {bool(np.isfinite(obs).all())}")
    print(f"reward        : {reward:+.4f}")
    print(f"terminated/truncated: {terminated}/{truncated}")
    print(f"base_z        : {info['base_z']:.4f}  tilt={info['tilt_deg']:.1f}deg")
    print(f"command       : {np.round(info['command'], 3)}")
    env.close()
    return 0


def run_training(args: argparse.Namespace) -> int:
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except ImportError as exc:  # pragma: no cover
        print(
            "缺少训练依赖。请先安装（详见 code/requirements-wbc.txt）：\n"
            r"  E:\conda_envs\desktop-robot\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128"
            "\n"
            r"  E:\conda_envs\desktop-robot\python.exe -m pip install stable-baselines3==2.9.2 tensorboard opencv-python",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    from callbacks import CurriculumLogCallback, ExportBestPolicyCallback

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else CODE_DIR / "checkpoints" / f"{args.stage}_{timestamp}"
    log_dir = Path(args.logdir) if args.logdir else CODE_DIR / "runs" / f"{args.stage}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)

    # 契约哈希：训练与运行必须看到同一个模型/关节布局
    model, layout = model_io.load(args.model)
    contract = model_io.contract_hash(model, layout)
    print(f"[model] {model_io.resolve_model_path(args.model)}")
    print(f"[model] nq={model.nq} nv={model.nv} nu={model.nu} joints={layout.num_joints}")
    print(f"[contract] {contract}")

    env_fns = [make_env(args.stage, args.seed + rank, args.model) for rank in range(args.envs)]
    venv = _build_venv(env_fns, args.vec_env, DummyVecEnv, SubprocVecEnv)
    venv = VecMonitor(venv, filename=str(log_dir / "monitor.csv"))

    eval_env = DummyVecEnv([make_env(args.stage, args.seed + 10_000, args.model)])
    eval_env = VecMonitor(eval_env)

    policy_kwargs = {
        "net_arch": {"pi": _parse_arch(args.net_arch), "vf": _parse_arch(args.net_arch)},
        "activation_fn": torch.nn.ELU,
        "log_std_init": float(args.log_std_init),
    }
    print(
        f"[net] {args.net_arch}  ent_coef={args.ent_coef}  "
        f"log_std_init={args.log_std_init}"
    )

    if args.init_from:
        init_path = Path(args.init_from)
        if init_path.suffix == ".pt":
            raise SystemExit(
                "policy.pt 是运行端格式，不能直接热启动；"
                "请用同一训练目录里的 model.zip：--init-from <run>/model.zip"
            )
        print(f"[init] 从 {init_path} 热启动")
        agent = PPO.load(str(init_path), env=venv, device=device)
        agent.tensorboard_log = str(log_dir)
    else:
        agent = PPO(
            "MlpPolicy",
            venv,
            learning_rate=linear_schedule(args.lr),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            verbose=args.verbose,
            device=device,
            seed=args.seed,
        )

    eval_callback = ExportBestPolicyCallback(
        eval_env,
        best_model_save_path=str(out_dir),
        log_path=str(log_dir),
        eval_freq=max(args.eval_freq // max(args.envs, 1), 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        export_path=out_dir / "policy.pt",
        contract_hash=contract,
        stage=args.stage,
        verbose=args.verbose,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(200_000 // max(args.envs, 1), 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix=args.stage,
    )
    callbacks = CallbackList(
        [CurriculumLogCallback(), eval_callback, checkpoint_callback]
    )

    started = time.time()
    agent.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - started

    from dino_wbc.policy_io import export_from_sb3

    agent.save(str(out_dir / "model.zip"))
    export_from_sb3(
        agent,
        out_dir / "policy.pt",
        contract_hash=contract,
        joint_names=config.JOINT_NAMES,
        meta={
            "stage": args.stage,
            "steps": int(args.steps),
            "elapsed_s": round(elapsed, 1),
            "seed": args.seed,
        },
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "format": "dino_wbc.train_run.v1",
                "stage": args.stage,
                "steps": args.steps,
                "envs": args.envs,
                "seed": args.seed,
                "device": device,
                "contract_hash": contract,
                "model": str(model_io.resolve_model_path(args.model)),
                "elapsed_s": round(elapsed, 1),
                "created": timestamp,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.no_register:
        _register_checkpoint(args.stage, out_dir / "policy.pt", contract)

    venv.close()
    eval_env.close()
    print(f"[done] {args.steps} 步，用时 {elapsed / 60:.1f} min")
    print(f"[done] policy: {out_dir / 'policy.pt'}")
    print(f"[done] 运行：python code/run/run.py --controller {args.stage} --checkpoint {out_dir / 'policy.pt'}")
    return 0


def _register_checkpoint(stage: str, policy_path: Path, contract: str) -> None:
    """把本次训练产物写进 ``code/registry.json``，运行端 ``--controller <stage>`` 直接可用。"""
    registry_path = CODE_DIR / "registry.json"
    payload: dict = {}
    if registry_path.exists():
        try:
            loaded = json.loads(registry_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}
    resolved = Path(policy_path).resolve()
    try:
        checkpoint = resolved.relative_to(CODE_DIR).as_posix()
    except ValueError:  # checkpoint 在 code/ 之外（--out 指到别处）
        checkpoint = resolved.as_posix()
    payload[stage] = {
        "stage": stage,
        "checkpoint": checkpoint,
        "contract_hash": contract,
    }
    registry_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[registry] {stage} → {checkpoint}")


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "wgl")
    if args.check:
        return run_check(args)
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())

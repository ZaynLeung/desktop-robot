"""MuJoCo 相关的运行时入口。

全部 MuJoCo 依赖都集中在这里，并且是**延迟导入**的：没装 MuJoCo 时，
``build`` / ``verify`` 以及整个测试套件照常工作，只有 ``view`` / ``simulate``
会给出可操作的中文提示而不是一堆 traceback。
"""

from __future__ import annotations

from pathlib import Path

INSTALL_HINT = (
    "未找到 MuJoCo。请先激活项目环境并安装依赖：\n"
    "    conda activate E:\\conda_envs\\desktop-robot\n"
    "    pip install -r requirements.txt\n"
    "（没有 MuJoCo 时，模型的静态校验仍然可用：python -m desktop_robot verify）"
)


def require_mujoco():
    """导入 mujoco，失败时以清晰的提示退出。"""
    try:
        import mujoco  # noqa: PLC0415 - 故意延迟导入
    except ImportError:  # pragma: no cover - 取决于运行环境
        raise SystemExit(INSTALL_HINT) from None
    return mujoco


def has_mujoco() -> bool:
    """当前解释器是否装了 MuJoCo。"""
    try:
        import mujoco  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def load(path: Path | str):
    """加载模型并返回 ``(mujoco, model, data)``。"""
    mj = require_mujoco()
    model = mj.MjModel.from_xml_path(str(path))
    data = mj.MjData(model)
    return mj, model, data


def reset_to_stand(mj, model, data) -> bool:
    """如果模型里有 ``stand`` 关键帧就复位到它，返回是否成功。"""
    key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        mj.mj_forward(model, data)
        return False
    mj.mj_resetDataKeyframe(model, data, key_id)
    mj.mj_forward(model, data)
    return True


def run_headless(path: Path | str, steps: int = 2000) -> dict[str, float]:
    """按 ``stand`` 关键帧无头跑若干步，返回稳定性指标。

    指标含义：位置源保持站姿时，基座高度不应该明显下坠或跳变。
    """
    mj, model, data = load(path)
    reset_to_stand(mj, model, data)

    qpos_start = data.qpos.copy()
    z0 = float(qpos_start[2])

    heights: list[float] = []
    for _ in range(max(1, steps)):
        mj.mj_step(model, data)
        heights.append(float(data.qpos[2]))

    return {
        "nq": float(model.nq),
        "nv": float(model.nv),
        "nu": float(model.nu),
        "nbody": float(model.nbody),
        "z_start": z0,
        "z_end": heights[-1],
        "z_min": min(heights),
        "z_max": max(heights),
        "z_drift": heights[-1] - z0,
        "steps": float(steps),
    }


def view(path: Path | str) -> None:
    """用 MuJoCo 被动查看器打开模型，直到关闭窗口。"""
    mj, model, data = load(path)
    reset_to_stand(mj, model, data)

    import time  # noqa: PLC0415

    with mj.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.2
        viewer.cam.lookat[:] = [0.0, 0.0, 0.16]
        while viewer.is_running():
            mj.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def dynamic_check(path: Path | str, steps: int = 1000) -> "dict[str, object]":
    """动态校验：能不能加载、维度对不对、站姿会不会塌。"""
    result: dict[str, object] = {}
    stats = run_headless(path, steps=steps)
    result.update(stats)
    problems: list[str] = []
    if abs(stats["z_drift"]) > 0.02:
        problems.append(
            f"基座高度漂移 {stats['z_drift']:+.4f} m 超过 0.02 m 阈值"
            f"（可能关节力矩不足或姿态不贴地）"
        )
    result["problems"] = problems
    return result

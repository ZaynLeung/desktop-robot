# 小恐龙 WBC 控制工程（`code/`）

`desktop_dino`（18 个位置伺服关节 + 浮动基座）的强化学习全身控制（WBC）：**站立并分别抬左右脚 → 仰卧/侧躺翻正起身 → 前后左右行走**。

设计上有两条硬边界：

* **训练端与运行端分离**：`code/train/` 依赖 stable-baselines3；`code/run/` 只用 numpy / mujoco / torch / opencv，绝不 import SB3。
* **控制器接口统一**：无论学到的策略还是脚本基线，都实现同一个 `Controller` 接口；运行端只认 checkpoint，不认训练代码。

```
code/
├─ dino_wbc/                 共享库（观测/动作/奖励/环境/控制器/相机）
├─ train/                    训练 + 评测（PPO / SB3）
├─ run/                      运行端（双画面实时仿真）
├─ tests/                    pytest（与根目录 tests/ 一起收集）
├─ registry.json             控制器名 → checkpoint 映射
├─ checkpoints/  runs/  videos/   产物（gitignore）
└─ requirements-wbc.txt
```

---

## 1. 环境与依赖

固定环境：`E:\conda_envs\desktop-robot`（Python 3.11，mujoco 3.17 / gymnasium 1.4 / numpy 2.4 已装好）。

额外依赖（装到同一个环境）：

```powershell
$py = "E:\conda_envs\desktop-robot\python.exe"

# torch 必须 cu128 及以上：RTX 5060 Laptop 是 sm_120，cu126 及以下跑不了
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128

& $py -m pip install -r code/requirements-wbc.txt

# 自检（必须看到 cuda True）
& $py -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

> 注意：本仓库的沙箱/权限设置可能让 pip 回退到 user site。务必确认安装后
> `python -c "import torch;print(torch.__file__)"` 的路径在 `E:\conda_envs\desktop-robot` 下。

---

## 2. 自由度 / 状态量速查

| 项 | 值 |
| --- | --- |
| 驱动关节 `nu` | **18**：头 2、肩 2、每条腿 5（hip_pitch / hip_roll / hip_yaw / knee / ankle）、尾 4 |
| 配置维 `nq` / 速度维 `nv` | 25 / 24（含 6 自由度浮动基座） |
| 执行器 | `position`：`ctrl` = 站姿相对目标角，kp=25、kv=1、±2.5 N·m（踝 ±2.0） |
| 观测 | **80 维** |
| 动作 | 18 维归一化，`target = clip(a,-1,1) × scale`，再裁到关节范围 |

关节零位 = 标称站姿（`stand` 关键帧），所以 `ctrl = 0` 就是站姿。

80 维观测：

| 块 | 维 | 内容 |
| --- | --- | --- |
| base | 10 | 基座高度误差×10、本体系投影重力(3)、本体系线速度(3)、本体系角速度(3) |
| joints | 36 | 关节角 q(18)、关节角速度 qd(18)×0.05 |
| feet | 6 | 足底离地高度(2)×10、接触(2)、水平滑移(2)×10 |
| action | 18 | 上一步动作 |
| command | 8 | 任务 one-hot(4) + vx + vy + 抬脚高度 + 目标脚 |
| phase | 2 | 步态时钟 sin/cos（仅 walk 非零） |

命令向量内部是物理单位（m/s、m、±1），拼观测时才缩放；奖励直接拿它当目标值。

---

## 3. 训练

```powershell
$py = "E:\conda_envs\desktop-robot\python.exe"
cd D:\2026\HKUST\EESM-Robotic\desktop

# 0) 不训练，只做环境自检（任何阶段都能跑）
& $py code/train/train.py --stage lift --check

# 1) 站立 + 外力轻推（所有后续阶段的初始化来源）
& $py code/train/train.py --stage stand --envs 16 --steps 4000000 --device auto

# 2) 站立 + 分别抬左右脚（从 stand 热启动）
& $py code/train/train.py --stage lift --envs 16 --steps 8000000 --device auto `
    --init-from code/checkpoints/stand_<时间戳>/model.zip

# 3) 翻正起身（课程：深蹲 → 侧躺 → 仰卧）
& $py code/train/train.py --stage standup --envs 16 --steps 8000000 --device auto `
    --init-from code/checkpoints/lift_<时间戳>/model.zip

# 4) 前后左右行走
& $py code/train/train.py --stage walk --envs 16 --steps 6000000 --device auto `
    --init-from code/checkpoints/lift_<时间戳>/model.zip
```

> **为什么先训 stand**：v1 直接从 lift 开始，策略学成了“躯干撑地 + 蹲着”的局部最优
> （实测 base z 0.113 m、躯干着地 211/400 步）。v2 把 stand 做成正式阶段
> （严格摔倒门限 + 躯干接触即终止 + 0→5 N 轻推课程），lift 从它的 `model.zip` 热启动。
> `--log-std-init` 默认 **-1.0**（初始动作更温和）；50k 步对照实验里 ep_len 33→164、
> 平均回报 84→459，是本站立任务的关键超参。


产物（默认 `code/checkpoints/<stage>_<时间戳>/`）：

| 文件 | 用途 |
| --- | --- |
| `policy.pt` | 运行端加载的确定性策略（eval 刷新时更新） |
| `model.zip` | SB3 模型，用 `--init-from` 继续训练/迁移 |
| `best_model.zip` | eval 最优 |
| `meta.json` | 阶段、步数、device、契约哈希 |
| `checkpoints/` | 定期快照 |

训练结束会自动把该阶段的 `policy.pt` 写进 `code/registry.json`（`--no-register` 可关闭），
所以之后 `--controller lift` 不用再手写路径。

> **并行环境**：默认 `--vec-env auto` = 用 `SubprocVecEnv`（16 个子进程）；在受限沙箱里
> 子进程管道可能被拒绝，会自动回退到 `DummyVecEnv`（单进程、慢很多）。正式训练请在
> 普通 PowerShell 里跑，或显式 `--vec-env subproc` 让问题暴露出来。

TensorBoard：`tensorboard --logdir code/runs`（可看 `curriculum/*`、`reward_terms/*`）。

**评测**：

```powershell
& $py code/train/evaluate.py --controller lift --episodes 20
& $py code/train/evaluate.py --checkpoint code/checkpoints/lift_<时间戳>/policy.pt --episodes 20
```

输出 `success_rate / fall_rate / mean_reward / mean_base_z / mean_tilt_deg / mean_vel_track`。

---

## 4. 运行（CPU 推理）

```powershell
$py = "E:\conda_envs\desktop-robot\python.exe"

# 先把 code/registry.json 里的 checkpoint 指向你的训练产物，然后：
& $py code/run/run.py --controller lift

# 也可以直接给路径
& $py code/run/run.py --checkpoint code/checkpoints/lift_<时间戳>/policy.pt

# 录视频 / 无窗口 / 快速跑完
& $py code/run/run.py --controller walk --record code/videos/walk.mp4
& $py code/run/run.py --controller lift --headless --seconds 10
& $py code/run/run.py --controller lift --fast
```

画面：**左 = 第三人称跟随**（平滑跟踪躯干），**右 = 头部第一视角**（`cam_head`，绑定 `head` body），底部 HUD 显示控制器/命令/基座高度/倾角/接触/FPS。

按键（非 headless）：`1–4` 热切换 stand / lift / standup / walk，`W/S` 调 vx，`A/D` 调 vy，`L/R` 抬左/右脚，`U` 起身命令，`V` 循环 standup 起始姿态（crouch / side / supine，默认仰卧），`P` 手动轻推一下（5 N × 0.1 s），`Space` 复位，`F` 切换跟随相机模式，`Esc` 退出。

`--lift-h 0.03` 可以改抬脚高度指令（默认 0.025 m，夹在 `LIFT_H_RANGE` 内）。

> 切到 `standup` 时运行端会用 `dino_wbc.state.apply_standup_pose` 把机器人摆成躺倒姿态
> （与训练 reset 完全同一个实现），所以能直接看到"从仰卧站起来"的过程。

---

## 5. 统一控制器接口与 checkpoint 契约

```python
class Controller(ABC):
    name: str          # 展示名
    stage: str         # stand / lift / standup / walk
    obs_dim: int
    act_dim: int
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...   # [-1,1]^18
    def close(self) -> None: ...
    def describe(self) -> dict: ...
```

* 内置 `scripted_stand` / `scripted_lift`（不依赖 checkpoint，用于冒烟）。
* `policy.pt` 含 `contract_hash`（关节名序列 + 关节范围 + 观测维度 + 动作尺度 + nq/nv/nu 的指纹）。
  运行端加载时用**当前模型**重算并比对；模型改过（比如重新 build-dino）会直接报错，
  提示重新训练或换 checkpoint——不会静默地跑错。

新增一个控制器：

1. 在 `dino_wbc/controllers/` 里实现 `Controller`；
2. 在 `registry.py` 里登记；
3. 在 `code/registry.json` 里加名字 → checkpoint。

---

## 6. 四个阶段与课程

| 阶段 | reset | 关键奖励 | 终止 |
| --- | --- | --- | --- |
| `stand` | 站姿 + 噪声；0–5 N 轻推课程 | 高度（σ 2 cm）+ 相对倾角 + 双足接触 + 低姿态/关节姿态惩罚 | 躯干/头/臂/腿着地、z<z0−0.06、倾角>40°、出界 |
| `lift` | 站姿 + 噪声；随机目标脚、抬脚高度 | 站立项 + **姿态门限内**的抬脚高度/线性抬脚/支撑脚接触/重心靠支撑脚 | 同 stand（躯干着地即终止） |
| `standup` | 课程：深蹲(z≈0.14) → 侧躺(90°) → 仰卧(180°) | 势函数进度 `5(γΦ_t−Φ_{t-1})` + 成功 +5 + 脚找地 | 连续站立 0.5 s 成功；超时/出界 |
| `walk` | 站姿 + 随机速度指令 | 速度跟踪 + 站直/高度 + 摆动足离地 + 交替接触 + 无偏航 | 跌倒/出界/超时 |

* 尾巴着地**允许**（T-rex 的尾巴本来就是配重/支撑），只给 −0.2/步的小惩罚；躯干/头/臂/腿着地会直接终止。
* `stand` 成功 = 整集不摔 + 平均高度项 >0.8；`lift` 成功 = **连续**保持 ≥0.5 s（不是累计），且每一步都满足 `foot_z ≥ 0.8h`、支撑脚接触、姿态门限。
* 课程推进：`stand` 只推轻推力度（0→5 N）；`lift` 抬脚高度 1→4 cm（成功 +0.03、失败 −0.01）；`walk` 速度随成功率爬坡；`standup` ≥70% 晋级、≤30% 退档。


所有数值集中在 `dino_wbc/config.py`；奖励权重在 `STAGES[...].reward`。

---

## 7. 调参入口

| 想改什么 | 改哪里 |
| --- | --- |
| 奖励权重 / episode 长度 | `dino_wbc/config.py` 的 `STAGES` |
| 动作幅度 | `ACTION_SCALE_BY_JOINT` |
| 观测缩放/裁剪 | `config.py` 的 `*_SCALE` / `*_CLIP` |
| 抬脚高度、行走速度范围 | `LIFT_H_RANGE` / `WALK_VX_RANGE` / `WALK_VY_RANGE` |
| 起身起始姿态 | `STANDUP_POSES`（关节角可直接调） |
| 跌倒/成功判据 | `FALL_*` / `SUCCESS_*` |
| 执行器 PD 增益（不改 XML） | `model_io.load(..., kp=, kv=, forcerange=)`（训练/运行传同一组） |

---

## 8. 常见问题

| 现象 | 处理 |
| --- | --- |
| `torch.cuda.is_available()` 为 False | 确认装的是 cu128 轮子、驱动 ≥ 对应 CUDA 版本；先用 `--device cpu` 跑通流程 |
| `PolicyContractError` | 模型改过（重新 build-dino / 改关节），用当前模型重新训练 |
| `找不到 checkpoint` | 先训练，或把 `--checkpoint` 指向已有 `policy.pt` |
| 抬脚时机器人趴/蹲在地上 | v1 的局部最优；已修（躯干着地终止 + 严格高度门限 + 连续保持课程）。若仍出现，先确认 `stand` 已训到验收标准，再用它 warm-start `lift` |
| stand 训得很慢 / ep_len 上不去 | 用默认 `--log-std-init -1.0`；确认没有把 `--device cuda` 当默认（物理在 CPU）；必要时把 `--envs` 提到 16 并在普通终端跑 |
| 按 `Space`/`1–4` 后窗口假死 | 已修：渲染节拍随复位回退 + 每控制步轮询 `cv2.pollKey` |
| 没有窗口/远程环境 | 用 `--headless`（可配合 `--record`） |
| 渲染太慢 | `--fast`、降低 `RENDER_WIDTH/HEIGHT`、或 `--no-head` |
| 训练很慢 | 减小 `--envs`（默认 16）观察 CPU 占用；物理在 CPU，GPU 只加速网络 |
| 起身一直学不会 | 先确认 `side` 档能成功，再放开 `supine`；必要时加步数/调奖励，或按计划里的备选（提高腿部 `forcerange`） |

---

## 9. 已知限制

* 没有腰 yaw：不能靠躯干原地转身；侧向行走靠 `hip_roll`/`hip_yaw` 小步幅。
* `kp=25` 偏软，是原模型的执行器设定；需要更强跟随可在 `model_io.load` 里覆盖。
* 仰卧翻正是三阶段里最难的（约 2 kg、单段手臂、±2.5 N·m）；首轮目标见计划文档。
  若仰卧迟迟学不会，依次尝试：加步数 → 调奖励/课程 → 经确认后提高腿部 `forcerange`。
* SB3 会对 MLP 策略用 GPU 训练给出警告：物理在 CPU、网络很小，GPU 收益有限；
  `--device cpu` 往往更快，`--device auto` 保留可用 GPU。
* 只做仿真，不含 sim2real。

# 使用
<!-- '''
$py = "E:\conda_envs\desktop-robot\python.exe"
cd D:\2026\HKUST\EESM-Robotic\desktop
''' -->

## 0 站立 + 轻推（所有后续阶段的初始化来源）
& $py code/train/train.py --stage stand --envs 16 --steps 4000000 --device auto

##  1 站立 + 抬左右脚（用上一步的 model.zip 热启动）
& $py code/train/train.py --stage lift --envs 16 --steps 8000000 --device auto `
    --init-from code/checkpoints/stand_<时间戳>/model.zip

## 2 翻正起身（用上一步的 model.zip 热启动）
& $py code/train/train.py --stage standup --envs 16 --steps 8000000 --device auto `
    --init-from code/checkpoints/lift_<时间戳>/model.zip

## 3 前后左右行走
& $py code/train/train.py --stage walk --envs 16 --steps 6000000 --device auto `
    --init-from code/checkpoints/lift_<时间戳>/model.zip

## 评测 / 运行（训练结束后 registry.json 已自动指向对应 checkpoint）
& $py code/train/evaluate.py --controller stand --episodes 20
& $py code/train/evaluate.py --controller lift --episodes 20 --lift-h 0.02
& $py code/run/run.py --controller stand
& $py code/run/run.py --controller lift --lift-h 0.02 --record code/videos/lift.mp4
& $py code/run/run.py --controller standup --record code/videos/standup.mp4
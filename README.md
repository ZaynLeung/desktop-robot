# desktop-robot

EESM 5800「桌面伴侣机器人」课程项目的仿真工程。

仓库里目前有**两台桌面机器人本体**，都可以直接用 MuJoCo 打开；整机参数由 Python 生成，
模型文件（`models/*.xml`）是确定性的生成物：

| 机型 | 造型 | 驱动关节 | 站立高度 | 总质量 | `nq / nv / nu` | 模型文件 |
| --- | --- | --- | --- | --- | --- | --- |
| `desktop_biped` | 腰部 + 双足 | 9 | ≈ 0.310 m | ≈ 1.51 kg | 16 / 15 / 9 | `models/biped.xml` |
| `desktop_dino` | 霸王龙（头 + 双臂 + 双足 + 4 节尾 + 头部相机） | 18 | ≈ 0.307 m | ≈ 2.00 kg | 25 / 24 / 18 | `models/dino.xml` |

设计细节、坐标约定、简化假设见：
**双足** → [`docs/model_design.md`](docs/model_design.md)；
**恐龙** → [`docs/dino_model.md`](docs/dino_model.md)。

![双足模型预览（45° 视角）](docs/preview_iso.png)

![恐龙模型预览（参考视角）](docs/preview_dino_ref.png)

---

## 环境

固定用这个环境（已建好）：

```
E:\conda_envs\desktop-robot        conda 环境，Python 3.11
```

选 Python 3.11 是因为它是 MuJoCo + Gymnasium + Stable-Baselines3 + PyTorch
这一套强化学习栈目前兼容性最好的版本。

```powershell
conda activate E:\conda_envs\desktop-robot

# 若环境丢失，可以重建：
conda env create -f environment.yml -p E:\conda_envs\desktop-robot
# 或者进入任意 3.11 环境后：
pip install -r requirements.txt
```

三分钟上手：

```powershell
conda activate E:\conda_envs\desktop-robot
cd D:\2026\HKUST\EESM-Robotic\desktop

# 双足机型
python -m desktop_robot build     # 生成 models/biped.xml 与 models/scene.xml
python -m desktop_robot verify    # 静态校验（加 --dynamic 会用 MuJoCo 试跑）
python -m desktop_robot view      # 打开 MuJoCo 查看器，按 Esc / 关窗退出

# 恐龙机型
python -m desktop_robot build-dino     # 生成 models/dino.xml 与 models/dino_scene.xml
python -m desktop_robot verify-dino    # 静态校验（加 --dynamic 会用 MuJoCo 试跑）
python -m desktop_robot view --model dino_scene

python -m pytest                  # 跑全部测试
```

没有装 MuJoCo 也能用：`build` / `build-dino` / `verify`（静态部分）/ 运动学测试都不依赖它，
`view`、`simulate` 和 `tests/*mujoco*.py` 会在缺包时给出提示或自动跳过。

---

## 小恐龙 WBC 控制工程（`code/`）

`code/` 里是一套完整的强化学习全身控制（WBC）工程：**站立并分别抬左右脚 → 仰卧/侧躺翻正起身 → 前后左右行走**。
要点：

* 训练端（`code/train/`，PPO + Stable-Baselines3）与运行端（`code/run/`，CPU 推理）完全分离；
* 控制器统一接口，运行端只按名字/路径加载不同 checkpoint（`code/registry.json`）；
* 运行画面 = 第三人称跟随 + 头部第一视角（`cam_head` 绑定 head body）双画面，可录制 mp4；
* 策略是 2×256 MLP，CPU 即可实时跑；训练可走 GPU，但物理在 CPU。

```powershell
$py = "E:\conda_envs\desktop-robot\python.exe"

& $py code/train/train.py --stage lift --check          # 不训练，只自检环境
& $py code/train/train.py --stage lift --envs 16 --device auto
& $py code/train/evaluate.py --controller lift --episodes 20
& $py code/run/run.py --controller lift                 # 双画面实时运行
```

安装、三阶段课程、按键、契约哈希与排障见 **[code/README.md](code/README.md)**。

---

## 目录结构

```
desktop/
├─ desktop_robot/            参数化模型生成器（Python 包，扁平布局，无需 pip install）
│  ├─ config.py              ★ 双足参数源：尺寸、质量、关节范围、力矩、标称姿态
│  ├─ builder.py             config → MJCF XML（确定性输出）
│  ├─ kinematics.py          双足正运动学 / 雅可比 / 站立高度反解（只依赖 numpy）
│  ├─ dino.py                ★ 恐龙参数源 + 生成器 + 站姿反解 + 静态校验
│  ├─ verifier.py            双足生成物与参数源的交叉校验
│  ├─ sim.py                 MuJoCo 相关入口（延迟导入，缺包时友好提示）
│  └─ __main__.py            CLI：build / build-dino / verify / verify-dino / view / simulate
│
├─ models/                   生成物，提交入库（没有 Python 也能直接导入 MuJoCo）
│  ├─ biped.xml              双足本体（8 body / 9 关节 / 9 执行器 / stand 关键帧）
│  ├─ scene.xml              include biped.xml + 地面 + 天空盒 + 灯光 + 两个机位 + 求解器设置
│  ├─ dino.xml               恐龙本体（14 body / 18 关节 / 18 执行器 / stand 关键帧 / cam_head 头部相机）
│  └─ dino_scene.xml         include dino.xml + 地面 + 灯光 + front / iso / ref 三个机位
│
├─ tests/                    pytest
│  ├─ test_config.py         双足参数自洽性
│  ├─ test_builder.py        XML 结构与确定性
│  ├─ test_kinematics.py     正运动学、对称性、雅可比
│  ├─ test_verifier.py       校验器（包含"故意改坏后能否抓到"）
│  ├─ test_committed_models.py  双足生成物必须与参数源同步
│  ├─ test_dino.py           恐龙参数自洽性、尺寸可调性、生成物同步
│  ├─ test_mujoco_load.py    双足模型真正加载 + 短时仿真（无 MuJoCo 时 skip）
│  └─ test_dino_mujoco.py    恐龙模型加载 / 站姿 / 新关节动作 / 头部相机 / 稳定性（无 MuJoCo 时 skip）
│
├─ docs/
│  ├─ model_design.md        双足模型设计说明：坐标、关节表、碰撞、假设
│  ├─ dino_model.md          恐龙模型设计说明 + 改尺寸/改姿态的工作流
│  ├─ preview_front.png      双足正视图（报告出图用）
│  ├─ preview_iso.png        双足 45° 视图
│  ├─ preview_dino_iso.png   恐龙 45° 视图
│  ├─ preview_dino_ref.png   恐龙参考视角（接近设计图的 3/4 视角）
│  └─ dino_vs_reference.png  恐龙模型与参考设计图的对照图
│
├─ ref/                      课程原始要求（EESM5800_project.md）
├─ build/                    测试临时目录与 pytest 缓存（git 忽略）
├─ environment.yml           conda 环境定义
├─ requirements.txt          pip 依赖
├─ pytest.ini
├─ README.md
└─ LICENSE
```

---

## 模型速查

**双足机型**：

```
pelvis ──waist_yaw──────────────────────→ torso      （腰部偏航）
   │
   ├──hip_roll──hip_pitch──→ thigh ──knee──→ shank ──ankle──→ foot   （左腿）
   └──hip_roll──hip_pitch──→ thigh ──knee──→ shank ──ankle──→ foot   （右腿）
```

* 俯仰关节轴统一 `(0,-1,0)`：**正角度 = 向前摆**；膝关节负角度 = 屈膝。
* 髋外展轴左 `(1,0,0)`、右 `(-1,0,0)`：**两腿正角度都是外展**，控制器不用翻号。
* 足底水平判据：`髋俯仰 + 膝 + 踝 = 0`。
* 标称站姿关键帧 `stand`：髋 `+0.06`、膝 `-0.12`、踝 `+0.06`，骨盆高 `0.1997 m`。

**恐龙机型**：

```
torso（浮动基座）
├─ head      neck_pitch（上下）→ neck_yaw（左右，轴 z）  + cam_head 头部前视相机
├─ arm_L/R   shoulder_pitch_L/R      单段小手臂
├─ thigh_L/R hip_pitch_L/R（主髋）→ hip_roll_L/R（内外翻，轴 x）→ hip_yaw_L/R（脚尖外旋，轴 z）
│  └─ shank  knee_L/R
│     └─ foot ankle_L/R              前趾深灰，和足底一起接触地面
└─ tail_1    tail1_yaw → tail_2 tail2_pitch → tail_3 tail3_yaw → tail_4 tail4_pitch
```

* 18 个关节 = 俯仰 11 + 偏航 5（头 1、脚尖 2、尾 2）+ 内外翻 2；两条腿的髋 / 脚尖 yaw
  互为镜像（**同一正角度 = 脚尖外旋**），躯干没有单独的腰 yaw。
* 足底水平判据：`base_pitch + 髋 + 膝 + 踝 = 0`，站姿里踝角由生成器自动反解。
* 标称站姿 `stand`：基座高 `0.1841 m`，站高 `0.307 m`；每个 link 的长度都在
  [`dino.py`](desktop_robot/dino.py) 的 `*_size` 里，改长度会自动带动下游关节。
* 头部相机 `cam_head` 挂在 `head` body 上（随头转动，相对头部下俯 12°、站姿净视线约下俯
  6.3°，fovy 75°），第一视角能看到前方地面，可直接用于观测 / 出图。

---

## 改参数的工作流

**双足**：

```powershell
# 1) 改 desktop_robot/config.py 的 DEFAULT_SPEC
# 2) 重新生成模型
python -m desktop_robot build
# 3) 校验（建议带 --dynamic）
python -m desktop_robot verify --dynamic
# 4) 测试：test_committed_models 会确认生成物已同步
python -m pytest
```

**恐龙**：同样是"先改参数源、再重新生成"，只是参数源在 `desktop_robot/dino.py` 的
`DEFAULT_DINO_SPEC`：

```python
from dataclasses import replace
from desktop_robot.dino import DEFAULT_DINO_SPEC, write_dino_models

spec = replace(DEFAULT_DINO_SPEC, thigh_size=(0.048, 0.040, 0.060))  # 大腿 50 → 60 mm
write_dino_models(spec)
```

```powershell
python -m desktop_robot build-dino
python -m desktop_robot verify-dino --dynamic
python -m pytest tests/test_dino.py tests/test_dino_mujoco.py
```

`models/*.xml` 虽然是生成物，但**提交进 git**——这样没有 Python 环境的人也能
双击打开模型。`tests/test_committed_models.py` 与 `tests/test_dino.py` 里的
"committed ... up to date" 用例保证生成物不会悄悄过期。

---

## git 约定

* `models/*.xml` 提交（生成物，但保证可用性）。
* `build/`、`.pytest_cache/`、`__pycache__/`、`.venv/` 忽略。
* 无二进制模型资产，仓库体积很小，不需要 Git LFS。
* 建议的提交粒度：参数改动 + 重新生成的 XML + 测试改动放在同一个 commit。
  提交前先跑 `python -m pytest`。

---

## 后续可以接的东西

这个工程原来只做到"模型 + 工具链"；下面这些已经在 `code/` 里接上了：

* 逆运动学与步态生成（`kinematics.py` 已有雅可比，可以直接接；恐龙的站姿反解在 `dino.py`）
* Gymnasium 环境封装 → `code/dino_wbc/env.py`（三阶段：lift / standup / walk）
* 强化学习训练 → `code/train/`（PPO，课程学习，checkpoint 导出给运行端）
* 恐龙的侧向自由度：髋内外翻、髋 / 脚尖 yaw、头 yaw、尾 yaw 已加；还缺腰 yaw，
  补上后原地转身会更自然（现在要转身得靠两腿的 yaw 配合步态）
* 硬件设计定稿后按实测尺寸替换 `*_size`，重新生成模型即可

---

## 附：仓库自带的 skill-creator

本项目还自带一个用于生成 skill 的工具：[skill-creator](.agents/skills/skill-creator/SKILL.md)。

- 调用：在对话里说「帮我生成一个 skill」，或输入 `/skill-creator`。
- 存放：新 skill 写到 `.agents/skills/<name>/SKILL.md`（DSH 只扫描根目录下一层）。
- 校验：`node .agents/skills/skill-creator/scripts/validate-skill.mjs <skill-directory>`。

# 桌面恐龙机器人（dino）模型设计说明

本文档描述 `models/dino.xml` 里那台**霸王龙式双足桌面机器人**的几何、关节、约定与简化假设。
**所有数值都来自 [`desktop_robot/dino.py`](../desktop_robot/dino.py) 的 `DEFAULT_DINO_SPEC`**；
本文档只做解释，改参数请改 `dino.py` 再重新生成，不要手改 XML。

![参考视角预览](preview_dino_ref.png)

![模型与参考图对照（左：设计图；右：models/dino_scene.xml 的 ref 机位）](dino_vs_reference.png)

---

## 1. 一句话规格

| 项目 | 数值 |
| --- | --- |
| 站立总高（脚底→头顶） | ≈ 0.353 m |
| 总质量（密度 800 kg/m³ 估算） | ≈ 2.38 kg |
| 驱动关节 | 11 个（全部矢状面俯仰）+ 6 自由度浮动基座 |
| `nq / nv / nu` | 18 / 17 / 11 |
| body 数量 | 12 |
| 几何 | 16 个 box（含 4 个前趾）+ 11 个视觉轴承圆柱，无外部网格 |
| 站立基座高度 | 0.2301 m（由正运动学反解，保证足底刚好贴地） |

## 2. 与参考图的编号对应

| 编号 | 关节 | 编号 | 关节 |
| --- | --- | --- | --- |
| ① | `neck_pitch`（颈） | ⑦ | `knee_R`（右膝） |
| ② | `shoulder_pitch_L`（左肩） | ⑧ | `ankle_L`（左踝） |
| ③ | `shoulder_pitch_R`（右肩） | ⑨ | `ankle_R`（右踝） |
| ④ | `hip_pitch_L`（左髋） | ⑩ | `tail1_pitch`（尾部第 1 节） |
| ⑤ | `hip_pitch_R`（右髋） | ⑪ | `tail2_pitch`（尾部第 2 节） |
| ⑥ | `knee_L`（左膝） | | |

## 3. 拓扑

```text
torso（浮动基座，freejoint: root）
├─ head          [neck_pitch]           ①
├─ arm_L         [shoulder_pitch_L]     ②
├─ arm_R         [shoulder_pitch_R]     ③
├─ thigh_L       [hip_pitch_L]          ④
│  └─ shank_L    [knee_L]               ⑥
│     └─ foot_L  [ankle_L]              ⑧
├─ thigh_R       [hip_pitch_R]          ⑤
│  └─ shank_R    [knee_R]               ⑦
│     └─ foot_R  [ankle_R]              ⑨
└─ tail_1        [tail1_pitch]          ⑩
   └─ tail_2     [tail2_pitch]          ⑪
```

图里没有腰关节，所以 **torso 直接做浮动基座**；标称站姿靠 freejoint 的四元数让躯干抬头。

## 4. 连杆尺寸（全尺寸，m）

| 连杆 | 形状 | 全尺寸 (x, y, z) | 相对父关节偏移 | 长度来源 |
| --- | --- | --- | --- | --- |
| torso | box | 0.150 × 0.095 × 0.085 | —（基座原点＝几何中心） | — |
| head | box | 0.100 × 0.072 × 0.055 | 颈在 torso (0.055, 0, 0.050)；头中心再 (+0.032, 0, +0.022) | 显式偏移 |
| arm_L/R | box | 0.045 × 0.032 × 0.075 | 肩在 torso (0.048, ±0.058, 0.018)；中心 (0, 0, −臂长/2) | `arm_size[2]` |
| thigh_L/R | box | 0.048 × 0.040 × 0.090 | 髋在 torso (−0.015, ±0.045, −0.030)；中心 (0, 0, −大腿长/2) | `thigh_size[2]` |
| shank_L/R | box | 0.044 × 0.036 × 0.075 | 膝在 thigh (0, 0, −大腿长)；中心 (0, 0, −小腿长/2) | `shank_size[2]` |
| foot_L/R | box | 0.085 × 0.055 × 0.032 | 踝在 shank (0, 0, −小腿长)；中心 (0.012, 0, −0.016) | 显式偏移 |
| toe_L/R ×2 | box | 0.028 × 0.018 × 0.026 | 贴住足前端与足底，`±toe_y` 左右各一 | `toe_size`、`toe_y` |
| tail_1 | box | 0.085 × 0.042 × 0.036 | 尾根在 torso (−0.068, 0, −0.020)；中心 (−尾 1 长/2, 0, 0) | `tail1_size[0]` |
| tail_2 | box | 0.080 × 0.036 × 0.030 | 尾 2 关节在 tail_1 (−尾 1 长, 0, 0)；中心 (−尾 2 长/2, 0, 0) | `tail2_size[0]` |

每个铰链处还有一个灰色圆柱（`cap_*`）：只做视觉、**零质量、不参与碰撞**，用来还原参考图里的轴承外壳。

## 5. 关节

| 关节 | 父 → 子 | 轴 | 范围 (rad) | kp / kv | 力矩上限 |
| --- | --- | --- | --- | --- | --- |
| `neck_pitch` | torso → head | (0, −1, 0) | ±0.70 | 25 / 1.0 | ±2.5 N·m |
| `shoulder_pitch_L/R` | torso → arm | (0, −1, 0) | ±1.20 | 25 / 1.0 | ±2.5 N·m |
| `hip_pitch_L/R` | torso → thigh | (0, −1, 0) | ±1.20 | 25 / 1.0 | ±2.5 N·m |
| `knee_L/R` | thigh → shank | (0, −1, 0) | [−1.60, 0.05] | 25 / 1.0 | ±2.5 N·m |
| `ankle_L/R` | shank → foot | (0, −1, 0) | [−0.60, 0.80] | 25 / 1.0 | ±2.0 N·m |
| `tail1_pitch` | torso → tail_1 | (0, −1, 0) | ±0.80 | 25 / 1.0 | ±2.5 N·m |
| `tail2_pitch` | tail_1 → tail_2 | (0, −1, 0) | ±0.80 | 25 / 1.0 | ±2.5 N·m |

符号约定与双足模型一致：**正角 = 子连杆向前摆**；对朝后的尾巴，正角表示**尾尖下压**。

## 6. 标称站姿

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `base_pitch` | 20° | 躯干抬头（freejoint 四元数，绕 (0, −1, 0)） |
| `stand_neck` | −0.25 | 头相对躯干低头，最终头约抬头 5.7° |
| `stand_shoulder` | −0.15 | 手臂自然下垂、略前摆 |
| `stand_hip` | −0.30 | 大腿基本竖直 |
| `stand_knee` | −0.10 | 轻微屈膝 |
| `stand_ankle` | **自动** | `−(base_pitch + stand_hip + stand_knee)`，保证足底水平 |
| `stand_tail1/2` | 0.20 / 0.12 | 尾巴向后下方垂 ≈ 31° → 38° |
| 基座高度 | 0.2301 m | 由 `solve_base_z()` 用足 + 前趾 24 个顶点反解，最低点贴地 |

## 7. 改尺寸 / 改姿态的正确姿势

**改某个连杆的长度**：直接改 `DinoSpec` 的 `*_size`，关节锚点在生成时按长度重算。

```python
from dataclasses import replace
from desktop_robot.dino import DEFAULT_DINO_SPEC, write_dino_models

spec = replace(
    DEFAULT_DINO_SPEC,
    thigh_size=(0.048, 0.040, 0.110),   # 大腿长 90 → 110 mm
    shank_size=(0.044, 0.036, 0.090),   # 小腿长 75 → 90 mm
    tail1_size=(0.100, 0.042, 0.036),   # 尾 1 长 85 → 100 mm
)
write_dino_models(spec)                 # 重新生成 models/dino.xml + dino_scene.xml
```

生成时自动联动的东西：

* 膝关节位置 = −大腿长，踝关节位置 = −小腿长；
* 臂/腿/尾的几何中心 = 各自长度的一半；
* 站立基座高度重新反解，保证脚底仍然贴地；
* 前趾始终贴住足前端与足底（`toe_center` 由 `foot_size` / `toe_size` 派生）。

想改「关节挂在躯干上的位置」就改 `neck_pos` / `shoulder_pos` / `hip_pos` / `tail1_pos`；
想改姿态就改 `base_pitch` / `stand_*`（踝角会自动反解，不用管）。

## 8. 生成 / 校验 / 查看

```powershell
conda activate E:\conda_envs\desktop-robot
cd D:\2026\HKUST\EESM-Robotic\desktop

python -m desktop_robot build-dino          # 生成 models/dino.xml + dino_scene.xml
python -m desktop_robot verify-dino --dynamic   # 静态 + MuJoCo 加载/站姿校验
python -m desktop_robot view --model dino_scene # 打开查看器
python -m desktop_robot simulate --model dino_scene --steps 1000
python -m pytest tests/test_dino.py tests/test_dino_mujoco.py
```

预览图由 `models/dino_scene.xml` 的 `iso` / `ref` 机位（后者是接近参考图的 3/4 视角、窄视场）用
`mujoco.Renderer` 出图，可以直接复用出报告图。

## 9. 简化假设

1. **完全平面机构**：11 个关节都是单轴俯仰，没有侧摆 / 偏航；因此不能原地转身、不能做侧向平衡。
   如果左右髋或肩在实际设计里是 2 自由度，需要在 `DinoSpec` 里加 roll 轴。
2. 图里没有腰关节，torso 直接做浮动基座；头/躯干/尾巴的俯仰都靠关节实现。
3. 手臂只有 1 段，没有肘/手；尾巴 2 节；足是方块 + 2 个深灰色前趾。
4. 关节轴承只是视觉圆柱（零质量、无碰撞），不模拟轴承间隙、摩擦和回差。
5. 密度取 800 kg/m³（含舵机、电池、结构件的等效平均密度），总质量 2.38 kg 是估计值。
6. 连杆之间不碰撞（只有足和地板接触），避免仿真里相邻零件互相弹开。
7. 尺寸来自参考图的像素比例，总高按桌面级取 0.35 m 量级；有实测尺寸时应直接替换 `*_size`。

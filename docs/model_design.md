# 桌面双足机器人模型设计说明

本文档描述 `models/biped.xml` 里那台机器人的几何、关节、约定与简化假设。
**所有数值都来自 [`desktop_robot/config.py`](../desktop_robot/config.py) 的 `DEFAULT_SPEC`**，
本文档只做解释；改参数请改配置再重新生成，不要手改 XML。

---

## 1. 一句话规格

| 项目 | 数值 |
| --- | --- |
| 站立总高 | ≈ 0.310 m |
| 总质量 | ≈ 1.510 kg |
| 自由度 | 9 个驱动关节 + 6 自由度浮动基座 |
| `nq / nv / nu` | 16 / 15 / 9 |
| body 数量 | 8 |
| 几何总数 | 8 个 box（无外部网格） |
| 站立时骨盆高 | 0.1997 m（由正运动学反解，保证足底刚好贴地） |

选这个体量是因为题目要求"桌面级"：能放在 1 m² 的桌面上、摔下来不会砸坏东西，
同时腿长又足够表达行走/蹲起等动作。

---

## 2. 坐标与符号约定

```
        +z  上
         |
         |    +x  前
         |   /
         | /
         +-------- +y  左
```

* 右手系，`x` 向前、`y` 向左、`z` 向上（MuJoCo 的惯用约定）。
* 单位：长度 m，角度 rad（`<compiler angle="radian"/>`）。
* **俯仰关节**（髋俯仰 / 膝 / 踝）的轴一律写成 `(0, -1, 0)`，于是

  > **正角度 = 子连杆向前摆**

  膝关节正角度是"伸展"，负角度是"屈膝"，所以标称站姿里膝角是 `-0.12`。
* **髋外展轴**左腿 `(1, 0, 0)`、右腿 `(-1, 0, 0)`，两腿都是**正角度 = 外展**。
  这样左右腿的控制器可以直接用同一个符号，不用翻号。
* **腰部偏航**轴 `(0, 0, 1)`，正角度 = 上身向左转。

---

## 3. 连杆几何

全部是 box 原始体。MJCF 里 `size` 是**半尺寸**，下表给的是**全尺寸**
（配置里也写全尺寸，方便和课程报告里的尺寸表对照）。

| 连杆 | 全尺寸 (x, y, z) | 相对父关节的偏移 | 质量 (密度 800 kg/m³) |
| --- | --- | --- | --- |
| `pelvis` 骨盆 | 0.090 × 0.120 × 0.040 | — （自由基座，原点即骨盆中心） | 0.346 kg |
| `torso` 腰部/上身 | 0.080 × 0.110 × 0.090 | 腰部关节在骨盆中心上方 +0.020；几何中心再上 +0.045 | 0.634 kg |
| `thigh_L/R` 大腿 | 0.040 × 0.040 × 0.080 | 髋在骨盆 (0, ±0.035, −0.020)；几何中心 z −0.040 | 0.102 kg ×2 |
| `shank_L/R` 小腿 | 0.036 × 0.036 × 0.080 | 膝在大腿下端 z −0.080；几何中心 z −0.040 | 0.083 kg ×2 |
| `foot_L/R` 足 | 0.100 × 0.050 × 0.020 | 踝在小腿下端 z −0.080；几何中心 (x +0.020, y 0, z −0.010) | 0.080 kg ×2 |

足部在 x 方向前移了 20 mm：踝关节落在足的中后部，站立时重心更容易落在支撑面上。

### 运动学链

```
pelvis ──hip_roll──   ┐
       ──hip_pitch──  ┴─→ thigh ──knee──→ shank ──ankle──→ foot
```

髋的两个关节**锚在同一点**（髋中心），在 MJCF 里放在同一个 `thigh` body 上按
`hip_roll → hip_pitch` 的顺序复合。这样做的好处是不产生"零质量的中间 body"
（MuJoCo 对零质量 body 会告警或给一个极小惯量），模型更干净。

---

## 4. 关节

| 关节 | 父 → 子 | 轴 | 范围 (rad) | kp | kv | 力矩上限 |
| --- | --- | --- | --- | --- | --- | --- |
| `root`（freejoint） | world → pelvis | — | 无限 | — | — | — |
| `waist_yaw` | pelvis → torso | (0, 0, 1) | ±1.570 | 25 | 1.0 | ±2.5 N·m |
| `hip_roll_L/R` | pelvis → thigh | (±1, 0, 0) | ±0.350 | 25 | 1.0 | ±2.0 N·m |
| `hip_pitch_L/R` | 同上 | (0, −1, 0) | [−0.900, 1.200] | 25 | 1.0 | ±2.5 N·m |
| `knee_L/R` | thigh → shank | (0, −1, 0) | [−1.600, 0.050] | 25 | 1.0 | ±2.5 N·m |
| `ankle_L/R` | shank → foot | (0, −1, 0) | [−0.600, 0.800] | 25 | 1.0 | ±2.0 N·m |

所有关节都带 `armature="0.005"`（模拟电机转子惯量）、`damping="0.05"`、
`frictionloss="0.02"`。这三个量对位置源仿真的稳定性影响很大，不要随手删掉。

**为什么只给 4 个自由度/腿**：髋外展 + 髋俯仰 + 膝 + 踝是最小的"能侧向平衡、
能走、能原地转身（配合腰部偏航）"的配置。再少一个自由度就只能在矢状面动，
一推就倒；再多就要加脚趾/腰俯仰，对第一版模型没有必要。

### 执行器

9 个 `position` 执行器，参数 `kp = 25`、`kv = 1.0`，并用
`inheritrange="1"` 让 `ctrlrange` 自动继承关节范围。

选位置源而不是力矩源，是因为参照的 Microduck 用的是 **RC 舵机**——舵机内部
本身就是位置闭环，`kp/kv` 相当于舵机的刚度/阻尼。后续要做力矩控制或者强化学习时，
把 `<position>` 换成 `<motor>` 再在控制器里换算即可，改动只在 `builder.py` 一处。

---

## 5. 碰撞设置

碰撞用位掩码分三类（MuJoCo 的判据是
`contype1 & conaffinity2` 或 `contype2 & conaffinity1` 非零）：

| 类别 | contype | conaffinity | 会撞到 |
| --- | --- | --- | --- |
| 地板 `floor`（在 scene.xml 里） | 1 | 1 | 双脚、躯干、骨盆 |
| 双脚 | 1 | 1 | 地板、另一只脚 |
| 其他连杆 | 2 | 1 | 只撞地板 |

效果：

* 脚能踩实地板，足底摩擦系数给到 1.2（橡胶感）；
* 摔倒时躯干/骨盆会实实在在地板上停住，不会穿模；
* 但连杆之间**不互相碰撞**——简单模型的常规做法，避免步态训练时接触点数量爆炸、
  也避免初始姿态里相邻连杆互相弹开。

如果后面要做"自碰撞感知"的步态，把其他连杆的 `contype` 改成 1 即可。

---

## 6. 怎么导入 MuJoCo

三种方式，按方便程度排：

```powershell
# 1) 用本工程的查看器（会直接跑物理）
conda activate E:\conda_envs\desktop-robot
python -m desktop_robot view

# 2) 用 MuJoCo 自带的 simulate 图形程序
python -m mujoco.simulate models/scene.xml

# 3) 代码里加载
python -c "import mujoco; m = mujoco.MjModel.from_xml_path('models/scene.xml'); print(m.nq, m.nu)"
```

`models/scene.xml` 通过 `<include file="biped.xml"/>` 引入机器人，另外带：

* 渐变天空盒与格子地面——都用 MuJoCo 的 `builtin` 纹理生成，
  **不引入任何外部图片文件**，仓库依然没有二进制资产；
* 两盏方向光，其中补光 `castshadow="false"`（两盏同时投影会让地面出现阴影摩尔纹）；
* 两个固定机位：`front`（正视，适合报告出图）和 `iso`（45°，适合看整体姿态）；
* `<visual><global offwidth="1280" offheight="960"/>`：离屏渲染缓冲已经开大，
  可以直接 `mujoco.Renderer` 出图或当强化学习观测用。

`models/biped.xml` 本身也是合法模型，可以单独加载（没有地面，机器人会往下掉——
这在做纯运动学/动力学分析时反而方便）。

---

## 7. 正运动学

实现见 [`desktop_robot/kinematics.py`](../desktop_robot/kinematics.py)，
只依赖 numpy，因此在没有 MuJoCo 的机器上也能算。

单腿链式关系（以左腿为例，全部相对骨盆坐标系）：

```
p_hip  = (0, +0.035, −0.020)
R_0    = R(roll_axis, q_roll) · R(pitch_axis, q_pitch)
p_knee = p_hip  + R_0 · (0, 0, −0.080)
R_1    = R_0 · R(pitch_axis, q_knee)
p_ankle= p_knee + R_1 · (0, 0, −0.080)
R_2    = R_1 · R(pitch_axis, q_ankle)
p_foot = p_ankle + R_2 · (0.020, 0, −0.010)
```

其中 `R(axis, θ)` 是 Rodrigues 旋转矩阵。

### 两条好用的结论

1. **足底水平判据**：三个俯仰关节同轴，所以
   `q_hip_pitch + q_knee + q_ankle = 0` ⟺ 足部坐标系与骨盆平行（足底水平）。
   标称站姿 `(0.06, −0.12, 0.06)` 正是按这条构造的，测试里有断言。
2. **站立骨盆高**：把两只足 box 的 8 个顶点都变换到世界系取最低点，
   令其为 0 反解骨盆高度（`solve_pelvis_z`）。用顶点而不是"足中心减半高"，
   是为了在足部有俯仰角时也正确。当前结果 0.1997 m。

### 雅可比

`numeric_jacobian()` 用中心差分给出足端位置对四个关节角的 3×4 雅可比。
后面做逆运动学、步态规划、或者分析关节速度需求时可以直接用。

---

## 8. 简化假设（写报告时要照抄这一段）

1. **不建头部和手臂**。题目只要求腰 + 双足；头/臂属于表达性交互的执行机构，
   确定交互方案后再加，避免现在就把自由度做复杂。
2. **连杆用 box，而不是带倒角的 CAD 实体**。惯量按均质 box 解析计算，
   与真实 3D 打印件的差异在 10% 量级，对选型和步态设计够用。
3. **密度取 800 kg/m³**（含舵机、电池、结构件的等效平均密度），
   所以总质量 1.51 kg 是估计值，实际要按 BOM 重新核算。
4. **关节是理想位置源**，没有建模舵机的死区、回差、带宽和温度漂移，
   力矩上限也只是按额定值截断。
5. **连杆之间不碰撞**（见第 5 节），足部按刚性 box 处理，没有橡胶垫的柔顺。
6. **没有脚趾关节**，踝关节到足底只有一个固定偏移。
7. 忽略线缆、装配公差、以及桌面倾斜等真实桌面环境因素。

这些假设对"能不能走"的定性结论影响不大，但对"能耗/续航/力矩余量"的定量结论
影响明显——如果报告要给出续航时间，需要按第 3 条重新核算质量。

---

## 9. 改参数的正确姿势

```powershell
# 1) 改 desktop_robot/config.py 里的 DEFAULT_SPEC（比如把腿加长）
# 2) 重新生成
python -m desktop_robot build
# 3) 校验
python -m desktop_robot verify --dynamic
# 4) 跑测试（其中 test_committed_models 会确认生成物已同步）
python -m pytest
```

`RobotSpec` 是 frozen dataclass，可以用 `dataclasses.replace()` 派生机型做对比实验：

```python
from dataclasses import replace
from desktop_robot.config import DEFAULT_SPEC
from desktop_robot.builder import write_models, MODELS_DIR

tall = replace(DEFAULT_SPEC, name="desktop_biped_tall", thigh_size=(0.04, 0.04, 0.10))
write_models(tall, MODELS_DIR / "variants")
```

**不要手改 `models/*.xml`**：`tests/test_committed_models.py` 会比对生成结果，
手改必然导致测试失败（这是故意的）。

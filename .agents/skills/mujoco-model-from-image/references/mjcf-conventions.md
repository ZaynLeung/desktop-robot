# MJCF 生成规范

结构图确认后，按这份规范写 XML。规范的目的只有一个：**生成出来的模型能一次加载成功，且和结构图逐项对得上。**

骨架直接从 `assets/model.template.xml` 复制，再按结构图改。

---

## 0. 文件拆分

输出两个文件：

| 文件 | 内容 |
| --- | --- |
| `<name>.xml` | 机器人本体：`<compiler>` + `<worldbody>` + `<actuator>` + `<keyframe>` |
| `scene.xml` | `<include file="<name>.xml"/>` + `<option>` + `<statistic>` + `<asset>` + `<visual>` + 自己的 `<worldbody>`（地面/灯光/相机） |

三条硬规则：

1. `<compiler>` **只能**出现在本体文件里，且只出现一次。场景文件再写一个会直接报重复错误。
2. `<option>`（时间步、积分器、重力）只放场景文件，本体保持纯净、可单独加载。
3. `include` 用相对文件名即可，MuJoCo 按「包含文件所在目录」解析；场景文件里可以再写一个 `<worldbody>`，多个 worldbody 会合并。

## 1. 单位与角度

`<compiler angle="radian" autolimits="true"/>`。长度一律米，角度一律弧度。
不要在模型里混用角度制——工程里所有表的数值都按 rad 写。

## 2. 坐标与轴向

x 前、y 左、z 上，右手系。轴向按用途固定：

| 用途 | 轴 | 符号含义 |
| --- | --- | --- |
| 俯仰（前后摆） | `(0, -1, 0)` | 正角 = 子连杆向前摆 |
| 外展/侧摆 | `(1, 0, 0)` | 左件正角 = 向外摆 |
| 左右镜像件 | 把轴的 x 分量取反，如 `(-1, 0, 0)` | 两件正角都表示「向外」，控制器不用翻号 |
| 偏航 | `(0, 0, 1)` | 正角 = 向左转 |

俯仰取 `-y` 而不是 `+y`，是为了让「正角=前摆」成立，读关节曲线时不会反直觉。

## 3. 命名

| 对象 | 规则 | 例 |
| --- | --- | --- |
| body | 部位名，左右加 `_L` / `_R` | `pelvis`、`thigh_L` |
| joint | `<部位>_<自由度>` | `hip_pitch_L`、`knee_L` |
| geom | `<body>_geom` | `thigh_geom_L`、`foot_geom_R` |
| actuator | `act_<关节名>` | `act_knee_L` |

全部名字在模型内唯一。左右对称件用同一套基名加后缀，不要发明 `left_hip` 这种第二种写法。

## 4. body 组织

- 自由基座：`<body name="base" pos="0 0 0"><freejoint name="root"/>`，几何就挂在这个 body 上。
- **同一点的多个 DOF 写在同一个 body 里**，按顺序排列（例如髋的 roll 与 pitch）。不要为每个 DOF 插一个空 body——零质量 body 会让 MuJoCo 告警或给一个极小惯量。
- 关节默认绕所在 body 的原点转；需要偏置时给 joint 加 `pos`，不要靠挪 body 来凑。

## 5. 关节参数

每个 hinge 都要写全：

```xml
<joint name="knee_L" type="hinge" axis="0 -1 0"
       range="-1.60 0.05" limited="true"
       armature="0.005" damping="0.05" frictionloss="0.02"/>
```

- `range` 必须 `lo < hi`，且结构图里的标称姿态要落在区间内。
- `armature` / `damping` / `frictionloss` 是位置控制稳定性的关键，别为了简洁省掉。

## 6. 执行器

```xml
<position name="act_knee_L" joint="knee_L" kp="25" kv="1"
          forcerange="-2.5 2.5" forcelimited="true"
          inheritrange="1" ctrllimited="true"/>
```

- 舵机式机构用 `<position>`，`kp`/`kv` 相当于舵机刚度和阻尼，`inheritrange="1"` 让 ctrl 范围继承关节范围。
- 明确要做力矩控制或强化学习时，改 `<motor>` 并在控制器里换算，其它位置不动。
- **每个驱动关节恰好一个执行器**，数量对不上是最常见的加载失败原因之一。

## 7. 碰撞掩码

用位掩码把「谁和谁碰」分开（判据：`contype1 & conaffinity2` 或 `contype2 & conaffinity1` 非零）：

| 类别 | contype | conaffinity | 实际撞到 |
| --- | --- | --- | --- |
| 地板 | 1 | 1 | 接触件、机身 |
| 接触件（足/轮/底盘） | 1 | 1 | 地板、彼此 |
| 其他连杆 | 2 | 1 | 只撞地板 |

这样脚踩得住地、摔倒时机身不会穿模，但连杆之间不自碰——避免仿真里零件互相弹开、接触点数量爆炸。
需要自碰撞时，把其他连杆的 contype 改成 1。

## 8. 视觉

- 每个 geom 都给 `rgba`：同类连杆同色，接触件用深色，方便在渲染图里辨认层级。
- 不引用外部贴图或 mesh；地面质感用 `builtin` 纹理（`gradient` / `checker`），仓库里不留二进制资产。
- 场景里放一两盏方向光 + 一个固定机位，命名成 `front` / `iso`，方便复用出图。

## 9. 关键帧

```xml
<keyframe>
  <key name="stand" qpos="..." ctrl="..."/>
</keyframe>
```

- `qpos` 长度 = `nq` = 7（自由基座）+ 关节数。**顺序 = body 深度优先遍历 + body 内 joint 的定义顺序**，写错顺序不会报错，但姿态会莫名其妙。
- 自由基座那 7 个值是 `x y z qw qx qy qz`，四元数要归一化。
- `ctrl` 长度 = 执行器数量，顺序 = actuator 定义顺序。
- 基座高度用正运动学算，不要目测：把接触件的 8 个角点变换到世界系取最低点，令其为 0。这样落地姿态才准。

## 10. 交付前自检

- [ ] 本体文件里 `<compiler>` 恰好一个；场景文件里一个都没有
- [ ] `nq` = 7 + 驱动关节数；`keyframe.qpos` 长度等于 `nq`
- [ ] `nu` = 驱动关节数；每个关节有且只有一个执行器
- [ ] 每个 hinge 都有 `range`，且 `lo < hi`
- [ ] body / joint / geom / actuator 名字无重复
- [ ] 接触件 `contype=1 conaffinity=1`，其他连杆 `2/1`
- [ ] 关键帧姿态下，接触件最低点 z ≈ 0
- [ ] 没有 mesh / 贴图外部引用（用户明确提供资产时除外）
- [ ] 模型里的尺寸、关节数、轴向与结构图完全一致（逐行对一遍表）

有 MuJoCo 时补一条：

```bash
python -c "import mujoco; m = mujoco.MjModel.from_xml_path('scene.xml'); print(m.nq, m.nv, m.nu)"
```

打印的数字应等于自检清单里算出来的 `nq` / `nv` / `nu`。

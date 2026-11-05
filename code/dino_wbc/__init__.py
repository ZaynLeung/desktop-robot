"""小恐龙 WBC（whole-body control）控制包。

分成四层，训练端与运行端只共享前两层：

``config`` / ``model_io``
    关节顺序、动作尺度、阶段参数、MuJoCo 模型与契约校验。

``state`` / ``observation`` / ``action`` / ``rewards`` / ``env`` / ``curriculum``
    本体感知状态量、80 维观测、动作映射、三阶段奖励与 Gymnasium 环境。
    训练端用这里的环境，运行端用这里的观测/动作构造，保证 obs 完全一致。

``controllers`` / ``policy_io``
    统一控制器接口；策略 checkpoint 的存取与契约哈希。

``cameras`` / ``render``
    头部第一视角 + 第三人称跟随的双画面渲染。

``train/`` 与 ``run/`` 是两个互不 import 的入口：
训练端依赖 stable-baselines3，运行端只依赖 numpy / mujoco / torch / opencv。
"""

__version__ = "0.1.0"

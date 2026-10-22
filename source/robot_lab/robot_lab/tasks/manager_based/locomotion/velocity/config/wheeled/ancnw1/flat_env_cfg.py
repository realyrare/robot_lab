# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ancnw1 —— 平地（Flat）速度跟踪任务的环境配置。

==================== 为什么要继承 Rough？ ====================
这里的继承 **不是** "平地 ⊂ 复杂地形" 的包含关系，而是 **配置复用** 关系：
平地任务和粗糙地形任务共用 16 个奖励项、全部观测/动作/随机化定义，
差别只有"地形相关的那几项"。

所以 Flat 的做法是：继承 Rough 的全部定义，然后**做减法**——
    1. 地形生成器关掉（terrain_type 改成纯平面）
    2. 高度扫描仪删掉（平地没有起伏，扫了也是全零）
    3. 地形难度课程删掉（没有难度可分）

这是 Isaac Lab 官方（Unitree Go2 等）的通用约定，好处是：
改奖励只需改一处（rough_env_cfg.py），两个任务自动同步，不会改漏。

==================== 用途 ====================
    - 快速验证机器人建模对不对（平地都站不住 → PD 增益/初始姿态有问题）
    - 训练基础步态，比 rough 快 3~5 倍（没有地形生成和射线扫描的开销）
    - 平地场景部署（仓库、车间）本身就只需要平地策略
"""

from isaaclab.utils import configclass

from .rough_env_cfg import AncnW1RoughEnvCfg


@configclass
class AncnW1FlatEnvCfg(AncnW1RoughEnvCfg):
    def __post_init__(self):
        # 先完整执行父类（Rough）的初始化，拿到全部奖励/观测/动作定义
        super().__post_init__()

        # ---- 奖励微调 ----
        # 平地没有地形起伏，基座高度传感器没有意义，直接关掉
        self.rewards.base_height_l2.params["sensor_cfg"] = None

        # ---- 地形：换成纯平面 ----
        self.scene.terrain.terrain_type = "plane"  # 用无限大平面，不再生成地形块
        self.scene.terrain.terrain_generator = None  # 删掉地形生成器（台阶/斜坡/随机块）

        # ---- 去掉高度扫描 ----
        # 平地扫描结果恒为 0，留着只是白白占用观测维度（还拖慢训练）
        self.scene.height_scanner = None  # 删掉传感器本体（省射线投射开销）
        self.observations.policy.height_scan = None  # 从策略观测里移除
        self.observations.critic.height_scan = None  # 从 Critic 观测里移除

        # ---- 去掉地形难度课程 ----
        # 没有难度分级，课程也就无从谈起
        self.curriculum.terrain_levels = None

        # 把 weight=0 的奖励项从配置里移除，减少无谓计算。
        # 只在最外层类（Flat 本身）执行，避免和父类的同名逻辑冲突。
        if self.__class__.__name__ == "AncnW1FlatEnvCfg":
            self.disable_zero_weight_rewards()

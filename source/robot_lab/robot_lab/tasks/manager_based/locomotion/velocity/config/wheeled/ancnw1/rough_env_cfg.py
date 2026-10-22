# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ancnw1 —— 粗糙地形（Rough）速度跟踪任务的环境配置。

这个文件是整个任务的"主配置"，定义了 MDP 的全部要素：
    动作空间、观测空间、奖励项、随机化事件、终止条件、课程学习、指令范围。

⚠️ 重要：平地（Flat）任务的配置 **继承本文件**（见 flat_env_cfg.py），
   它只是在本配置基础上把地形相关的内容关掉。
   所以即使只想训平地，这个文件也必须存在。

继承关系：
    LocomotionVelocityRoughEnvCfg（通用轮足 rough 基类）
        └── AncnW1RoughEnvCfg（本文件，注入 ancnw1 的关节名与调好的权重）
                └── AncnW1FlatEnvCfg（平地，关掉地形/高度扫描/课程）

以 M20 配置为模板创建，主要差异：
    M20   关节名：hipx（侧摆）/ hipy（大腿）/ knee（小腿）
    ancnw1 关节名：hip（侧摆）  / thigh（大腿）/ calf（小腿）
    两者的轮关节名完全一致（fl/fr/hl/hr_wheel_joint），base_link 名也一致。
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from robot_lab.assets.ancn import ANCNW1_CFG  # isort: skip


@configclass
class AncnW1ActionsCfg(ActionsCfg):
    """动作空间（Action specifications for the MDP）。

    轮足机器人有两类执行器，因此动作也是两段拼接的：
        1. joint_pos  → 12 个腿部关节的"目标角度偏移量"（位置伺服）
        2. joint_vel  → 4  个轮子的"目标转速"（速度伺服）

    这里的 joint_names=[""] 只是占位，真实关节名在下面
    AncnW1RoughEnvCfg.__post_init__ 里注入。
    """

    # 腿部：输出的是相对默认站姿的角度偏移（use_default_offset=True）
    # scale=0.25 表示网络输出 ±1 对应 ±0.25 rad 的角度变化
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    # 轮子：输出目标转速，scale=20.0 表示网络输出 ±1 对应 ±20 rad/s
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=20.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class AncnW1RewardsCfg(RewardsCfg):
    """额外新增的奖励项（M20 模板里没有的）。

    下面三项都是 weight=0.0（默认关闭），因为它们是"轮子专用"的惩罚项。
    保留在这里是为了方便调参时直接打开，不必改基类。
    weight=0 的项在 __post_init__ 末尾会被 disable_zero_weight_rewards() 自动移除，
    不会占用任何计算资源。
    """

    # 轮子转速惩罚：抑制轮子无意义地高速空转
    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    # 轮子角加速度惩罚：抑制轮子转速突变（抖轮）
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    # 轮子力矩惩罚：省电 / 保护轮电机
    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )


@configclass
class AncnW1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """ancnw1 粗糙地形环境配置。

    把通用的 rough 基类和 ancnw1 的专属动作/奖励组合起来，
    并在 __post_init__ 里注入 ancnw1 的关节名与调好的参数。
    """

    # 用上面定义的动作/奖励配置覆盖基类的
    actions: AncnW1ActionsCfg = AncnW1ActionsCfg()
    rewards: AncnW1RewardsCfg = AncnW1RewardsCfg()

    # -------------------- 机器人专属常量 --------------------
    # 基座连杆名：高度扫描仪、摔倒终止、基座高度奖励都挂在它上面
    base_link_name = "base_link"
    # 足端（轮子）连杆名正则：接触类奖励和"只允许轮子碰地"的判断用它
    foot_link_name = ".*_wheel"

    # fmt: off
    # 12 个腿部关节（4 条腿 × hip/thigh/calf）
    leg_joint_names = [
        "fl_hip_joint", "fl_thigh_joint", "fl_calf_joint",
        "fr_hip_joint", "fr_thigh_joint", "fr_calf_joint",
        "hl_hip_joint", "hl_thigh_joint", "hl_calf_joint",
        "hr_hip_joint", "hr_thigh_joint", "hr_calf_joint",
    ]
    # 4 个轮关节
    wheel_joint_names = [
        "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
    ]
    # 全部 16 个关节：观测（joint_pos / joint_vel）用
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # 先执行父类（通用 rough 环境）的初始化，再覆盖 ancnw1 专属的部分
        super().__post_init__()

        # ------------------------------场景 Scene------------------------------
        # 把通用基类里的占位机器人替换成 ancnw1 的资产配置
        # {ENV_REGEX_NS} 会在运行时展开成 /World/envs/env_0 之类，实现多环境并行
        self.scene.robot = ANCNW1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 高度扫描仪挂在基座上（用于观测前方地形起伏）
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------观测 Observations------------------------------
        # 轮子是连续转动的，角度没有物理意义（会一直累加），
        # 所以用 joint_pos_rel_without_wheel：观测里剔除轮子的绝对角度
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        # 观测缩放：让不同量纲的量进入网络时尺度接近
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # 不用真实的基座线速度（真机上拿不到，属于"作弊观测"），
        # 让策略从 IMU + 关节历史里自己推断
        self.observations.policy.base_lin_vel = None
        # [保持开启] 让策略能"看到"地形起伏。
        # 若设为 None，rough 训练会变成"闭眼走地形"，摔倒率明显更高。
        self.observations.policy.height_scan.scale = 1.0
        # 指定观测要读哪些关节
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------动作 Actions------------------------------
        # 侧摆关节（hip）活动范围小，给更小的动作幅度，避免侧摆动作过猛
        # 其余关节（thigh/calf）给 0.25
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        # 轮子速度动作的缩放：网络输出 ±1 → ±5 rad/s 的转速增量
        # （M20 用 20.0；ancnw1 轮电机更小型，先给保守的 5.0，跑通后可上调）
        self.actions.joint_vel.scale = 5.0
        # 动作裁剪范围（这里给得极宽，实际由电机限速约束）
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        # 关键绑定：位置动作 → 腿关节；速度动作 → 轮关节
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------随机化 Events（域随机化）------------------------------
        # 每次 reset 时随机化基座位姿和速度，提升策略鲁棒性（Sim2Real 的关键）
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        # 随机化基座质量（模拟负载变化）
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        # 随机化除基座外其他连杆的质量（负向前瞻：排除 base_link）
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        # 随机化质心位置 + 施加随机外力和力矩（抗扰动）
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------奖励 Rewards------------------------------
        # 说明：负权重 = 惩罚项（越接近 0 越好）；正权重 = 鼓励项（越大越好）
        # 权重为 0 的项会在文件末尾被自动移除，相当于"预留但不用"

        # ---- 通用 ----
        self.rewards.is_terminated.weight = 0  # 回合被终止的惩罚，通常不用（会鼓励苟活）

        # ---- 基座（Root）惩罚 ----
        self.rewards.lin_vel_z_l2.weight = -2.0  # 惩罚上下颠簸
        self.rewards.ang_vel_xy_l2.weight = -0.05  # 惩罚横滚/俯仰角速度（防翻车）
        self.rewards.flat_orientation_l2.weight = 0  # 惩罚机身倾斜（轮式一般不需要）
        self.rewards.base_height_l2.weight = 0  # 惩罚离地高度偏离目标（下面设了目标值，需要时打开）
        self.rewards.base_height_l2.params["target_height"] = 0.45  # 按 ancnw1 站立高度 0.537 取略低值
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0  # 惩罚基座线加速度（平滑性）
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # ---- 关节惩罚 ----
        self.rewards.joint_torques_l2.weight = -2.5e-5  # 腿部力矩过大
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0  # 轮子力矩过大（预留未启用）
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0  # 腿部关节速度过大（预留未启用）
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0  # 轮子转速过大（预留未启用）
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7  # 腿部关节加速度（抑制抖动）
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9  # 轮子角加速度（抑制抖轮）
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0  # 关节接近限位时重罚
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0  # 轮子接近限速（预留未启用）
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2e-5  # 功率消耗（省电）
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -2.0  # 收到"静止"指令时腿部还在乱动 → 罚
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = -1.0  # 偏离默认站姿 → 罚（保持姿态收敛）
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_vel_penalty.weight = 0  # 无指令时轮子还在转 → 罚（预留未启用）
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = -0.05  # 对角腿对称性（让步态更自然）
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["fl_(hip|thigh|calf).*", "hr_(hip|thigh|calf).*"],  # 左前 ↔ 右后
            ["fr_(hip|thigh|calf).*", "hl_(hip|thigh|calf).*"],  # 右前 ↔ 左后
        ]

        # ---- 动作平滑 ----
        self.rewards.action_rate_l2.weight = -0.01  # 相邻两步动作变化过大 → 罚（防抽搐）

        # ---- 接触 ----
        # 只允许轮子碰地：除轮子外任何部位碰到地面都罚（防摔倒/蹭地）
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        # 轮子接触力过大 → 罚（保护轮胎和减速器）
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # ---- 速度跟踪（任务主目标，权重最大）----
        self.rewards.track_lin_vel_xy_exp.weight = 3.0  # 线速度跟踪（前后 + 左右）
        self.rewards.track_ang_vel_z_exp.weight = 1.5  # 转向角速度跟踪

        # ---- 其他（足式常用，轮式默认关闭）----
        self.rewards.feet_air_time.weight = 0  # 抬腿腾空时间（轮式不需要）
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1  # 无指令时轮子保持接地（站稳别抬）
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0  # 轮子撞到竖直障碍
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0  # 轮子侧滑
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0  # 抬轮高度
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0  # 对角步态同步性
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
        self.rewards.upward.weight = 1.0  # 机身保持朝上（防止翻车，正向奖励）

        # 把上面所有 weight=0 的奖励项从配置里移除，避免无谓计算。
        # 只在最外层类（Rough 本身）执行，保证子类（Flat）继承时不被重复处理。
        if self.__class__.__name__ == "AncnW1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------终止条件 Terminations------------------------------
        # [已开启] 摔倒判定：基座（base_link）受到的接触力超过阈值即判定摔倒并结束回合。
        # 这是 rough 训练里最重要的信号之一 —— 没有它，机器人会拖着摔坏的姿态继续"混时间"。
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.illegal_contact.params["threshold"] = 1.0

        # ------------------------------课程学习 Curriculums------------------------------
        # 速度指令的课程（先给慢指令，学好了再给快指令）默认关闭
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        # 注意：地形难度课程（terrain_levels）保持开启 ——
        # 机器人跑得好就自动升级到更难的地形，跑得差就降级。平地任务里会被关掉。

        # ------------------------------指令 Commands------------------------------
        # 速度指令的采样范围（默认继承自基类：x/y/z 各 ±1.0）
        # ⚠️ ancnw1 是轮式机器人，**横向（y）移动在物理上做不到**（轮子无法侧移），
        #    保留 ±1.0 会导致：速度跟踪误差永远降不下来、奖励被不可能完成的任务浪费。
        #    建议平地任务坚定改成 0，如需启用请取消下面第二行注释。
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        # ⚠️ ancnw1 是轮式机器人，侧向移动在物理上做不到（轮子只能沿轮面滚动）。
        #    300 轮冒烟实测：error_vel_xy ≈ 0.63，其中约 0.58 来自永远追不上的横移指令，
        #    既污染指标又白扣奖励（track 奖励被不可能完成的任务压着拿不满）。
        #    因此把横向指令范围锁死为 0：指令库里不再出现横移。
        #    代价：策略不会横移（本来也不会）。若哪天换成万向轮机型，把这行注释掉即可。
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)

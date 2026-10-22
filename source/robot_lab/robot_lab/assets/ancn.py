# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ancnw1 轮足机器人资产配置（Asset Configuration）。

本文件只负责一件事：把 URDF 模型包装成 Isaac Lab 能加载的 ``ArticulationCfg``。
任务侧（环境/奖励/动作）不在这里定义，见 config/wheeled/ancnw1/ 目录。

============================== 关节构成 ==============================
ancnw1 是"轮足"机器人：四条腿，每条腿末端带一个驱动轮，共 16 个关节。

    命名规则：<位置>_<部位>_joint
        位置：fl=左前  fr=右前  hl=左后  hr=右后
        部位：hip=侧摆关节  thigh=大腿  calf=小腿  wheel=驱动轮

    每条腿：fl_hip_joint / fl_thigh_joint / fl_calf_joint / fl_wheel_joint （以此类推）
    合计  ：12 个腿部关节 + 4 个轮关节 = 16 个

============================== URDF 实测参数 ==============================
以下数值直接从 data/Robots/ancn/ancnw1_description/urdf/ancnw1.urdf 读出，
不是抄别家机器人的，改模型时记得同步这里。

    关节      旋转轴      功能        下限      上限     力矩(N·m)  速度(rad/s)
    hip       (1, 0, 0)   侧摆        -0.611    0.436     113        15.708
    thigh     (0,-1, 0)   大腿俯仰    -2.583    2.286     113        15.708
    calf      (0,-1, 0)   小腿俯仰    -2.792    2.809     113        15.708
    wheel     (0,-1, 0)   轮子转动    连续无限制            19.3       94.248

    轮半径：0.11 m
    基座碰撞盒：0.62 × 0.30 × 0.17 m

    零位姿态下（所有关节角为 0）轮子轴心相对基座的高度：
        大腿段下降 0.19800 m + 小腿段下降 0.22936 m = 0.42736 m
    加上轮半径 0.11 m → 基座站立高度约 0.537 m

⚠️ 注意：URDF 里**没有** PD 增益（stiffness / damping）。
   下面的刚度/阻尼暂用 M20 的值，属于"能跑起来"的量级，
   正式训练前必须按真机电机的实际响应调，否则会出现抖动或站不住。
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

ANCNW1_CFG = ArticulationCfg(
    # ============ spawn：如何把 URDF 变成仿真里的机器人 ============
    spawn=sim_utils.UrdfFileCfg(
        # fix_base=False → 基座是浮动的（会摔倒），训练 locomotion 必须浮动
        fix_base=False,
        # 把 URDF 里的 fixed 关节合并进父连杆，减少刚体数量、提升仿真速度
        # 注意：合并后这些关节不再作为独立关节出现，关节数会少于 URDF 里的总数
        merge_fixed_joints=True,
        # 不把圆柱碰撞体替换成胶囊体（胶囊更稳定但会改变接触形状，这里保持原始模型）
        replace_cylinders_with_capsules=False,
        # URDF 文件路径。ISAACLAB_ASSETS_DATA_DIR 指向 source/robot_lab/data
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/ancn/ancnw1_description/urdf/ancnw1.urdf",
        # 开启接触传感器：undesired_contacts（非轮子碰地）等奖励项依赖它
        activate_contact_sensors=True,
        # ---- 刚体物理属性 ----
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # 正常受重力
            retain_accelerations=False,  # 不保留上一帧加速度（省一点计算）
            linear_damping=0.0,  # 不额外加线性阻尼，避免掩盖真实的动力学
            angular_damping=0.0,  # 不额外加角阻尼
            max_linear_velocity=1000.0,  # 速度上限，防止数值爆炸
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,  # 穿透修正速度上限，防止穿地时"弹飞"
        ),
        # ---- 关节求解器属性 ----
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 关闭自碰撞：训练时腿之间偶尔会穿模，开启会导致大量误判摔倒
            enabled_self_collisions=False,
            solver_position_iteration_count=4,  # 位置求解迭代次数（越大越硬，越慢）
            solver_velocity_iteration_count=0,  # 速度求解迭代次数
        ),
        # ---- URDF 转换器：把 URDF 的 joint_drive 转成 Isaac 的驱动 ----
        # 这里把增益强制归零，真正的 PD 参数由下面的 actuators 决定，
        # 避免 URDF 里的默认值和我们的执行器配置打架
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    # ============ init_state：每次 reset 时的初始状态 ============
    init_state=ArticulationCfg.InitialStateCfg(
        # 基座初始位置 (x, y, z)。z 按上面推算的 0.537 m 取 0.54，
        # 稍微留一点余量避免初始瞬间轮子穿地。实际站姿调好后再回来改这个值。
        pos=(0.0, 0.0, 0.54),
        # 初始关节角（正则匹配关节名）。
        # 这里统一用 0（URDF 零位）：此时大腿"向下后"、小腿"向下前"，
        # 已经是一个自然的屈膝站姿，轮心正好在基座下方 0.427 m 处。
        # 若想让机器人蹲得更低/更高，改这里（注意 thigh/calf 正负号含义见文件头注释）。
        joint_pos={
            ".*_hip_joint": 0.0,  # 侧摆归零：四条腿不外张
            ".*_thigh_joint": 0.0,  # 大腿：0 = URDF 零位
            ".*_calf_joint": 0.0,  # 小腿：0 = URDF 零位
            ".*_wheel_joint": 0.0,  # 轮子角度无所谓，速度才是控制量
        },
        # 初始关节速度全为 0：开局静止站立
        joint_vel={".*": 0.0},
    ),
    # ============ 软限位 ============
    # 实际可用行程 = URDF 限位 × 0.9，留 10% 余量防止关节撞到硬限位产生冲击
    soft_joint_pos_limit_factor=0.9,
    # ============ actuators：电机模型 ============
    # 分两组：腿部关节用位置控制（JointPositionAction），轮子用速度控制（JointVelocityAction）
    actuators={
        # ---- 腿部：12 个关节，位置伺服 ----
        "joint": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=113.0,  # ← URDF 实测最大力矩
            saturation_effort=113.0,  # 力矩饱和值，通常与 effort_limit 一致
            velocity_limit=15.708,  # ← URDF 实测最大转速 (rad/s)
            stiffness=80.0,  # ⚠️ PD 的 P：URDF 里没有，暂用 M20 值，需按真机调
            damping=2.0,  # ⚠️ PD 的 D：同上。太小腿会抖，太大动作会迟钝
            friction=0.0,  # 电机静摩擦，先设 0
        ),
        # ---- 轮子：4 个关节，速度伺服 ----
        "wheel": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=19.3,  # ← URDF 实测轮电机力矩
            saturation_effort=19.3,
            velocity_limit=94.248,  # ← URDF 实测轮最大转速（约 15 转/秒 × 0.11 m ≈ 10 m/s）
            stiffness=0.0,  # 速度控制模式下 P 必须为 0，否则会和速度指令冲突
            damping=0.6,  # ⚠️ 暂用 M20 值，需按真机调
            friction=0.0,
        ),
    },
)

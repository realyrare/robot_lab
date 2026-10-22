# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ancnw1 的 rsl_rl PPO 超参配置 —— 训练时真正生效的那一套。

文件里的两个类分别对应两个任务：
    AncnW1RoughPPORunnerCfg → RobotLab-Isaac-Velocity-Rough-AncnW1-v0
    AncnW1FlatPPORunnerCfg  → RobotLab-Isaac-Velocity-Flat-AncnW1-v0

参数沿用 M20（同为轮足，结构一致），可直接在下面调。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AncnW1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """粗糙地形任务的 PPO 配置。"""

    # ---------------- 训练节奏 ----------------
    # 每次迭代每个环境采样多少步。24 × 4096 环境 = 98,304 个样本/迭代
    num_steps_per_env = 24
    # 总训练迭代次数。rough 任务样本需求大，默认 20000
    # （实测 M20 rough 约 2.3 秒/迭代 → 20000 轮 ≈ 13 小时）
    max_iterations = 20000
    # 每隔多少迭代存一次权重（model_100.pt、model_200.pt ...），断电也不丢进度
    save_interval = 100

    # 日志目录名：logs/rsl_rl/<experiment_name>/<时间戳>_<run_name>
    experiment_name = "ancnw1_rough"

    # ---------------- 网络结构 ----------------
    policy = RslRlPpoActorCriticCfg(
        # 初始探索噪声。1.0 = 起步时动作随机性拉满，随训练自动收敛
        init_noise_std=1.0,
        # 观测归一化。关闭：因为环境侧已经做了缩放（scale），再归一化反而多余
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # Actor / Critic 都是 3 层 MLP，逐层收窄
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",  # ELU 在连续控制任务里通常比 ReLU 稳
    )

    # ---------------- PPO 算法参数 ----------------
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # 价值函数损失在总损失里的占比
        use_clipped_value_loss=True,  # 裁剪价值损失，防止价值网络更新过猛
        clip_param=0.2,  # PPO 的 ε，限制每次策略更新的幅度
        entropy_coef=0.01,  # 熵奖励系数：鼓励继续探索，太大会不收敛
        num_learning_epochs=5,  # 每批数据重复训练 5 轮
        num_mini_batches=4,  # 每批数据切成 4 个小批量
        learning_rate=1.0e-3,  # 初始学习率
        schedule="adaptive",  # 按 KL 散度自适应调整学习率（配合下面的 desired_kl）
        gamma=0.99,  # 折扣因子
        lam=0.95,  # GAE 的 λ
        desired_kl=0.01,  # 目标 KL 散度：超过就降学习率，低于就升
        max_grad_norm=1.0,  # 梯度裁剪，防梯度爆炸
    )


@configclass
class AncnW1FlatPPORunnerCfg(AncnW1RoughPPORunnerCfg):
    """平地任务的 PPO 配置：完全继承 Rough，只改轮数和日志目录名。

    平地没有地形生成和射线扫描，训练快很多，所以轮数可以给得更少。
    """

    def __post_init__(self):
        super().__post_init__()

        # 平地任务收敛更快，5000 轮通常足够
        # （实测 M20 flat 约 0.44 秒/迭代 → 5000 轮 ≈ 37 分钟）
        self.max_iterations = 5000
        # 平地产物单独一个目录，和 rough 的结果互不覆盖
        self.experiment_name = "ancnw1_flat"

# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ancnw1 的 cusrl 训练器配置。

cusrl 是本仓库支持的另一套强化学习库（rsl_rl 之外的选择）。
**用 rsl_rl 训练时不会用到本文件**，它只在 `--agent cusrl` 时才会被加载。

保留它是为了和 M20 的配置结构保持一致（任务注册里也引用了它）。
如果环境里没装 cusrl，只要不显式使用，就不会触发 import，不影响 rsl_rl 训练。
"""

from dataclasses import dataclass

import cusrl
from cusrl.environment.isaaclab import TrainerCfg


@dataclass
class AncnW1RoughTrainerCfg(TrainerCfg):
    """粗糙地形任务的 cusrl 训练器配置（参数与 rsl_rl 版对齐）。"""

    max_iterations = 20000
    save_interval = 100
    experiment_name = "ancnw1_rough"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        # ---- Actor：输出动作分布 ----
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
            distribution_factory=cusrl.NormalDist.Factory(),  # 高斯分布采样动作
        ),
        # ---- Critic：估计状态价值 ----
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        # ---- 训练流程钩子（等价于 rsl_rl 里的各项配置）----
        hooks=[
            cusrl.hook.ValueComputation(),  # 计算 V(s)
            cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),  # GAE
            cusrl.hook.AdvantageNormalization(),  # 优势归一化
            cusrl.hook.ValueLoss(),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),  # PPO 裁剪目标
            cusrl.hook.EntropyLoss(weight=0.008),  # 熵正则
            cusrl.hook.GradientClipping(max_grad_norm=1.0),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.01),  # 自适应学习率
        ],
    )


@dataclass
class AncnW1FlatTrainerCfg(AncnW1RoughTrainerCfg):
    """平地任务的 cusrl 训练器配置：只改轮数和日志目录名。"""

    max_iterations = 5000
    experiment_name = "ancnw1_flat"

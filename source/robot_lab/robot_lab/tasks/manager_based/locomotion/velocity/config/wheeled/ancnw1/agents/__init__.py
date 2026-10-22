# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ancnw1 的算法配置包（PPO 超参等）。

⚠️ 这里**故意不写任何 import**，和 M20 保持一致。

原因：cusrl_ppo_cfg.py 里有 `import cusrl`。
如果本文件显式导入它，而环境里没装 cusrl，
整个 agents 包就会导入失败 → 连 rsl_rl 的任务注册也会一起挂掉。

保持空导入后，两个配置文件都只在被真正用到时才由 Isaac Lab 按需加载，
互不干扰（用 rsl_rl 训练时完全不会碰 cusrl）。
"""

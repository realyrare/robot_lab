# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ancnw1 任务的 Gym 注册入口。

这是让 `--task=RobotLab-Isaac-Velocity-Flat-AncnW1-v0` 能被识别的地方。

注册机制说明：
    robot_lab/tasks/__init__.py 里的 import_packages() 会**自动扫描并导入**
    本包下的所有子目录，所以只要这个 __init__.py 存在就会被加载，
    无需去任何"总注册表"里手动登记 —— 新建机器人目录即可，零侵入。

每个 gym.register 做三件事：
    id                        → 命令行 --task 用的任务名
    env_cfg_entry_point       → 环境配置类（决定地形、奖励、观测）
    rsl_rl_cfg_entry_point    → PPO 超参配置类（你要用的算法配置）
    cusrl_cfg_entry_point     → 另一套 RL 库（cusrl）的配置，不用可忽略
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ---- 平地任务 ----
gym.register(
    id="RobotLab-Isaac-Velocity-Flat-AncnW1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AncnW1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AncnW1FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:AncnW1FlatTrainerCfg",
    },
)

# ---- 粗糙地形任务 ----
gym.register(
    id="RobotLab-Isaac-Velocity-Rough-AncnW1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AncnW1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AncnW1RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:AncnW1RoughTrainerCfg",
    },
)

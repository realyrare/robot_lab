# robot_lab 入门学习指南（M20 轮足专题 · 详细版）

> 专为 C# 工程师定制 · 以云深处 M20 轮足机器人为切入点
> 前置要求：会编程（任何语言均可），不需要机器人学基础
>
> 本文档三大目标：
> ① 完整的代码结构地图（每个文件夹是干嘛的）
> ② 分阶段学习路线（做轮足机器狗需要哪些知识点、哪些代码要精读）
> ③ 自定义机器人接入的完整规划（改哪些文件、按什么顺序）

---

## 目录

- [第一部分：项目原理速览](#第一部分项目原理速览)
- [第二部分：完整代码结构地图](#第二部分完整代码结构地图)
- [第三部分：分阶段学习路线与知识点](#第三部分分阶段学习路线与知识点)
- [第四部分：调试导向的代码精读地图](#第四部分调试导向的代码精读地图)
- [第五部分：自定义机器人接入完整规划](#第五部分自定义机器人接入完整规划)
- [第六部分：参数实验手册](#第六部分参数实验手册)
- [第七部分：概念速查表](#第七部分概念速查表)
- [第八部分：常见报错急救](#第八部分常见报错急救)

---

# 第一部分：项目原理速览

一句话：**用 NVIDIA 的物理仿真器（Isaac Sim），同时模拟几千台机器人，让神经网络通过 millions 次试错，自己学会走路。**

## 1.1 三个角色

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   仿真环境        │     │   神经网络(策略)    │     │   RL 训练算法     │
│  Isaac Sim       │ ←→  │   PyTorch          │ ←→  │   PPO (RSL-RL)  │
│  物理引擎+渲染     │     │   输入:传感器数据    │     │   决定怎么更新网络 │
│  4096个M20并行    │     │   输出:16个关节指令  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## 1.2 训练与播放（C# 类比）

**训练 (train.py)**：4096 个虚拟 M20 并行试错（无画面 `--headless`），产出 `.pt` 权重文件（类比序列化的模型对象）。耗时数小时到数天。

**播放 (play.py)**：加载 `.pt`，1 个机器人 + 图形界面，每帧"传感器→网络→关节指令"。耗时几秒就能看到效果。

**为什么播放必不可少**：训练曲线只说明"分数在涨"，分数高 ≠ 走得好（可能学会原地抖动作弊）。眼见为实。

## 1.3 版本依赖（版本不匹配是最常见报错来源）

| robot_lab | IsaacLab | Isaac Sim |
|---|---|---|
| main 分支 | main | 4.5 / 5.0 / 5.1 |
| v2.3.2 | v2.3.2 | 4.5 / 5.0 / 5.1 |

---

# 第二部分：完整代码结构地图

## 2.1 仓库根目录

```
h:\CompanyCode\robot_lab\
├── docker/                  # Docker 部署配置（docker-compose.yaml + .env.base）
│                            #   用途：在容器里跑训练，免本地装环境的痛苦（进阶用）
├── docs/                    # 仅 24 张机器人截图（README 表格引用），无文档
├── scripts/                 # ★ 所有可执行脚本的入口（训练/播放/工具），见 2.2
├── source/                  # ★ 核心源码包，见 2.3
├── pyproject.toml           # 代码规范配置（ruff/pyright/pytest），不影响运行逻辑
├── VERSION                  # 版本号文件
├── CONTRIBUTORS.md / LICENSE / README.md
└── LEARNING_GUIDE_M20.md    # 本文档
```

## 2.2 scripts/ —— 你敲的每条命令都在这

```
scripts/
├── reinforcement_learning/
│   ├── rl_utils.py          # ★ 公共训练工具函数（train/play 都 import 它）
│   ├── rsl_rl/              # RSL-RL 框架（主力，用这个）
│   │   ├── train.py         #   训练入口
│   │   ├── play.py          #   播放入口
│   │   ├── play_cs.py       #   播放变体（cusrl 用的）
│   │   └── cli_args.py      #   命令行参数定义（--num_envs/--load_run 等）
│   ├── cusrl/               # CusRL 框架（实验性，先不管）
│   └── skrl/                # skrl 框架（AMP 舞蹈任务用，先不管）
└── tools/
    ├── list_envs.py         # ★ 列出所有已注册环境名（装完环境第一个跑的）
    ├── zero_agent.py        # ★ 零动作测试：机器人趴着不动，验证模型加载 OK
    ├── random_agent.py      #   随机动作测试：乱动，验证动作空间 OK
    ├── convert_urdf.py      #   URDF→USD 转换工具（自定义机器人时会用到）
    ├── convert_mjcf.py      #   MJCF(MuJoCo)→URDF 转换
    ├── clean_trash.py       #   清理 /tmp 下的 USD 缓存垃圾
    └── beyondmimic/         #   动作模仿数据处理（G1 专用，先不管）
```

**学习期只需关注带 ★ 的 6 个脚本。**

## 2.3 source/ —— 核心包（注意三层嵌套）

```
source/
└── robot_lab/                    ← 第 1 层：pip 安装边界（pip install -e source/robot_lab）
    ├── pyproject.toml / setup.py #   安装配置，不用动
    ├── config/
    │   └── extension.toml        #   扩展元数据（名称/版本），被 assets/__init__.py 读取
    ├── data/                     # ★ 机器人 3D 资源库（573 个文件）
    │   └── Robots/
    │       ├── unitree/          #   每厂商一个目录：URDF + STL/DAE 网格文件
    │       ├── deeprobotics/     #   ★ M20 在这：m20_description/urdf/m20.urdf + 34 个 STL
    │       ├── agibot/ booster/ ddt/ fftai/ magiclab/
    │       ├── openloong/ roboparty/ robotera/ zsibot/
    │       └── （共 11 个厂商目录）
    └── robot_lab/                ← 第 2、3 层：Python 包本体（源码真正在这里！）
        ├── __init__.py           #   包入口：触发所有 Gym 环境注册
        ├── ui_extension_example.py  # Omniverse UI 扩展示例（无关，忽略）
        ├── assets/               # ★ 机器人"身体定义"（见 2.4）
        └── tasks/                # ★ 训练环境定义（见 2.5）
```

> **为什么找文件总迷路**：源码不在根目录，而在 `source\robot_lab\robot_lab\`。IDE 快捷定位：**Ctrl+Shift+F 全局搜类名**，比如搜 `DeeproboticsM20`。

## 2.4 assets/ —— 机器人"身体定义"（12 个文件）

```
robot_lab/assets/
├── __init__.py          # ★ 定义路径变量 ISAACLAB_ASSETS_DATA_DIR（指向 data/ 目录）
│                        #   所有 URDF 路径都通过它拼接，自定义机器人必经之地
├── unitree.py           # Unitree 全家桶（A1/Go2/B2/G1/H1...的 ArticulationCfg）
├── deeprobotics.py      # ★ M20 和 Lite3 的定义（你的重点文件）
├── agibot.py  booster.py  ddtrobot.py  fftai.py
├── magiclab.py  openloong.py  roboparty.py
├── robotera.py  zsibot.py  # 其余 9 家厂商，结构与 deeprobotics.py 完全一致
```

**每个文件的内容结构都一样**（以 M20 为例）：

```python
DEEPROBOTICS_M20_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(...),          # ① 指向 data/ 里的 URDF + 物理属性
    init_state=InitialStateCfg(...), # ② 初始位姿和关节角（内膝式的特征就在这）
    actuators={...},                 # ③ 电机模型：力矩/速度上限、PD 增益
)
```

## 2.5 tasks/ —— 训练环境定义（本项目最核心的目录）

```
robot_lab/tasks/
├── __init__.py              # ★ 自动扫描注册机制：
│                            #   import_packages() 递归 import 所有子包的 __init__.py，
│                            #   从而触发所有 gym.register()。新环境不用手动登记
├── direct/                  # 【Direct 风格】环境逻辑全部手写在一个类里
│   └── g1_amp/              #   只有 G1 AMP 舞蹈任务用了这种（实验性，忽略）
└── manager_based/           # ★ 【Manager 风格】本项目主力（推荐，也是 M20 用的）
    ├── beyondmimic/         #   G1 动作模仿任务（实验性，忽略）
    │   ├── tracking_env_cfg.py
    │   ├── mdp/  config/
    └── locomotion/          # ★★ 运动控制任务
        └── velocity/        # ★★★ 速度跟踪任务 —— 全项目 90% 的精华在这
            ├── velocity_env_cfg.py   # 公共模板（700 行，所有机器人共享）
            ├── mdp/                  # MDP 组件实现（奖励/观测/事件/指令/课程）
            └── config/               # 各机器人的专属配置
                ├── quadruped/        #   四足 8 种：anymal_d, unitree_a1/b2/go2...
                ├── wheeled/          #   ★ 轮足 6 种：deeprobotics_m20 在这
                ├── humanoid/         #   人形 10 种
                └── others/           #   实验（A1 倒立等）
```

## 2.6 velocity/mdp/ —— 游戏规则的实现代码（精读对象）

```
velocity/mdp/
├── __init__.py      # ★ import 链关键：先 import isaaclab.envs.mdp 全部，
│                    #   再 import isaaclab_tasks 的 locomotion mdp，
│                    #   最后 import 本目录 5 个模块
│                    #   → 意味着配置里写的 mdp.xxx 可能定义在上游！
│                    #   找不到函数定义时，顺着这条链往上游搜
├── commands.py      # 速度指令采样器 UniformThresholdVelocityCommand
│                    #   特色：检测"坑地形"时只允许前进（防掉坑）
├── curriculums.py   # 课程学习：跟踪奖励达标 80% 才扩大速度指令范围
├── events.py        # 域随机化事件的扩展实现
├── observations.py  # 自定义观测项
├── rewards.py       # ★ 奖励函数实现（几十个，调参时的数学依据）
├── utils.py         # 工具函数（地形判断等）
└── symmetry/        # 对称性数据增强（进阶，可让策略学得更对称）
    └── anymal.py
```

## 2.7 config/wheeled/deeprobotics_m20/ —— M20 专属配置（重点中的重点）

```
config/wheeled/deeprobotics_m20/
├── __init__.py          # ① 环境注册表：gym.register() 把任务名映射到配置类
├── rough_env_cfg.py     # ② ★ M20 核心配置：关节命名、混合动作空间、奖励权重
├── flat_env_cfg.py      # ③ 平地版（继承 ②，关掉地形/高度扫描/地形课程）
└── agents/              # ④ RL 算法超参数
    ├── __init__.py
    ├── rsl_rl_ppo_cfg.py        # ★ PPO 超参 + 网络结构（[512,256,128]）
    └── cusrl_ppo_cfg.py         # CusRL 框架的对应配置
```

## 2.8 一次训练的调用链全景（串起所有文件夹）

```
你敲的命令
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0
   │
   ├─ scripts/.../train.py     解析参数 → 启动 Isaac Sim
   │    └─ rl_utils.py         公共流程：建环境→包装→建 runner→训练循环
   │
   ├─ gym.make(任务名)
   │    └─ tasks/__init__.py   扫描注册 → 找到 config/wheeled/deeprobotics_m20/__init__.py
   │         └─ gym.register   拿到 env_cfg_entry_point → DeeproboticsM20FlatEnvCfg
   │
   ├─ 实例化环境配置类
   │    ├─ flat_env_cfg.py     → rough_env_cfg.py → velocity_env_cfg.py（三层继承）
   │    │       └─ 引用 assets/deeprobotics.py 的 DEEPROBOTICS_M20_CFG
   │    │             └─ 加载 data/Robots/deeprobotics/ 下的 URDF+STL
   │    │       └─ 奖励/观测函数指向 mdp/ 目录（或上游 isaaclab）
   │    └─ ManagerBasedRLEnv   按 7 个 Manager 组装环境
   │
   └─ rsl_rl OnPolicyRunner    按 agents/rsl_rl_ppo_cfg.py 的超参训练
        └─ 产出 logs/<实验名>/<run名>/model_*.pt
```

**看懂这张图，你就理解了整个项目。**

---

# 第三部分：分阶段学习路线与知识点

## 3.0 做一台轮足机器狗，需要的知识全景

按"必须掌握 → 边做边学 → 可以不懂"分层：

### 🔴 必须掌握（不学走不动路）

| 知识点 | 需要掌握到什么程度 | 学习材料 |
|---|---|---|
| **Python 基础** | 语法、装饰器、类型标注、dataclass | 你有 C# 底子，2 周够 |
| **MDP 概念** | 什么是状态/动作/奖励/回合 | 任意 RL 入门课前 2 讲 |
| **奖励函数设计** | 会读 `exp(-误差²/σ²)`，理解权重正负含义 | 本项目 rewards.py 就是教材 |
| **PD 控制** | `力矩 = K·位置误差 + D·速度误差`，对应 stiffness/damping | 一篇博客的量 |
| **URDF 结构** | 连杆 link / 关节 joint / 关节限位 / 惯量 | 看着 M20 的 URDF 对照学 |
| **轮足控制范式** | 腿=位置控制 + 轮=速度控制的混合动作空间 | M20 的 ActionsCfg 就是答案 |

### 🟠 边做边学（遇到再补）

| 知识点 | 什么时候会用到 |
|---|---|
| PPO 算法细节 | 训练不收敛、想调 learning_rate 时 |
| 四元数/旋转矩阵 | 看观测代码、调姿态奖励时 |
| 域随机化 | 准备 sim-to-real 时 |
| 课程学习 | 从 Flat 转 Rough 地形时 |
| 张量广播（PyTorch） | 读 rewards.py 的向量化代码时 |
| 关节限位/软限位 | 机器人疯狂抽搐时 |

### 🟡 可以先不懂（不影响入门）

- PPO 数学推导、GAE、KL 散度
- BeyondMimic / AMP 等模仿学习
- 对称性数据增强、蒸馏训练
- 多 GPU 分布式训练

## 3.1 六个学习阶段（兼职节奏，每周 8~10 小时）

### 阶段 0：环境搭建（1~2 周）

```
目标：list_envs.py 能打印环境列表
内容：装 Isaac Sim 5.1 → IsaacLab 2.3.2 → pip install -e source/robot_lab
检验：python scripts/tools/list_envs.py 出现 RobotLab-Isaac-... 字样
```

### 阶段 1：跑通 M20 闭环（1~2 周）★ 第一个里程碑

```
目标：完整体验 训练→曲线→播放 三步
命令：
  python scripts/tools/zero_agent.py --task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0
  python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0 --headless --num_envs 1024
  tensorboard --logdir=logs          # 另开终端，看 Episode Reward 是否上升
  python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0
同步阅读：本文档第一、二部分 + 2.8 调用链
```

### 阶段 2：精读 5 个文件（2~3 周）★ 理解的关键期

按此顺序读（先具体后抽象）：

| 顺序 | 文件 | 精读重点 | 跳过什么 |
|---|---|---|---|
| 1 | `config/wheeled/deeprobotics_m20/rough_env_cfg.py` | 关节命名列表、ActionsCfg 混合动作、每个奖励权重 | 注释掉的部分 |
| 2 | `assets/deeprobotics.py` 的 M20 段 | init_state（内膝式关节角）、actuators 的 PD 参数 | Lite3 配置 |
| 3 | `config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py` | 网络结构 [512,256,128]、学习率、迭代数 | — |
| 4 | `velocity_env_cfg.py` | 7 个 Manager 的分工、ObservationsCfg 的 8 个观测项、EventCfg 的随机化 | 每个奖励的细节 |
| 5 | `mdp/rewards.py` | track_lin_vel_xy_exp、joint_mirror、GaitReward 的数学 | 一次性全读完，用到哪个读哪个 |

**读代码方法（C# 工程师提醒）**：不要从第一行顺序读到最后一行。先看类结构（documentSymbol 大纲），再挑关键方法读。RL 代码是"配置+数学公式"的组合，不是业务逻辑流。

### 阶段 3：改参数实验（2~3 周）★ 建立手感

见第六部分实验手册。核心训练：**改一个参数 → 训练 → TensorBoard 对比 → 播放验证 → 解释因果**。

### 阶段 4：Rough 地形 + 调参实战（3~4 周）

```
目标：让 M20 在粗糙地形上稳定走
新知识点：
  - 课程学习 terrain_levels_vel：地形难度随表现递增
  - command_levels：速度指令范围逐步扩大（curriculums.py）
  - height_scan 观测：高度扫描怎么帮机器人"看见"地形
调试重点：地形太难→全摔；太简单→学不会越障
```

### 阶段 5：自定义机器人接入（4~8 周）

按第五部分的规划走。这是把前面所有学习的综合应用。

---

# 第四部分：调试导向的代码精读地图

**这一部分回答一个问题：出了某种症状，该去哪个文件看什么。**

## 4.1 症状 → 文件定位表（日常调试贴在手边）

| 症状 | 首先检查的文件 | 具体看什么 |
|---|---|---|
| 机器人一出生就穿地/悬空 | `assets/<robot>.py` | `init_state.pos` 的 z 高度是否匹配腿长 |
| 出生就疯抖、抽搐 | `assets/<robot>.py` | `stiffness/damping` 比例；URDF 惯量是否合理 |
| 某几个关节不动 | `rough_env_cfg.py` | `joint_names` 正则是否漏了关节；assets 里 `joint_names_expr` 是否匹配 |
| 动作方向反了/左右颠倒 | `rough_env_cfg.py` | 关节命名顺序（preserve_order=True 时顺序敏感）|
| 学会用轮子滑行作弊 | `rough_env_cfg.py` | `wheel_vel_penalty`、`stand_still` 奖励权重 |
| 速度跟踪不上指令 | `rough_env_cfg.py` + `velocity_env_cfg.py` | `track_lin_vel_xy_exp` 权重是否够大；指令范围是否超出能力 |
| 走路姿态难看（乱晃） | `rough_env_cfg.py` | `ang_vel_xy_l2`、`flat_orientation_l2`、`action_rate_l2` 权重 |
| 上台阶必摔 | `velocity_env_cfg.py` | 地形难度曲线、height_scan 分辨率、feet 相关奖励 |
| 训练前期涨后期崩 | `agents/rsl_rl_ppo_cfg.py` | learning_rate（adaptive 模式看 desired_kl）、entropy_coef |
| 显存爆 | 命令行 + `velocity_env_cfg.py` | `--num_envs` 减小；height_scanner 分辨率/范围 |
| 环境加载直接报错 | `config/<robot>/__init__.py` | 任务名拼写；entry_point 路径是否正确 |
| 找不到某 mdp 函数定义 | `velocity/mdp/__init__.py` | import 链：先本目录，再 isaaclab_tasks，再 isaaclab |

## 4.2 奖励权重调参口诀（轮足专属）

M20 的奖励分四类，调参时的思考顺序：

```
1. 任务奖励（正分，让机器人"想做对的事"）
   track_lin_vel_xy_exp  速度跟踪 —— 最核心，权重通常 1.5~3.0
   track_ang_vel_z_exp   转向跟踪 —— 通常 1.0~1.5
   upward                保持直立 —— 轮足容易忽略，建议开

2. 惩罚项（负分，让机器人"别做坏事"）
   action_rate_l2        动作突变 —— 必开，-0.01 量级
   joint_torques_l2      力矩过大 —— 真机安全的保险丝
   undesired_contacts    不该碰的地方碰了 —— 摔倒预警

3. 轮足专属（防作弊）
   wheel_vel_penalty     站立时轮子不能转（防滑行作弊）
   stand_still           无指令时关节回默认位（防乱动）

4. 姿态美化（后期再加）
   flat_orientation_l2 / feet_air_time / feet_slide ...
   —— 先让机器人走起来，再让走得好看
```

**调参铁律：一次只改一个权重；改动幅度 ×2 或 ÷2；每个实验起 `--run_name`。**

## 4.3 TensorBoard 看图指南

| 曲线 | 健康形态 | 病态形态及病因 |
|---|---|---|
| `Episode Reward` | 前期震荡上升，后期平稳 | 一直贴 0 → 环境配置问题；暴涨后跳水 → 奖励作弊或学习率过大 |
| `Policy/mean_noise_std` | 从 1.0 缓慢下降到 0.3~0.5 | 不降 → 探索过剩；骤降到 0 → 陷入局部最优 |
| `Episode Length` | 逐渐增长到 20s（满长） | 一直很短 → 机器人总在摔倒被终止 |
| 分项奖励（Reward/xxx） | 各分项同步增长 | 某项为 0 → 该奖励条件从未满足（检查参数） |

---

# 第五部分：自定义机器人接入完整规划

以"接入你自己的轮足机器狗"为目标，**8 步流程，每步有检验标准**。

## 5.0 总览：需要新建/修改的文件清单

```
需要新建（6 个文件）：
  data/Robots/<你的品牌>/                        ← URDF + 网格文件
  source/.../assets/<你的品牌>.py                 ← ArticulationCfg
  source/.../config/wheeled/<你的型号>/__init__.py       ← 环境注册
  source/.../config/wheeled/<你的型号>/rough_env_cfg.py  ← 核心配置
  source/.../config/wheeled/<你的型号>/flat_env_cfg.py   ← 平地配置
  source/.../config/wheeled/<你的型号>/agents/rsl_rl_ppo_cfg.py

需要修改（0 个！）：
  —— 不需要动任何现有文件。tasks/__init__.py 会自动扫描注册新包
```

## 5.1 第 1 步：准备 URDF（机械组提供，或从 CAD 导出）

**质量检查清单**（URDF 问题会在仿真里变成玄学故障）：

- [ ] 所有关节 `type="revolute"`（轮子是 `continuous"`），限位 `limit` 齐全
- [ ] 关节命名规范统一（建议模仿 M20：`fl_hipx_joint` / `fl_wheel_joint` 风格）
- [ ] 每个连杆有合理的 `inertial`（质量 + 惯量张量，**不能是 0 或默认值**）
- [ ] 网格文件路径正确，单位是米（CAD 导出常犯 mm 单位错误）
- [ ] **collision 碰撞体**用简化网格（不要拿视觉网格当碰撞体，仿真会慢 10 倍）
- [ ] base_link 在机身几何中心，z 轴向上

工具链：如果只有 SolidWorks/URDF 上游是 MJCF，用 `scripts/tools/convert_urdf.py` / `convert_mjcf.py` 转换。

**检验**：把 URDF 丢进 Isaac Sim / 任意 URDF 查看器，模型完整、关节能拖动。

## 5.2 第 2 步：放置资源文件

```
data/Robots/<你的品牌>/
└── <你的型号>_description/
    ├── urdf/<你的型号>.urdf
    └── meshes/ *.STL
```

照抄 `data/Robots/deeprobotics/m20_description/` 的目录组织。

## 5.3 第 3 步：写 ArticulationCfg（`assets/<你的品牌>.py`）

**直接复制 `deeprobotics.py` 的 M20 段落改**，逐项替换：

```python
from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/<你的品牌>/<型号>/urdf/<型号>.urdf",
        # rigid_props / articulation_props 段基本照抄 M20
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, Z),          # ★ Z = 站立高度。怎么定：腿完全伸直的高度 × 0.75
        joint_pos={
            # ★ 内膝式参考 M20 的镜像模式：
            #   前腿 hipy=-0.6 / knee=+1.0
            #   后腿 hipy=+0.6 / knee=-1.0
            #   （膝朝内折，膝盖初始不锁死、留 20°~40° 弯曲缓冲）
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ★ 填真实电机参数（问硬件组要规格书）：
        "joint": DCMotorCfg(
            joint_names_expr=[".*hipy_joint", ".*knee_joint"],  # 匹配你的命名
            effort_limit=实际峰值力矩,      # 宁可保守，别虚标
            velocity_limit=实际最大转速,
            stiffness=80~150,              # 腿越重刚度越大；初始值可先抄同类机器人
            damping=1~3,
        ),
        "wheel": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=..., velocity_limit=...,   # 轮子速度按 轮径反算
            stiffness=0.0, damping=0.6,             # 轮子不做位置控制
        ),
    },
)
```

然后在 `assets/__init__.py` 末尾加一行 `from . import <你的品牌>`（这是唯一需要修改的现有文件，可选）。

**检验**：跑 zero_agent，机器人以正确姿态趴在地面，不穿地、不弹飞、不自爆抖动。

> **这一步是整个流程中最容易出玄学问题的一步**（URDF 惯量错误、初始高度不对、PD 增益不匹配），预留 1~2 周调试。zero_agent 阶段一定要让机器人"安静地趴着"，任何抖动都要在此步解决。

## 5.4 第 4 步：写环境配置（`config/wheeled/<你的型号>/rough_env_cfg.py`）

复制 M20 的 `rough_env_cfg.py`，改四处：

```python
@configclass
class MyRobotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # ① 关节命名 —— ★改成你的 URDF 实际关节名，一个都不能漏
    leg_joint_names = [...]    # 12 个腿关节
    wheel_joint_names = [...]  # 4 个轮关节

    def __post_init__(self):
        super().__post_init__()

        # ② 场景：换成你的机器人
        self.scene.robot = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ③ 动作空间：确定 joint_pos（腿）和 joint_vel（轮）的正则匹配
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ④ 奖励权重：轮足起步配方（可直接抄，跑通后再调）
        self.rewards.track_lin_vel_xy_exp.weight = 2.0   # 速度跟踪
        self.rewards.track_ang_vel_z_exp.weight = 1.0    # 转向跟踪
        self.rewards.upward.weight = 1.0                 # 直立
        self.rewards.action_rate_l2.weight = -0.01       # 平滑
        self.rewards.joint_torques_l2.weight = -2e-5     # 省力
        self.rewards.wheel_vel_penalty.weight = -0.5     # 防轮子作弊
        self.rewards.undesired_contacts.weight = -1.0    # 防摔
        # ... 身体部位相关的 body_names 要按你的 URDF 改（foot_link_name 等）
```

**⚠️ 最容易翻车的点：所有 `body_names` / `joint_names` 正则必须和你的 URDF 命名完全对得上。** 建议写个小脚本打印 URDF 全部关节/连杆名，逐一核对。

## 5.5 第 5 步：写平地配置（`flat_env_cfg.py`）

照抄 M20 的，只改类名。逻辑固定：关地形、关高度扫描、关地形课程。

## 5.6 第 6 步：写算法配置（`agents/rsl_rl_ppo_cfg.py`）

照抄 M20 的，改三处：

```python
class MyRobotFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "<你的型号>_flat"     # 日志目录名
    max_iterations = 5000                    # Flat 平地 5000 轮起步
    policy = ...([512, 256, 128]...)         # 网络结构先不改
```

## 5.7 第 7 步：注册环境（`__init__.py`）

照抄 M20 的 `__init__.py`，替换任务名和类名：

```python
gym.register(
    id="RobotLab-Isaac-Velocity-Flat-MyRobot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MyRobotFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotFlatPPORunnerCfg",
    },
)
```

**检验**：`python scripts/tools/list_envs.py` 能看到你的任务名。

## 5.8 第 8 步：训练调试循环（核心阶段，2~6 周）

```
循环：
  zero_agent      → 姿态对不对？抖不抖？        【有问题回第 3 步】
  random_agent    → 动作空间映射对不对？          【有问题回第 4 步】
  train (Flat)    → 奖励曲线涨不涨？             【不涨看第四部分调参表】
  play            → 走得像不像样？               【作弊/难看 → 调奖励】
  ↓ 满意后
  train (Rough)   → 上台阶/越障能力              【加课程学习难度】
  域随机化加码     → 为 sim-to-real 做准备
```

**时间预算参考**（一个新轮足机器人从 URDF 到 Flat 平地能走）：

| 步骤 | 顺利 | 不顺利 |
|---|---|---|
| URDF 检查 + 资源放置 | 2 天 | 1 周（CAD 问题） |
| ArticulationCfg 调到趴稳 | 3 天 | 2 周（惯量/增益玄学） |
| 环境配置 + 注册跑通 | 1 天 | 3 天 |
| Flat 训练出能走的策略 | 3 天 | 2 周（奖励调参） |
| **合计** | **~2 周** | **~6 周** |

---

# 第六部分：参数实验手册

每个实验：改一处 → `--run_name=实验名` 训练 → TensorBoard 对比 → 播放验证。

| # | 改什么 | 位置 | 预期现象 | 学到什么 |
|---|---|---|---|---|
| 1 | `joint_pos.scale` 0.25→0.5 | rough_env_cfg.py | 动作更猛，学得快但粗暴 | 动作空间设计 |
| 2 | `track_lin_vel_xy_exp.weight` 2.0→6.0 | rough_env_cfg.py | 拼命追速度，姿态变差 | 奖励之间的挤压效应 |
| 3 | `action_rate_l2.weight` -0.01→0 | rough_env_cfg.py | 动作高频抖动 | 平滑惩罚的价值 |
| 4 | `wheel_vel_penalty.weight` →0 | rough_env_cfg.py | 站立时用轮子滑行作弊 | 轮足防作弊机制 |
| 5 | `num_envs` 4096→512 | 命令行 | 收敛变慢但不影响上限 | 采样量 vs 速度 |
| 6 | 随机化范围 (0.5,2.0)→(0.9,1.1) | velocity_env_cfg.py 事件 | 收敛更快更脆 | 域随机化的意义 |
| 7 | `max_iterations` 5000→15000 | agents cfg | 后期还在缓慢提升 | 训练时长的边际效应 |

---

# 第七部分：概念速查表

| 术语 | 白话解释 |
|---|---|
| **RL / 强化学习** | 让程序通过"试错 + 打分"自己学会做事 |
| **策略 (Policy)** | 神经网络：输入传感器数据，输出关节指令 |
| **PPO** | 最常用 RL 算法。核心思想：每次只小幅更新网络，防止学崩 |
| **奖励 (Reward)** | 每一步的得分，训练目标 = 长期总分最大 |
| **观测 (Observation)** | 网络输入：IMU、关节角/速度、上一步动作、地形高度图 |
| **动作 (Action)** | 网络输出：16 个关节目标值（12 腿位置 + 4 轮速度） |
| **回合 (Episode)** | 从站起到摔倒/超时的完整经历，本项目 20 秒 |
| **Manager** | IsaacLab 的模块化管理者：场景/指令/动作/观测/事件/奖励/课程 各一个 |
| **configclass** | IsaacLab 的配置类装饰器（≈ C# 的 record + 特性） |
| **`__post_init__`** | 配置类初始化钩子（≈ C# 构造函数末尾），子类靠它覆盖父类参数 |
| **域随机化** | 训练时故意随机改质量/摩擦/电机参数，换取真实世界的鲁棒性 |
| **课程学习 (Curriculum)** | 先简单后困难：地形难度、指令范围随表现递增 |
| **sim-to-real** | 仿真训的策略迁移到真机 |
| **URDF** | 机器人的 XML 描述：连杆、质量、关节轴向和限位 |
| **PD 控制** | `力矩 = K·位置误差 + D·速度误差`，对应 stiffness / damping |
| **内膝式** | 前后腿膝盖朝内对折（M20 特征），折叠紧凑、越障好，小腿易干涉 |
| **decimation** | 物理步:策略步 = 4:1，即物理 200Hz、策略 50Hz |
| ** ArticulationCfg** | 机器人身体总配置（模型路径+初始姿态+电机参数） |
| **gym.register** | 把任务名映射到配置类（≈ DI 容器注册） |

---

# 第八部分：常见报错急救

| 症状 | 原因 | 解法 |
|---|---|---|
| `CUDA out of memory` | 显存不够 | `--num_envs 1024` 或更小；降低 height_scanner 分辨率 |
| 找不到 task | 环境没注册 | 跑 `list_envs.py` 核对名字拼写；确认 `__init__.py` 没语法错 |
| RSL-RL 版本报错退出 | 依赖不匹配 | `pip install rsl-rl-lib==3.0.1` |
| Isaac Sim 黑屏/崩溃 | 驱动旧 | 更新 NVIDIA 驱动到最新 |
| 机器人加载后穿地 | 初始 z 高度错 | 改 assets 里 `init_state.pos` |
| 机器人高频疯抖 | PD 增益错 / 惯量错 | 降 stiffness、查 URDF inertial |
| 训练奖励恒为 0 | 某奖励条件永不满足 / 环境配置坏 | TensorBoard 看分项奖励，定位为 0 的项查参数 |
| 训练崩了（先涨后崩） | 学习率过大 / 奖励作弊 | 看 4.3 的曲线诊断表 |
| `/tmp` 暴涨 | USD 缓存堆积 | `python scripts/tools/clean_trash.py` |
| Windows 莫名崩溃 | 平台兼容 | 优先 WSL2 / Ubuntu 双系统 |

---

## 附：文档使用建议

1. **第二部分（结构地图）打印贴手边**，前两个月你会反复查
2. **第四部分（调试地图）在第一次训练失败时再精读**，那时才有代入感
3. **第五部分（自定义机器人）在完成阶段 3 后再开始**，顺序不能乱
4. 学习中产生的疑问，随时回到第七部分速查表对照

---

# 第九部分：ArticulationCfg 逐行精讲

ArticulationCfg 是机器人在仿真里的"全部身份声明"——质量分布、关节限位、电机模型、初始姿态全在这里。**M20 的 16 个关节能稳定训练出来，80% 取决于这个文件写得对不对**。下面逐行讲。

## 9.0 完整代码（M20 实测版）

```python
# 来自 assets/deeprobotics.py 第 65-121 行
DEEPROBOTICS_M20_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(                   # ① 加载 URDF
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/deeprobotics/m20_description/urdf/m20.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(...)
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(...)
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(...)
    ),
    init_state=ArticulationCfg.InitialStateCfg(    # ② 初始姿态
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*hipx_joint": 0.0,
            "f[l,r]_hipy_joint": -0.6,
            "h[l,r]_hipy_joint": 0.6,
            "f[l,r]_knee_joint": 1.0,
            "h[l,r]_knee_joint": -1.0,
            ".*wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,                # ③ 软限位
    actuators={                                     # ④ 电机模型
        "joint": DCMotorCfg(joint_names_expr=[".*hipx_joint", ".*hipy_joint", ".*knee_joint"],
                            effort_limit=76.4, saturation_effort=76.4,
                            velocity_limit=22.4, stiffness=80.0, damping=2.0, friction=0.0),
        "wheel": DCMotorCfg(joint_names_expr=[".*_wheel_joint"],
                            effort_limit=21.6, saturation_effort=21.6,
                            velocity_limit=79.3, stiffness=0.0, damping=0.6, friction=0.0),
    },
)
```

下面把每个字段拆开讲。

## 9.1 spawn = UrdfFileCfg —— "从哪加载模型 + 物理属性"

| 字段 | 含义 | M20 值 | 怎么定 |
|---|---|---|---|
| `fix_base` | 机器人的 base 是否固定 | False | 移动机器人永远 False |
| `merge_fixed_joints` | 是否把无运动的关节合并掉 | True | 通常 True，让图结构更简单 |
| `replace_cylinders_with_capsules` | 圆柱网格换胶囊体 | False | 除非视觉与碰撞冲突才开 |
| `asset_path` | URDF 绝对路径 | 通过 `ISAACLAB_ASSETS_DATA_DIR` 拼 | 你的 URDF 放对路径即可 |
| `activate_contact_sensors` | 开启接触力传感器 | True | 几乎永远 True（轮足需要检测脚着地） |

### 9.1.1 rigid_props（刚体物理属性）

```python
RigidBodyPropertiesCfg(
    disable_gravity=False,             # 是否关重力（调试用，训练必须 False）
    retain_accelerations=False,        # 是否保存加速度历史（通常不要，浪费内存）
    linear_damping=0.0,                # 线速度阻尼（空气阻力近似，仿真里一般 0）
    angular_damping=0.0,               # 角速度阻尼（同上）
    max_linear_velocity=1000.0,        # 物理上限，几乎用不到，给个超大值
    max_angular_velocity=1000.0,       # 同上
    max_depenetration_velocity=1.0,    # ★ 穿透恢复速度上限，乱填会导致卡死
)
```

**C# 类比**：这就是 `RigidBody` 的 `material` 配置类。

### 9.1.2 articulation_props（关节树求解器属性）

```python
ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,             # ★ 关键：要不要机器人撞自己
    solver_position_iteration_count=4,         # 位置求解迭代次数（4 起步，复杂接触时提到 8~16）
    solver_velocity_iteration_count=0,         # 速度求解迭代次数（设 0 让位置求解接管）
)
```

`enabled_self_collisions=False` 对轮足机器人**强烈建议关掉**——开了之后大腿、小腿之间会持续自碰报警告，性能暴跌。M20 关掉是基于实战经验。

### 9.1.3 joint_drive（URDF 里的默认驱动器，必须"先关后开"）

```python
UrdfConverterCfg.JointDriveCfg(
    gains=PDGainsCfg(stiffness=0, damping=0)   # URDF 里 <transmission> 的默认驱动，强制设为 0
)
```

**为什么 stiffer=0, damping=0？** 因为我们要在下面 `actuators={}` 里**重新定义**电机增益，URDF 自带的不能让它生效——否则两套 PD 参数打架，机器人行为完全不可预测。**这是一个隐藏坑，新手几乎必踩**。

## 9.2 init_state —— "出生姿态"

```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.52),          # ★ base_link 的世界坐标 (x, y, z)
    joint_pos={...},                # 每个关节的初始角度
    joint_vel={".*": 0.0},         # 所有关节初始速度 = 0
)
```

### 9.2.1 pos[2]（z 高度）怎么定？

**经验值**：`站立高度 ≈ 整机完全站直时的最低脚部离地 × 1.0`

```
太高 → 出生瞬间掉下来，仿真起步浪费几秒
太低 → 出生即穿地，PhysX 报错，训练从头失败
```

M20 用 0.52m，因为它的大腿+小腿展开后总长约 0.5m，加上机身厚度预留一点。

**调参技巧**：跑 `zero_agent` 看渲染画面。如果机器人出生瞬间往下沉一点但很快稳定，z 就对了；如果整个爆开或整个埋地，z 偏差太大。

### 9.2.2 joint_pos —— 内膝式的数学指纹

```python
joint_pos={
    ".*hipx_joint": 0.0,           # 所有 hipx 侧摆关节 = 0 度（身体正对前方）
    "f[l,r]_hipy_joint": -0.6,     # 前腿 hipy = -0.6 弧度 ≈ -34°
    "h[l,r]_hipy_joint": 0.6,      # 后腿 hipy = +0.6 弧度 ≈ +34°
    "f[l,r]_knee_joint": 1.0,      # 前腿 knee = +1.0 弧度 ≈ +57°
    "h[l,r]_knee_joint": -1.0,     # 后腿 knee = -1.0 弧度 ≈ -57°
    ".*wheel_joint": 0.0,          # 轮子 = 0
}
```

**符号模式读法**：
- `f[l,r]_xxx` → 前腿左/右（front left / front right）
- `h[l,r]_xxx` → 后腿左/右（hind left / hind right）
- 前腿 hipy 负 + 后腿 hipy 正 → **前腿往后折、后腿往前折**，膝盖朝向机身中心
- 前腿 knee 正 + 后腿 knee 负 → **前后膝朝内对折**

这就是"内膝式"的数学特征。**如果你换了构型（外膝/全后折），这套数字要重新算**。

**joint_pos 调试铁律**：
- 初始时**腿不能完全伸直**（否则奇异构型，PD 控制爆）；预留 20~40° 弯曲
- **不能违背 URDF 的关节限位**；超出限位 PhysX 会硬卡
- **膝盖弯曲方向**决定构型——别填反了，否则腿会绕一圈

## 9.3 soft_joint_pos_limit_factor = 0.9

```python
soft_joint_pos_limit_factor = 0.9   # ★ 关节"软限位"是硬限位的 90%
```

**作用**：训练时把 URDF 里写的硬限位放宽到 90%，给策略探索空间。如果填 1.0，策略会被限位频繁打断；如果填 0.5，机器人动作幅度被强制压缩。

**轮足推荐值 0.85~0.95**。M20 用 0.9 是稳健中庸。

## 9.4 actuators = {} —— 电机模型（仿真里的"电调"）

每个 key 对应一类关节。**joint_names_expr** 用正则匹配 URDF 关节名，**类比 C# 的 `Action` 委托 + LINQ Where**：

```python
"joint": DCMotorCfg(                          # 关键名："joint"，本项目多处按 key 引用
    joint_names_expr=[".*hipx_joint", ".*hipy_joint", ".*knee_joint"],   # ★ 正则匹配 12 个腿关节
    effort_limit=76.4,         # ★ 峰值力矩 (N·m)
    saturation_effort=76.4,    # 饱和力矩（达到此值电机开始失能，几乎永远等于 effort_limit）
    velocity_limit=22.4,       # ★ 最大转速 (rad/s)
    stiffness=80.0,            # ★ PD 控制的 P 增益
    damping=2.0,               # ★ PD 控制的 D 增益
    friction=0.0,              # 减速器摩擦，仿真理想化一般 0
)
```

### 9.4.1 ★ 三个必须照实填的参数

| 参数 | 必须怎么填 | 为什么不能虚标 |
|---|---|---|
| `effort_limit` | **查硬件规格书** | 虚标 → 仿真认为能输出的力比真实大，sim-to-real 时真实电机扛不住 |
| `velocity_limit` | **查硬件规格书** | 虚标 → 仿真跑出 1000 rad/s 的离谱关节速度，sim-to-real 直接失稳 |
| `stiffness/damping` | **从真实电机特性估算或拿同类已工程化**

M20 的 76.4 N·m、22.4 rad/s 来自云深处公开规格。

### 9.4.2 ★ stiffness/damping（PD 增益）怎么定

这是**最玄学的参数**——没有理论公式，只有经验范围：

```
★ 起手公式（按机器人重量档位）：

  12kg 档（Unitree Go2）：stiffness ≈ 40
  30kg 档（Lite3）：       stiffness ≈ 60
  60kg 档（M20）：         stiffness ≈ 80
  135kg 档（你的目标）：   stiffness ≈ 120~150
  
  damping ≈ stiffness × 0.025（经验比）
```

**调试方法**：zero_agent 状态下看画面
- 关节乱抖 → stiffness 太小或 damping 太小
- 关节僵硬、机器人像木桩 → stiffness 太大
- 关节能稳稳承重、轻微形变 → 合适

**sim-to-real 黄金法则**：仿真里的 PD 增益要**略低于真实值**（真实机器人减速器有摩擦会帮忙，仿真没有）。即"仿真软一些"。

### 9.4.3 为什么要分两个 actuator（joint 和 wheel）？

```
"joint":   stiffness=80   →  位置控制（PD 把关节拉到目标位置）
"wheel":   stiffness=0    →  速度控制（不强求轮子到任何角度，只给目标速度）
```

对应上层 `ActionsCfg`：
```python
joint_pos = JointPositionActionCfg(...)   # 腿用位置
joint_vel = JointVelocityActionCfg(...)  # 轮用速度
```

**这是轮足运控范式的关键**——两个 actuator 的刚度差异，刚好对应两种控制模式。新手常犯的错：给轮子也填 stiffness=80，结果轮子死活转不到目标速度、训练发散。

## 9.5 调试 ArticulationCfg 的完整流程

```
跑 zero_agent
   │
   ├─ 机器人不在画面里         → 检查 asset_path 路径
   ├─ 出生乱爆 / 变形           → URDF 惯量/质量异常或 z 高度错
   ├─ 关节抖动                  → stiffness/damping 不匹配
   ├─ 某关节完全不动            → joint_names_expr 正则没匹配上
   ├─ 关节方向反了              → URDF 里 joint axis 方向或初始角符号错
   └─ 一切正常但训练失败        → 怀疑 reward/observation，问题不在 ArticulationCfg
```

**一个常见坑**：URDF 里 `joint axis="0 0 1"` 但视觉里轴是反的。这是 URDF 导出问题，仿真里**反向关节角**就能修：

```python
joint_pos={
    "broken_joint": 0.6,   # 负号补偿：填 -0.6 把方向修正回来
}
```

## 9.6 ArticulationCfg 数值速查表（写新机器人时直接套）

| 参数 | 推荐范围 | 不可接受的值 |
|---|---|---|
| `pos[2]` | 腿长 × 0.7~1.0 | 负数、过大（>2m） |
| `init_state.joint_pos` 各值 | 在 ±1.5 rad 范围 | = URDF 硬限位（无探索空间） |
| `soft_joint_pos_limit_factor` | 0.85~0.95 | ≤0.5 或 ≥1.0 |
| `effort_limit` | 查硬件规格书 | 0 |
| `velocity_limit` | 查硬件规格书 | 0 |
| 腿 `stiffness` | 体重(kg) × 1.3 ± 30 | 0 或 > 500 |
| 腿 `damping` | stiffness × 0.02~0.05 | 0 或 > stiffness |
| 轮 `stiffness` | 0 | > 0（会与速度控制打架） |
| 轮 `damping` | 0.3~1.0 | 0 或 > 5 |

---

# 第十部分：模型部署全链路

训练产出 `.pt` 后，部署分三个层次。你大概率需要从 1 走到 3。

```
训练产物：logs/<实验名>/<run名>/model_*.pt
   │
   ├─ 层次 1：仿真内播放（play.py）      ← 你已经在用了
   ├─ 层次 2：导出独立模型（ONNX/JIT）    ← play.py 自动做了
   └─ 层次 3：真机部署（Gazebo/真机器人）  ← 需同作者另一个项目 rl_sar
```

## 10.1 层次 1：仿真内播放

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0

# 加载最新 checkpoint + 自动播放
# 默认加载 logs/rsl_rl/deeprobotics_m20_flat/<最新run>/model_*.pt
```

`play.py` 的关键工作（来自源码 130-260 行）：

```python
# 1. 找 checkpoint 文件
resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

# 2. 关掉所有域随机化（play 不需要）
env_cfg.observations.policy.enable_corruption = False
env_cfg.events.randomize_apply_external_force_torque = None
env_cfg.curriculum.command_levels_lin_vel = None

# 3. 加载策略
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=env.unwrapped.device)

# 4. 推理循环（每帧：观测→策略→动作→环境推进）
while simulation_app.is_running():
    with torch.inference_mode():           # ★ 关闭梯度，省显存、加速
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        policy.reset(dones)                 # 重置 RNN 状态
```

### 10.1.1 键盘手动控制（调试神器）

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=... --keyboard
```

按 Numpad-8/2/4/6 控制 vx/vy/方向，按 Numpad-7/9 或 Z/X 控制 yaw 旋转。**调试时用它测你的 reward 设计是否合理——如果人工操作也站不稳，说明 reward 有问题**。

## 10.2 层次 2：导出独立模型（ONNX / JIT）

**重点：play.py 已经自动帮你导出了！** 每次跑完 play，模型会自动变成两种格式，存到 `<checkpoint_dir>/exported/`：

```python
# play.py 源码 209-233 行（自动执行，不需要你手动跑）
if version.parse(installed_version) >= version.parse("4.0.0"):
    runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")     # JIT (TorchScript)
    runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")  # ONNX（通用）
```

产出物位置：

```
logs/rsl_rl/deeprobotics_m20_flat/2026-09-28_10-30-12/
├── model_5000.pt          ← RSL-RL 训练权重（含 optimizer/RNN 等，**不能直接部署**）
├── exported/
│   ├── policy.pt          ← ★ TorchScript 格式，可加载到 C++/真机
│   └── policy.onnx        ← ★ ONNX 格式，最通用
```

**两种格式的取舍**：

| 格式 | 适合场景 | 优势 | 劣势 |
|---|---|---|---|
| **policy.pt** (TorchScript) | 真机用 PyTorch C++/LibTorch | 精度无损 | 必须装 LibTorch，依赖重 |
| **policy.onnx** | ONNX Runtime（嵌入式/C++/ROS） | 通用，支持各种推理引擎 | 极少数算子可能转换精度损失 |

## 10.3 层次 3：真机部署（rl_sar）

**真机部署需要用配套项目 [`rl_sar`](https://github.com/fan-ziqi/rl_sar)**（README 第 45 行明确推荐）。

### 10.3.1 rl_sar 是什么

- **同作者**（Ziqi Fan）维护的 C++ 部署框架
- 轻量级、可在真机机器人上直接运行
- 支持仿真（Gazebo）和真机两种目标
- **接口已经跟 robot_lab 对接**：策略文件格式兼容

### 10.3.2 完整部署链路

```
robot_lab 训练                 rl_sar 部署
─────────────                  ──────────
训练 1~3 天        ────→      模型加载、推理执行
产出 model_*.pt               用 ONNX 或 JIT 加载
                              与底层电机驱动通信
                              发布关节指令到真实机器人
```

### 10.3.3 部署到真机的 7 步走

```
1. 训练出可播放的 .pt
   ↓
2. 跑 play.py → 自动产出 policy.pt / policy.onnx（不必手动）
   ↓
3. 把你的机器人接入 rl_sar：
   - 在 rl_sar 仓库里加你的机器人 model
   - 写关节映射表（仿真关节 → 真实电机编号）
   - 写观测数据接口（IMU 数据怎么喂给网络）
   ↓
4. Gazebo 中先验证（比真机便宜得多）
   ↓
5. 真机静态测试：把机器人吊起来，喂关节指令，确认电机响应
   ↓
6. 真机悬挂安全测试：吊带保护，慢速运行
   ↓
7. 地面试跑（带急停，全员撤离至安全距离）
```

### 10.3.4 sim-to-real 的常见坑

| 坑 | 原因 | 解法 |
|---|---|---|
| 真机乱抖 | 观测没对齐（仿真里的"关节角度顺序"与真机不同） | 在 rl_sar 里严格按 M20 `leg_joint_names` 顺序传数据 |
| 真机不响应 | 关节方向反了 | URDF 关节 axis 与真机安装方向要对齐 |
| 真机只摔不站 | URDF 质量/惯量与真实不符 | 用真实机器人质量分布表重新写 URDF `<inertial>` |
| 真机性能远差于仿真 | 域随机化不够 | 加大 mass/friction/gain 随机化范围 |
| 真机响应慢 | 通信延迟 | 仿真里加 `sim.dt` 延迟参数模拟 |

## 10.4 部署完整流程检查清单

```
□ 1. 模型在 IsaacLab 仿真里能稳定走（10 分钟以上不倒）
□ 2. play.py 自动产出 policy.onnx 和 policy.pt
□ 3. Gazebo 里跑通（先于真机）
□ 4. 关节顺序、力矩、速度匹配物理真实
□ 5. 通信接口测试通过（CAN/串口/EtherCAT）
□ 6. 安全机制：急停、吊带、安全距离
□ 7. 域随机化强度验证：在极端参数下仿真仍能跑
□ 8. 真机第一跑：低速、空载、地面平坦、人工陪护
```

## 10.5 真机部署的"最小可行路径"

如果你是 C# 工程师，第一次做 RL + 真机，**强烈建议**：

1. **不要直接上真机**。先用 Gazebo 仿真 1~2 周
2. Gazebo 仿真走通了再上**小尺寸、低速度、安全保护的入门机器人**（如 Unitree Go2 教学版，云深处 M20 教学版）
3. 跑通 Go2 完整 sim-to-real 后，再迁你的 135kg 大狗

---

# 第十一部分：rl_sar 接入细节（真机部署框架）

`rl_sar` 是同作者（Ziqi Fan）开源的部署框架，"sar" = Simulation And Real。它**专门为 robot_lab 训练出的策略提供仿真验证与真机部署通路**。本部分告诉你怎么把它用上、怎么踩坑。

## 11.1 它到底是什么

```
robot_lab (Python)        rl_sar (C++)
─────────────┐               │
│ Isaac Sim 训练 │              │
│ PPO 试错     │              │
│ 产出 .pt     │  ──policy──→ │ LibTorch 或 ONNX Runtime 加载
└─────────────┘               │ 与 ROS Gazebo 或真机接口
                              │ 输出关节指令
                              ↓
                          仿真 / 真机器人
```

- **语言**：C++（不是 Python），目的是嵌入式 / 板载运行
- **依赖**：ROS（ROS1 Noetic 或 ROS2 Foxy/Humble）
- **推理后端**：libtorch 或 onnxruntime 二选一
- **仿真器**：Gazebo（主流）、MuJoCo（部分支持）

## 11.2 ⚠️ 三个必须先知道的事实（决定你会不会踩坑）

### 11.2.1 政策文件必须用导出版本

`rl_sar` **不接受 RSL-RL 训练产出的原始 `.pt`**（那个文件里含 optimizer 状态、RNN 隐藏状态等）。

它要的是你在 robot_lab 里用 play.py **自动导出的 TorchScript/ONNX**：

```
✓ rl_sar 接受的：
  policy/<ROBOT>/<CONFIG>/xxx.pt        ← ★ play.py 导出的 TorchScript
  policy/<ROBOT>/<CONFIG>/xxx.onnx      ← ★ play.py 导出的 ONNX

✗ rl_sar 不接受的：
  logs/rsl_rl/.../model_5000.pt         ← 原始 RSL-RL 权重，部署用不了
```

### 11.2.2 ★ 关节顺序是硬约束（最致命陷阱）

`rl_sar/policy/<ROBOT>/base.yaml` 中的关节顺序**必须与物理真机的硬件关节顺序一致**，与 robot_lab 的 `leg_joint_names` 一一对应。

**核心警告**（来自 rl_sar README 的 CAUTION）：
> Sim2Sim 配置中的默认关节映射与真机部署时的不同，**若不对应更新，会导致机器人行为错误甚至安全事故**——使用前必须复查 `policy/<CONFIG>/config.yaml` 与 `rl_real_<ROBOT>.cpp` 中的映射

**这是 sim-to-real 失败的头号原因**：关节顺序错一位，神经网络输出就完全错乱，但表面上看机器人"在动"，你可能直到摔了才察觉。

### 11.2.3 ⚠️ M20 不在 rl_sar 官方支持列表

rl_sar 官方支持的轮足机器人：**Unitree Go2W / B2W / DDTRobot Tita**。**M20 没在支持列表里**。

对你的 135kg 项目意味着：
- 你不能直接用现成的 `m20/` 配置
- 你的接入路径是**参考 `go2w/robot_lab`**（同为 Unitree 轮足）或 **`tita/robot_lab`**（同为内膝式轮足）
- 复制一份配置，按你机器人的关节命名/硬件接口改造

## 11.3 rl_sar 目录结构（你需要改的地方）

```
rl_sar/
├── policy/<ROBOT>/                    ← ★ 你要建的目录
│   ├── base.yaml                      # ★ 真机关节顺序（硬约束）
│   └── <CONFIG>/                      # 一个训练任务一个子目录
│       ├── config.yaml                # ★ 关节映射、参数
│       ├── policy.pt                  # ★ 从 robot_lab 复制过来
│       └── policy.onnx                # ★ 备选
└── src/rl_sar/
    ├── src/rl_real_<ROBOT>.cpp        # ★ 真机控制代码（按机型新增）
    └── rl_sar_zoo/<ROBOT>_description/
        ├── xacro/                     # URDF 描述
        └── robot_control.yaml         # 控制器配置
```

## 11.4 接入 12 个支持机器人外的机器人（比如你的 135kg 大狗）

要做 4 件事：

| 步骤 | 文件 | 做什么 |
|---|---|---|
| ① 复制 base 模板 | `policy/<ROBOT>/base.yaml` | 列出你机器人真实的关节顺序 |
| ② 关节映射 | `policy/<ROBOT>/<CONFIG>/config.yaml` | 按 M20 的 16 关节顺序配置 |
| ③ 真机控制程序 | `src/rl_sar/src/rl_real_<ROBOT>.cpp` | 新建 C++ 文件，定义 forward()/compute() |
| ④ 机器人描述 | `src/rl_sar/rl_sar_zoo/<ROBOT>_description/` | xacro + robot_control.yaml

最难的通常是 ③ 和 ④ —— 因为这需要你**机器人的 SDK 资料**（电机通信协议、控制命令格式、传感器读取接口）。

## 11.5 真机部署的连接方式（按机器人机型差异大）

| 机器人 | 连接方式 | IP/端口约定 |
|---|---|---|
| Unitree Go2/Go2W/B2W/G1 | **以太网** | 板载 IP `192.168.123.161`，运行 `rl_real_go2 <网卡名>` |
| Unitree A1 | WiFi 或有线 | 电脑设 IP `192.168.123.162` |
| Deeprobotics Lite3 | **无线** | 需修改 `rl_real_lite3.cpp` IP/端口 |
| Agibot D1 | WiFi | 默认 `192.168.234.1`，需配 sdk_config.yaml |
| **你的 135kg 大狗** | 取决于硬件方案 | 提前与电控组沟通 |

**你的 135kg 项目在硬件板设计时**，建议直接按主流方案的 IP/网络模式做（默认 192.168.123.x 网段，Ethernet UDP），后期 rl_sar 接入会简单得多。

## 11.6 三种调试友好的部署模式

| 模式 | 工具 | 适用阶段 |
|---|---|---|
| **Web 网页控制** | 浏览器访问 `http://robot.robotsfan.com/` | 远程调试、展会演示 |
| **键盘控制** | 键位 A/D 横移、W/S 前后、Q/E 偏航 | 室内调试 |
| **手柄控制** | 标准游戏手柄 | 室外测试、遥操作遥现 |

**推荐**：第一周用 Web 远程控制（局域网内任何浏览器都行），出问题可立刻切断。

## 11.7 编译运行

```bash
git clone --recursive https://github.com/fan-ziqi/rl_sar.git
cd rl_sar
./build.sh                            # ROS 模式（默认）
./build.sh -m                         # 纯 CMake（板载 Jetson 部署用）
```

之后两个终端：

```bash
# 终端 1：启动 Gazebo 仿真
roslaunch rl_sar gazebo.launch rname:=<你的机器人>

# 终端 2：启动控制程序
rosrun rl_sar rl_sim                  # ROS1
ros2 run rl_sar rl_sim                # ROS2
```

**重要警告**（README 强调）：仿真启动后必须在另一终端**立刻启动控制程序**，否则机器人没人在 control 回路，会自由落体摔毁。

## 11.8 你接入自定义机器人的 5 步流程

```
1. 训练策略
   robot_lab 训出 policy.pt / policy.onnx
        ↓
2. 建机器人配置
   复制 go2w/ 或 tita/ 到 policy/<你的机器人>/
   改 base.yaml 的关节顺序
   改 config.yaml 的关节映射
        ↓
3. 复制机器人 URDF
   rl_sar_zoo/<你的机器人>_description/xacro/
        ↓
4. 写真机控制代码
   src/rl_sar/src/rl_real_<你的机器人>.cpp
   （要熟悉你机器人 SDK 的通信协议）
        ↓
5. 编译 + Gazebo 验证
   ./build.sh
   roslaunch + rosrun
```

---

# 第十二部分：Gazebo 仿真验证流程

**为什么必须过 Gazebo？** 直接上真机的风险：摔毁（机器人价值 50~200 万）、场地/人员损伤、调参周期长。Gazebo 是 sim-to-real 链路的中间验证站——成本几乎为零，能复现 ~80% 的真机问题。

## 12.1 Gazebo 与 IsaacLab 的关系

```
Isaac Lab（训练）     Gazebo（验证）        真机
─────────────────    ─────────────────    ────────────
GPU 加速、4096 并行   CPU、单台仿真          物理真实
域随机化开启          域随机化关闭            物理真实
策略快速迭代          验证策略可行性          验证稳定性
模型理想化            URDF 精确模型           URDF 精确模型
```

## 12.2 Gazebo 仿真环境的构成

```
Gazebo 世界（sdf 文件）
├── 物理引擎（ODE/Bullet/DART，默认 ODE）
├── 机器人模型（URDF → Gazebo 解析器）
├── 地面/障碍物（sdf 或 mesh）
├── 传感器插件（IMU、LiDAR、接触力）
├── ros_control 控制器（关节控制器、状态发布器）
└── tf 坐标系变换
```

**Gazebo 需要 URDF 包含 Gazebo 专属标签**：

```xml
<robot>
  <!-- 关节 Gazebo 化（关节传动+控制器） -->
  <transmission name="...">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="..."><hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface></joint>
    <actuator name="..."><mechanicalReduction>1</mechanicalReduction></actuator>
  </transmission>

  <!-- Gazebo 物理插件 -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/</robotNamespace>
    </plugin>
  </gazebo>
</robot>
```

**M20 的 URDF 直接来自云深处**，已包含 Gazebo 标签，但可能有 Gazebo 版本兼容问题需要适配。

## 12.3 rl_sar 在 Gazebo 上的双终端工作流

```bash
# 终端 1：启动 Gazebo + URDF + ros_control
roslaunch rl_sar gazebo.launch rname:=go2w
# ↑ 加载机器人 URDF、生成世界、加载 controllers

# 终端 2：启动控制节点
rosrun rl_sar rl_sim
# ↑ 加载 policy.onnx / policy.pt
# ↑ 订阅 /joint_states、发布 /cmd_joint_pos / /cmd_joint_vel
```

两个终端必须都在，且**仿真启动后 3 秒内控制节点必须 ready**（机器人从站姿自由落体只有 ~0.5 秒，但控制器启动延迟会累积）。

## 12.4 验证清单（机器人站起来之前的必检项）

### 12.4.1 静态验证

| 检查项 | 怎么检查 | 期望结果 |
|---|---|---|
| URDF 加载成功 | Gazebo 控制台无红字 error | 机器人在世界里渲染正常 |
| joint_states 频率 | `rostopic hz /joint_states` | 稳定 100~1000Hz |
| tf 树完整 | `rosrun tf2_tools view_frames.py` | base_link → 各连杆 |
| 控制器加载 | `roscontrol list_controllers` | 每个关节有 EffortJointInterface 控制器 |

### 12.4.2 单关节响应测试

```bash
# 给一个关节下发一个目标角度，看实际值
rostopic pub /fl_hipy_joint_controller/command std_msgs/Float64 "data: 0.5"
rostopic echo /joint_states | grep fl_hipy
```

期望：几秒内实际关节角度跟到 0.5 rad 附近。

### 12.4.3 策略启动测试

```bash
rosrun rl_sar rl_sim   # 启动策略控制
```

观察：
- 机器人是否**立刻获得稳定性**（不在重力下摔倒）？
- 静态站立**有没有漂移**？稳定 30 秒以上视为通过
- 关节是否在限位内？有没有奇异姿态？

### 12.4.4 速度指令测试（用 Web 控制或键盘）

观察机器人在轮子驱动下的响应：

- 给 vx=0.5 命令，机器人**直线**前进（不偏）
- 给 ωz=0.5 命令，机器人**匀速**原地旋转
- 命令归零，机器人**快速停止**

### 12.4.5 模拟故障测试

| 测试 | 怎么触发 | 期望 |
|---|---|---|
| 关节卡死 | 给某关节发巨大力矩命令 | 机器人跌倒不失控飞出去 |
| 通信中断 | kill 控制节点 | 机器人进入安全姿态（电机 soft stop）|
| 推力扰动 | 终端发外力消息 | 机器人恢复平衡 |

## 12.5 常见 Gazebo 集成坑

| 坑 | 症状 | 解法 |
|---|---|---|
| 关节方向反了 | 机器人在原地跳舞 | 检查 URDF `axis xyz="..."` 和 rl_sar `forward()` 函数的关节顺序 |
| 坐标系不对齐 | 机器人以奇怪姿态悬浮 | 检查 base_link 的位置朝向、IMU 的安装位置 |
| joint_states 延迟 | 策略看到的状态滞后 | 调高 controller update rate（默认 1kHz）|
| `JointTrajectory` 报错 | 控制命令下发失败 | 检查 controllers.yaml 中 joint 名是否与 URDF 完全一致 |
| tf 树断链 | 传感器数据帧错乱 | 重写 xacro 里 `<joint name="imu_joint">` |
| Gazebo 启动卡住 | GPU 渲染冲突 | 改用 CPU 渲染或升级显卡驱动 |

## 12.6 sim2sim 与 sim2real 差异（必修认知）

```
Isaac Lab → 真机：
  - 物理引擎：PhysX（GPU）    → 物理引擎：真机（CPU 控制板）
  - 执行器：DCMotorCfg (理想)  → 执行器：真实电机（延迟、饱和、非线性）
  - 传感器：无噪声              → 传感器：真值+偏差+噪声
  - 地面：参数化粗糙地形        → 地面：现实未知
  - 通信：无延迟                → 通信：1~10ms 延迟
```

**Gazebo 的作用是模拟这些中间态**：
- 用 CPU 物理引擎（DART/Bullet）对比 GPU 物理引擎
- 给关节加力矩延迟
- 给传感器加噪声
- 测试不同的地面摩擦

**Gazebo 通过了 ≠ 真机成功**，但 Gazebo 都不通过 ≈ 真机必失败。

---

# 第十三部分：域随机化深度配置

**域随机化（Domain Randomization, DR）是 sim-to-real 能否成功的决定性因素。**

## 13.1 直觉：为什么要域随机化

```
不随机化：
  仿真里 dog 长这样 → 策略死记硬背 → 真机 dog 长不同 → 策略崩溃

随机化后：
  仿真里 dog 长千万种样子 → 策略学会"无论 dog 长啥样都要站起来"
                          → 真机狗遇到任何细节差异 → 依然能玩
```

**本质是牺牲仿真性能，换真机鲁棒性**。但随机化太强 → 仿真任务太难 → 策略学不会；随机化太弱 → 仿真变成没训练 → 策略过拟合。**这是个调参艺术**。

## 13.2 M20 的 7 类随机化（velocity_env_cfg.py）

```
class EventCfg:
    """Configuration for events."""
    # ------------------ startup（环境初始化）------------------
    randomize_rigid_body_material       # ① 刚体表面摩擦/弹性
    randomize_rigid_body_mass_base      # ② 机身质量
    randomize_rigid_body_mass_others    # ③ 腿部质量
    randomize_com_positions            # ④ 重心位置

    # ------------------ reset（每次回合重置）------------------
    randomize_apply_external_force_torque   # ⑤ 重置时施加的扰动力
    randomize_reset_joints                  # ⑥ 关节初始角度
    randomize_actuator_gains                 # ⑦ PD 增益（执行器参数）
    randomize_reset_base                    # ⑧ base 位置/姿态重置范围

    # ------------------ interval（按时间间隔触发）------------------
    randomize_push_robot                   # ⑨ 模拟推力扰动
```

下面逐项给推荐范围 + 调参原则。

## 13.3 各项随机化详解

### 13.3.1 ① 摩擦/弹性（地面+机身接触）

```python
randomize_rigid_body_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",                              # ★ 一次性，每个回合开始不再变
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "static_friction_range": (0.3, 1.0),     # 静摩擦系数
        "dynamic_friction_range": (0.3, 0.8),    # 动摩擦系数
        "restitution_range": (0.0, 0.5),         # 弹性恢复系数
        "num_buckets": 64,                        # ★ 离散档位（不同环境用不同值）
    },
)
```

**调参原则**：
- **真实地面的常见摩擦**：水泥 0.6、木板 0.4、草地 0.5、湿滑 0.2
- **推荐范围**：`static_friction_range=(0.4, 1.2)`、`dynamic_friction_range=(0.3, 1.0)`
- **轮足重点**：摩擦下限不要 < 0.2，否则轮子会打滑失控

### 13.3.2 ②③ 质量随机化（★ 影响最大的一类）

```python
randomize_rigid_body_mass_base = EventTerm(       # 机身：偏移型
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=""),  # body_names="" 指 base
        "mass_distribution_params": (-1.0, 3.0),  # 偏移范围：±1~3 kg
        "operation": "add",                         # 偏移 vs 缩放
        "recompute_inertia": True,                  # ★ 必改，质量变了惯量要重算
    },
)

randomize_rigid_body_mass_others = EventTerm(    # 腿部：缩放型
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "mass_distribution_params": (0.7, 1.3),  # 缩放范围：70%~130%
        "operation": "scale",                       # 偏移 vs 缩放
        "recompute_inertia": True,
    },
)
```

**调参原则**：
- **机身用 `add`（绝对值）**：因为真实电池重量增减是固定的（拿掉一块电池 = 减固定重量）
- **肢体用 `scale`（比例）**：因为腿部重量误差通常是比例性的
- **范围选取**：根据你对机器人制造误差的估计。一般 ±30% 是合理起点

### 13.3.3 ④ 重心位置（COM）扰动

```python
randomize_com_positions = EventTerm(
    func=mdp.randomize_rigid_body_com,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        # COM 偏移范围 ±5cm
    },
)
```

**调参原则**：
- ±5cm 是仿真里的"误差"，对应你**实际装配误差 + 传感器/电池位置公差**
- 135kg 大狗推荐 ±5~10cm（重量大、重心偏移影响大）

### 13.3.4 ⑤⑥⑦⑧ 重置时的随机化（每回合变）

```python
randomize_apply_external_force_torque = EventTerm(
    func=mdp.apply_external_force_torque,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=""),
        "force_range": (-10.0, 10.0),         # 推力（N）
        "torque_range": (-10.0, 10.0),        # 扭矩（N·m）
    },
)

randomize_reset_joints = EventTerm(
    func=mdp.reset_joints_by_scale,
    mode="reset",
    params={
        "position_range": (1.0, 1.0),        # 关节初始角度 = 默认角 × 1.0（不变）
        "velocity_range": (0.0, 0.0),
    },
)

randomize_actuator_gains = EventTerm(            # ★ PD 增益随机化（关键）
    func=mdp.randomize_actuator_gains,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "stiffness_distribution_params": (0.5, 2.0),    # 缩放范围：50%~200%
        "damping_distribution_params": (0.5, 2.0),
        "operation": "scale",
        "distribution": "uniform",
    },
)

randomize_reset_base = EventTerm(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
        },
    },
)
```

**PD 增益随机化是 sim-to-real 命脉**——因为真实电机在低温/低电压/老旧状态下 K、d 都会变。
- **推荐范围**：stiffness 和 damping 都 `(0.5, 2.0)` 即 50%~200%
- **太大会崩**：超过 `(0.2, 3.0)` 策略基本学不会
- **太小没意义**：低于 `(0.8, 1.2)` 不起作用

### 13.3.5 ⑨ 推力扰动（按时间触发，模拟风/碰撞）

```python
randomize_push_robot = EventTerm(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(10.0, 15.0),              # 每 10~15 秒触发一次
    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
)
```

**调参原则**：
- 频率 10~15 秒一次太稀疏（机器人可能永远碰不到）
- 频率 2~3 秒一次太频繁（机器人一直处于恢复状态，学不会前进）
- 工业巡检场景推荐 5~8 秒一次
- 推力幅值按机器人质量档位，135kg 推荐 ±1.0 m/s

## 13.4 域随机化强度速查表（按机器人重量）

| 参数 | 12kg 档 (Go2) | 60kg 档 (M20) | 135kg 档 (你的) |
|---|---|---|---|
| `mass_distribution_params` (others) | (0.8, 1.2) | (0.7, 1.3) | (0.7, 1.3) |
| `mass_distribution_params` (base, add) | (-0.5, 1.5) | (-1.0, 3.0) | (-2.0, 5.0) |
| `com_range` (m) | ±0.03 | ±0.05 | ±0.08 |
| `stiffness_distribution_params` | (0.7, 1.5) | (0.5, 2.0) | (0.5, 2.0) |
| `push velocity_range` (m/s) | ±0.3 | ±0.5 | ±0.8~1.0 |
| `friction_range` | (0.4, 1.2) | (0.3, 1.0) | (0.3, 1.0) |

## 13.5 调参原则（★ 最重要的一条）

**域随机化要"够用就行"，不要"过度随机化"**：

```
训练难度 = 基础任务难度 + 随机化引入的难度

如果：
  任务难度陡增（地形变难、速度变快）→ 减小随机化
  任务难度变简单 → 可以加大随机化

原则：让基础任务上能学成（80% 成功率），但不能简单到让策略记住
```

## 13.6 验证方法：消融实验

依次关掉一类随机化，看训练和最终性能：

```
全部开启（基线）     → 训练慢、最终鲁棒
关掉质量            → 训练快、但真机失败
关掉 PD 增益        → 训练快、但电机参数一变就崩
关掉推力扰动        → 训练快、但遇外力推就倒
关掉 COM 扰动       → 训练快、但重心不准就失败
```

如果关掉某一类**对训练效率提升很大**、但**对真机性能影响很小**——说明这一类随机化没必要开。

---

# 第十四部分：PPO 超参数详解

你训练时 80% 的失败原因**不是 PPO 公式**，而是 PPO 超参数没设对。本部分把 M20 的 13 个核心超参数逐一讲清楚。

## 14.1 PPO 一句话解释

**PPO（Proximal Policy Optimization，近端策略优化）= 限制每次更新的幅度，避免训练崩溃。

```
θ_new = θ_old + step
但 step 不能太大，否则会把已经学好的东西冲掉

PPO 的诀窍：把更新幅度强行限制在 old 周围一个"信赖域"里
```

对你来说这就是个"比较稳的 RL 算法"。它有大量超参数，但**好消息：默认值在 M80 那里都已经调好了，你的首要工作是诊断"该不该改"，而不是"怎么改"**。

## 14.2 M20 完整 PPO 配置（重看一遍，对照讲解）

```python
# rsl_rl_ppo_cfg.py
class DeeproboticsM20FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24                  # A. 每个环境每轮采样的步数
    max_iterations = 5000                   # B. 总训练轮数
    save_interval = 100                     # C. 每隔多少轮保存 checkpoint
    experiment_name = "deeprobotics_m20_flat"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                 # D. 初始探索噪声标准差
        actor_obs_normalization=False,      # E. 是否归一化观测
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],  # F. 策略网络结构
        critic_hidden_dims=[512, 256, 128], # G. 评论家网络结构
        activation="elu",                   # H. 激活函数
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,                # I. 价值损失权重
        use_clipped_value_loss=True,
        clip_param=0.2,                     # K. PPO 截断参数（★最重要）
        entropy_coef=0.01,                  # L. 熵正则（鼓励探索）
        num_learning_epochs=5,              # M. 每批数据学几遍
        num_mini_batches=4,                 # N. 切成几小块学
        learning_rate=1.0e-3,               # O. 学习率
        schedule="adaptive",                # P. 学习率调度
        gamma=0.99,                         # Q. 折扣因子
        lam=0.95,                           # R. GAE 参数
        desired_kl=0.01,                    # S. KL 散度阈值（adaptive 调度用）
        max_grad_norm=1.0,                  # T. 梯度裁剪
    )
```

## 14.3 13 个超参数逐一讲

### A. `num_steps_per_env = 24`

**含义**：每个虚拟机器人每轮收集多少步经验后才更新一次网络。

**为什么是这个数**：
- 太小（<10）→ 数据利用率低，学习慢
- 太大（>50）→ 单次更新的"数据太老"，相关性低
- 24 是 RSL-RL 默认，对轮足场景经验证稳定

**调整时机**：除非显存或速度瓶颈，否则别改。

### B. `max_iterations = 5000`（Flat）/ 20000（Rough）

**含义**：训练总迭代次数。

**经验值**：

| 场景 | 推荐迭代数 | 原因 |
|---|---|---|
| 平地速度跟踪 | 3000~5000 | 简单任务 |
| 粗糙地形跟踪 | 15000~30000 | 复杂地形 + 课程学习 |
| 上台阶/越障 | 20000~50000 | 离散地形 |
| 推力扰动强 | +50% | 抗扰动需要更多数据 |

**判断标准**：TensorBoard 看 `Episode Reward` 是否还在涨。如果 1000 轮以上没变化，可以提前停。

### C. `save_interval = 100`

**含义**：每 100 轮保存一次 checkpoint（用于回溯挑选最佳模型）。

**默认就够用**。磁盘不够可以改 200~500。

### D. `init_noise_std = 1.0`

**含义**：策略输出初始噪声标准差（影响训练初期探索范围）。

**调参**：
- 太大（>2.0）→ 训练初期动作剧烈，机器人疯狂乱动
- 太小（<0.3）→ 探索不足，陷入局部最优
- M20 用了 1.0（标准值）

### E. `actor_obs_normalization = False`

**含义**：是否对策略网络的输入做归一化。

**False 是因为**：IsaacLab 的观测管理器已经做了归一化（在 observations 配置中），这里再归一化一次会出问题。False 是对的，别动。

### F/G. 网络结构 `[512, 256, 128]`

**含义**：三层隐藏层，神经元数依次减半。

**调参经验**：

| 任务 | 推荐结构 |
|---|---|
| 简单平地 | [256, 128, 64] 或 [128, 128] |
| 标准速度跟踪 | [512, 256, 128]（M20 在用） |
| 复杂地形 + 高度图 | [512, 256, 128] 或 [1024, 512, 256] |
| 你 135kg 大狗 | 起步 [512, 256, 128]，不够再加宽 |

**原则**：宽度不够（学不会）就加宽；深度不够（欠拟合）就加深。先调宽度。

### H. `activation = "elu"`

**含义**：激活函数。

**RL 经验**：ELU > ReLU（更适合负值输入，避免梯度消失）。M20 用 ELU 是对的。其他可选 `relu` `tanh` `gelu`，但 **elur** 通常最优。

### I. `value_loss_coef = 1.0`

**含义**：评论家（value function）的损失权重。

**默认 1.0 别动**。它平衡 actor 损失和 critic 损失。

### K. ★ `clip_param = 0.2`（最重要）

**含义**：PPO 截断参数——限制策略更新的最大幅度。

```
r_t(θ) = new_prob / old_prob
clip(r_t, 1-0.2, 1+0.2)
```

**为什么 0.2**：
- 太大（>0.3）→ 更新太激进，训练可能崩溃
- 太小（<0.1）→ 学得极慢，浪费时间
- 0.2 是 PPO 原始论文的"黄金值"，M20 直接用了

**调整时机**：几乎永远别改，除非你明确知道在做什么。

### L. `entropy_coef = 0.01`

**含义**：熵正则项权重（鼓励策略保持多样性，避免过早收敛）。

**调参经验**：
- 训练后期不收敛、曲线仍在波动 → 加大到 0.02~0.05
- 训练过早收敛到次优解 → 加大到 0.02
- 训练前期正常后期崩溃 → 减小到 0.005

**M20 用 0.01 是中间偏保守值**。

### M. `num_learning_epochs = 5`

**含义**：每批数据训练几轮。

**调整**：
- 太大（>10）→ 过拟合风险
- 太小（<3）→ 数据浪费
- 5 是 RSL-RL 默认

### N. `num_mini_batches = 4`

**含义**：每批数据切成几小块。

**计算关系**：每个环境的样本数 = `num_steps_per_env × num_envs`，批次大小 = 样本数 / `num_mini_batches`。

**调整**：除非你明确知道 batch size 的影响，否则别改。

### O. `learning_rate = 1.0e-3`

**含义**：神经网络学习率。

**调参经验**（★ 经常需要调）：

| 训练状态 | 调整 |
|---|---|
| 训练前期涨不动 | 加大到 3e-3 |
| 训练后期震荡/不收敛 | 减小到 5e-4 |
| 课程学习换挡时崩溃 | 临时降到 5e-4 再恢复 |
| 默认通用 | 1e-3 |

### P. ★ `schedule = "adaptive"` + S. `desired_kl = 0.01`

**含义**：自适应学习率——当 KL 散度（策略更新的总幅度）超过 0.01 时，自动降低学习率。

**为什么是 adaptive**：
- 比 `constant`（恒定）更稳定
- 比 `linear`（线性衰减）更灵活
- `adaptive` 是 RSL-RL 推荐值

**调参**：如果你训练曲线后期震荡，可以把 `desired_kl` 从 0.01 降到 0.005（更保守）。

### Q. `gamma = 0.99`

**含义**：折扣因子——未来奖励按 `0.99^t` 折现到现在。

**含义解释**：gamma=0.99 意味着 100 步未来的奖励相当于现在的 `0.99^100 ≈ 0.366`。

**调整时机**：几乎永远别动。

### R. `lam = 0.95`

**含义**：GAE（Generalized Advantage Estimation）的 λ 参数。

**调整**：通常和 gamma 一起调整，**别改**。

### T. `max_grad_norm = 1.0`

**含义**：梯度裁剪——梯度范数超过 1.0 时强制缩小。

**作用**：防止梯度爆炸。

**调整**：通常 1.0~0.5，**别动**。

## 14.4 训练不收敛的诊断表（★ 实战用）

| 现象 | 病因 | 调整 |
|---|---|---|
| **Episode Reward 一直贴 0** | 环境配置问题（观测/奖励有 bug） | 先 zero_agent + 手动操作测试 |
| **Reward 涨一会儿然后暴跌** | 学习率过大 | 降 lr 到 5e-4 |
| **Reward 涨到某值卡住** | 策略陷入局部最优（用轮子滑行等作弊） | 加 penalty、调奖励设计 |
| **`Policy/mean_noise_std` 骤降到 0** | entropy_coef 太小 | 加 entropy_coef 到 0.02 |
| **`mean_noise_std` 一直在 1.0** | entropy_coef 太大或学习率太小 | 减 entropy_coef 到 0.005；加 lr 到 3e-3 |
| **训练快但真机失败** | 域随机化不够（详见第十三部分） | 加 mass/com/gain 随机化 |
| **Episode Length 一直很短** | 终止条件太严 | 检查 illegal_contact 是否误触发 |
| **训练后期越来越慢（性能反而下降）** | 学习率没衰减 | 改 schedule=adaptive，或手动线性衰减 |
| **显存爆** | num_envs 太多 / height_scanner 太密 | --num_envs 2048、降低 height_scan 分辨率 |

## 14.5 训练经验法则（★ 总结）

```
1. 永远先看 Episode Reward 曲线，再看其他指标
2. 永远改一个参数，对照一组实验
3. 永远用 --run_name 命名每个实验
4. 永远备份一个"已知能跑通"的 baseline 配置
5. 90% 的"训练失败"其实是奖励设计问题，不是 PPO 问题
6. 域随机化越激进，前期越慢，但后期越好
7. GPU 够的话，num_envs 越大越好（4096 是甜点）
```

## 14.6 PPO 超参数调参顺序（重要 → 不重要）

```
1. learning_rate ★★  （最常需要调，影响收敛速度和稳定性）
2. entropy_coef ★    （应对过/欠探索）
3. desired_kl ★      （adaptive 调度的"刹车灵敏度"）
4. max_iterations    （训练时长，不够就加）
5. 网络结构            （欠拟合时加宽）
6. num_steps_per_env / num_mini_batches / num_learning_epochs  （极少见需要调）
7. clip_param / gamma / lam / max_grad_norm  （不要动）

---

*文档基于 robot_lab main 分支（IsaacLab 2.3.2 / Isaac Sim 5.1）实际代码结构编写，所有文件路径均已核对。*

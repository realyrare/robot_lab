# M20 训练速查手册

> 适用分支：**dev-W1**（robot_lab 2.3.2 ↔ IsaacLab 2.3.2 ↔ Isaac Sim 5.1）
> 机器：RTX 4090 24GB / conda 环境 `isaaclab`

---

## 0. 每次开工先做（切分支后必做）

```bash
cd /home/ancn/mhgcode/robot_lab
conda activate isaaclab
pip install -e source/robot_lab --no-deps
```

`robot_lab` 是 editable 安装（指向 `source/robot_lab`），切分支后代码自动生效，
但重装能刷新版本元数据，避免残留旧版本号。

确认当前分支和版本：

```bash
git branch --show-current
cat VERSION          # dev-W1 应为 2.3.2
```

---

## 1. 冒烟测试（先跑这个，2~5 分钟）

```bash
python scripts/tools/zero_agent.py \
  --task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0 --num_envs 32

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0 \
  --headless --num_envs 1024 --max_iterations 300 --run_name smoke
```

---

## 2. 正式长训（后台跑 + 终端看进度）

```bash
cd /home/ancn/mhgcode/robot_lab
conda activate isaaclab

nohup timeout 7800 python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0 \
  --headless --num_envs 4096 --max_iterations 6000 --run_name v6000 \
  > train_m20.log 2>&1 & tail -f train_m20.log | grep --line-buffered -E "Learning iteration|Mean reward|Traceback"
```

- `Ctrl-C` **只停 tail，训练继续**；再想看就重敲 `tail -f`
- `timeout 7800` = 130 分钟保险丝（6000 轮约 90 分钟）
- 被截断也不怕：`save_interval=100`，最近一个整百 checkpoint 已落盘

续跑（注意 `--max_iterations` 是**再跑多少轮**，不是总数）：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0 \
  --headless --num_envs 4096 \
  --resume --load_run <run目录名> --load_checkpoint model_4999.pt \
  --max_iterations 1000 --run_name ext
```

Rough 地形版（任务名换 Rough，默认 20000 轮）：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Deeprobotics-M20-v0 \
  --headless --num_envs 4096 --max_iterations 20000 --run_name rough_v1
```

---

## 3. 看曲线（TensorBoard，另开一个终端）

```bash
conda activate isaaclab
cd /home/ancn/mhgcode/robot_lab
tensorboard --logdir logs/rsl_rl/deeprobotics_m20_flat --port 6008
```

浏览器：**http://localhost:6008**

- 看全部 run 对比 → `--logdir` 指到 `deeprobotics_m20_flat`
- 只看某一次 → 指到具体 `<时间戳>_<run_name>` 目录
- 其他机器访问 → 加 `--bind_all`

值得看的指标：

| 指标 | 含义 |
|---|---|
| `Episode/Reward/total` | 总奖励，应持续上升 |
| `rew_track_lin_vel_xy_exp` | 线速度跟踪，越接近 1 越好 |
| `error_vel_xy` | 实际速度误差，越接近 0 越好 |
| `mean_episode_length` | ⚠️ 见下方「坑 4」，**不可信** |

---

## 4. 播放 / 看效果

```bash
conda activate isaaclab
cd /home/ancn/mhgcode/robot_lab

# 实时窗口（不加 --headless）
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0 \
  --num_envs 4 \
  --load_run 2026-10-13_15-32-02_v6000

# 或者：直接给 checkpoint 完整路径
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0 \
  --num_envs 4 \
  --checkpoint /home/ancn/mhgcode/robot_lab/logs/rsl_rl/deeprobotics_m20_flat/<run目录>/model_5999.pt

# 卡顿时改用 headless 录视频（最稳）
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0 \
  --num_envs 8 --headless --video --video_length 500 \
  --load_run <run目录>
```

视频落在 `logs/rsl_rl/deeprobotics_m20_flat/<run目录>/videos/play/`

---

## 5. 关键参数速记

| 参数 | 含义 | 常用值 |
|---|---|---|
| `--num_envs` | **并行仿真多少台机器人** | 冒烟 1024 / 长训 4096 / 播放 4~8 |
| `--max_iterations` | 训练多少轮 | flat 默认 5000，rough 默认 20000 |
| `num_steps_per_env` | 每台每轮跑几步 | 24（配置里写死） |
| `save_interval` | 每几轮存一次 | 100 |

样本量 = `num_envs` × `num_steps_per_env` × `max_iterations`
例：4096 × 24 × 6000 ≈ **5.9 亿 env-step**

`--num_envs` 只影响**每轮样本量和速度**，不影响轮数。

---

## 6. 目录结构

```
logs/rsl_rl/<experiment_name>/<时间戳>_<run_name>/
        ↑                          ↑
  deeprobotics_m20_flat      2026-10-13_15-32-02_v6000
  （任务名，cfg 里写死）      （启动时刻 + --run_name）

里面：model_0.pt / model_100.pt / ... / model_5999.pt
      events.out.tfevents.*  （TensorBoard 数据）
      params/  git/  videos/
```

⚠️ **日志目录由「运行时的 cwd」决定**（`train.py` 里是相对路径 `logs/rsl_rl/...`），
先 `cd` 到哪就落在哪。`.gitignore` 有 `**/logs/*`，不会污染 git。

---

## 7. 踩过的坑（重要）

**坑 1：GUI 模式报 `cannot import name 'TypeForm'`**
两份旧版 `typing_extensions` 抢先加载（Isaac Sim 预置的 4.12.2 / 4.10.0）。
已修复：把 `site-packages/typing_extensions.py`（4.16.0）覆盖到这两处：

```bash
isaacsim/kit/data/Kit/Isaac-Sim/5.1/exts/3/omni.kit.pip_archive-*/pip_prebundle/typing_extensions.py
isaacsim/extscache/omni.services.pip_archive-*/pip_prebundle/typing_extensions.py
```

**坑 2：`handle_deprecated_rsl_rl_cfg` 导入失败**
IsaacLab 2.3.2 不提供这个符号。dev-W1 的 `train.py` 已加 `try/except` 兜底。

**坑 3：`play.py` 的 `--checkpoint` 是「文件路径」不是「文件名」**
传 `model_5999.pt` 会报 `FileNotFoundError`，且会忽略 `--load_run`。
要么给完整路径，要么只给 `--load_run` 让它自动挑最新。

**坑 4：`mean_episode_length` 不能用来判断摔倒** ← **已修复（2026-10-21）**
原来 `rough_env_cfg.py` 里 `terminations.illegal_contact = None`，没有摔倒终止，
这条恒为 1000。现在已改为：

```python
self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
self.terminations.illegal_contact.params["threshold"] = 1.0
```

`mean_episode_length` 现在**能反映摔倒率**了（越接近 episode 上限越好）。
回退：把这两行改回 `self.terminations.illegal_contact = None`。

**坑 5：Rough 任务其实是「盲跑」** ← **已修复（2026-10-21）**
原来 `observations.policy.height_scan = None`，策略看不到地形。
现在已打开，策略能拿到 **160 维**高度扫描
（`height_scanner` 网格 `resolution=0.1, size=[1.6,1.0]` → 16×10 点）。

⚠️ 副作用：观测维度 +160，训练变慢；flat 与 rough 的观测维度不再一致，
**flat 训出来的权重不能直接给 rough 用**。
回退：把那行改回 `self.observations.policy.height_scan = None`。

**坑 6：GUI 播放时窗口「无响应」**
CPU 在 powersave 模式 + 32 个环境全渲染导致。
解决：`--num_envs 4`、切 `performance` 模式、或用 headless 录视频。

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**坑 7：github 连不上，`git fetch` 超时**
不影响切分支 —— 本地若已有 `origin/dev-W1` 引用可直接 `git checkout -b dev-W1 origin/dev-W1`。

---

## 8. 日常排查命令

```bash
# 训练在不在跑
pgrep -af "rsl_rl/train.py"

# 看实时进度（数 checkpoint）
watch -n 30 'ls -t logs/rsl_rl/deeprobotics_m20_flat/<run目录>/model_*.pt | head -1'

# 确认日志落点（启动 1 分钟后）
grep "Logging experiment in directory" train_m20.log

# GPU 状态
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 2
```

# MotrixSim VLA 彩色方块堆叠

本项目对应测试题中的 `stack_color_blocks`：在 MotrixSim 中使用 UR5e 和 Robotiq 2F85 夹爪，将指定颜色方块抓起并叠放到另一颜色方块上。下文命令均从仓库根目录执行。场景的尺寸、相机外参和随机范围是本项目的仿真假设，详情见 [任务说明](docs/task_spec.md)。

## 方法与结果总结

**方法。** 基于 [场景配置](configs/scene.yaml)搭建机械臂、夹爪、彩色方块、全局相机与腕部相机。采集脚本随机化方块初始位姿和光照，运行脚本化专家轨迹，仅保存通过任务成功及数据完整性检查的回合。发布数据包含 **105 条成功轨迹、40,226 帧**；以 20 Hz 记录两路 224 × 224 RGB、10 维本体状态、10 维动作及语言指令。主任务为黄色方块叠放到橙色方块上；其他颜色组合属于扩展任务。[数据集与数据卡](https://huggingface.co/datasets/y-s-h/Motrix_Robochanllenge_Colorblock_v3.1)

策略使用 [OpenPI π0.5 后训练仓库](https://github.com/yang-si-hang/openpi05-post-train)。导出的[训练配置](training/stack_lora_20260929_003952/config.yaml)记录 batch size 33、随机种子 42、计划 25,000 step、峰值学习率 2.5×10⁻⁵、1,000 step warmup、动作预测长度 20 和执行长度 10，采用 LoRA 变体及 bfloat16。[loss CSV](training/stack_lora_20260929_003952/loss.csv)覆盖 step 0–18,000，采样值从 **0.2381** 降至 **0.001623**。评测通过 OpenPI socket 服务闭环执行动作，保存逐轮结果与双视角视频。

**评测结果。** 两组评测均为 20 轮初始场景，分别复用训练集种子和使用未见种子。未见种子 `20260928`–`20260947` 与发布训练集的 105 条采集日志无重合。

| 指标 | 训练集种子，checkpoint `16000` | 未见种子，checkpoint `18000` |
| --- | ---: | ---: |
| 任务成功 | 0/20（0%） | 3/20（15%） |
| 抓取阶段成功 | 20/20（100%） | 19/20（95%） |
| 稳定放置成功 | 0/20（0%） | 3/20（15%） |
| 平均代理得分 | 3.45/10 | 4.25/10 |
| 总分 | 69/200 | 85/200 |
| 平均仿真时长 | 70.64 秒 | 62.25 秒 |

[训练集种子评测](https://huggingface.co/datasets/y-s-h/Motrix_Robochanllenge_Colorblock_v3.1/tree/main/evaluation/train20)和[未见种子评测](https://huggingface.co/datasets/y-s-h/Motrix_Robochanllenge_Colorblock_v3.1/tree/main/evaluation/18000_heldout20)均已上传，包含汇总、20 轮逐轮记录和双视角 LeRobot 视频。未见种子评测成功回合对应种子 `20260929`、`20260942`、`20260944`。

**结论。** checkpoint `18000` 在未见种子中完成 3 次叠放；17 次失败中，16 次完成抓取但未稳定放置，1 次未完成抓取（种子 `20260938`）。两组评测使用不同 checkpoint，结果分别报告。得分由 MotrixSim 的阶段判定计算；评分实现见[评测说明](docs/robochallenge_checkpoint_scoring_implementation.md)。

训练轮次需要继续增加并评测。

| 补充说明 | 内容 |
| --- | --- |
| 模型权重 | [LoRA checkpoint `16000`](https://huggingface.co/y-s-h/Motrix_Robochallenge_Colorblock/tree/main/lora/16000) 与[未见种子评测使用的 `18000`](https://huggingface.co/y-s-h/Motrix_Robochallenge_Colorblock/tree/main/lora/18000) 均已上传 |
| 失败案例分析 | 模型动作块预测误差使后续观测逐渐偏离训练数据分布，进而导致后续操作失败 |

## 交付物索引

| 交付物 | 位置与状态 |
| --- | --- |
| 源码与复现命令 | 本仓库 `scripts/reproduce.sh`；[OpenPI 训练与推理仓库](https://github.com/yang-si-hang/openpi05-post-train) |
| 合成数据与数据卡 | [Hugging Face 数据集及数据卡](https://huggingface.co/datasets/y-s-h/Motrix_Robochanllenge_Colorblock_v3.1)；数据卡为该页面的 `README.md`。 |
| 训练参数与 loss | [导出的训练配置](training/stack_lora_20260929_003952/config.yaml)、[loss CSV](training/stack_lora_20260929_003952/loss.csv)。 |
| 模型 checkpoint | [LoRA checkpoint `16000`](https://huggingface.co/y-s-h/Motrix_Robochallenge_Colorblock/tree/main/lora/16000) 与 [`18000`](https://huggingface.co/y-s-h/Motrix_Robochallenge_Colorblock/tree/main/lora/18000) 已上传至同一模型仓库 |
| 20 轮训练集种子评测 | [训练集种子评测目录](https://huggingface.co/datasets/y-s-h/Motrix_Robochanllenge_Colorblock_v3.1/tree/main/evaluation/train20) |
| 20 轮未见种子评测 | [公开评测目录](https://huggingface.co/datasets/y-s-h/Motrix_Robochanllenge_Colorblock_v3.1/tree/main/evaluation/18000_heldout20)；本地对应 `artifacts/score_18000_heldout20/`，成功 3/20，平均得分 4.25/10 |
| 方法与结果总结 | 本 README 上方“方法与结果总结”章节。 |
| Demo 录屏 | [训练集种子 Demo 视频](https://huggingface.co/datasets/y-s-h/Motrix_Robochanllenge_Colorblock_v3.1/tree/main/evaluation/train20/lerobot_dataset/videos)，包含固定视角和腕部视角 |

训练记录来自 W&B run `stack_lora_20260929_003952`。CSV 包含 step 0–18000、每 200 step 一个 loss 采样点，以及 W&B 导出的 MIN/MAX 列。[模型仓库 `lora/16000/`](https://huggingface.co/y-s-h/Motrix_Robochallenge_Colorblock/tree/main/lora/16000) 已包含 `params/`、`train_state/` 和 `assets/color_stack_v3.1_filter_vid/norm_stats.json`。未见种子评测对应的 [`18000` checkpoint](https://huggingface.co/y-s-h/Motrix_Robochallenge_Colorblock/tree/main/lora/18000) 已上传至同一模型仓库。训练源码 Git commit 为 `38af0a45f876878719053b895be87115f03695f1`。

## 一键复现入口

从仓库根目录依次运行；每条命令完成一个阶段，`setup` 只需执行一次：

```bash
bash scripts/reproduce.sh setup
bash scripts/reproduce.sh smoke
EPISODES=100 bash scripts/reproduce.sh collect
```

训练使用上述数据集、[已导出的配置](training/stack_lora_20260929_003952/config.yaml)和 [OpenPI 仓库](https://github.com/yang-si-hang/openpi05-post-train)。评测前在服务端加载与 `CHECKPOINT_LABEL` 对应的权重并启动 OpenPI socket 服务，随后运行：

```bash
CHECKPOINT_LABEL=18000 EVAL_SEED=20260928 EPISODES=20 \
  EVAL_OUTPUT=artifacts/score_18000_heldout20_reproduction \
  bash scripts/reproduce.sh eval
```

`eval` 默认连接 `127.0.0.1:8000`，可用 `OPENPI_HOST`、`OPENPI_PORT` 覆盖。输出目录须为空；复跑时使用新的 `EVAL_OUTPUT`。`CHECKPOINT_LABEL` 写入结果元数据，服务端加载对应权重。

## 1. 环境与仿真场景

```bash
python3.10 -m venv .venv
./.venv/bin/python -m pip install -r requirements-sim.txt
./.venv/bin/python scripts/check_motrixsim.py
./.venv/bin/python scripts/build_scene.py
./.venv/bin/python scripts/check_scene.py
```

`build_scene.py` 根据 `configs/scene.yaml` 生成可加载的 `assets/stack_color_blocks.xml`；`check_scene.py` 检查资产、动力学、控制和两路相机。仓库已包含所需资产；如需重新获取 Menagerie 模型，可运行 `bash scripts/fetch_assets.sh`。

| 场景部分 | 配置与场景文件 | 资产文件 |
| --- | --- | --- |
| UR5e 机械臂、Robotiq 2F85 夹爪 | `configs/scene.yaml` 的 `robot`、`assets/stack_color_blocks.xml` | `assets/vendor/menagerie/universal_robots_ur5e/ur5e.xml`、`assets/vendor/menagerie/robotiq_2f85/2f85.xml` 及各自 `assets/` 网格 |
| 桌面、黄色/橙色/天蓝色方块、四周色带 | `configs/scene.yaml` 的 `table`、`block`、`stickers`；由 `scripts/build_scene.py` 写入 `assets/stack_color_blocks.xml` | 几何体由场景 XML 定义，无单独模型文件 |
| 全局和腕部 RGB 相机、灯光 | `configs/scene.yaml` 的 `global_camera`、`wrist_camera`、`lighting`、`headlight`、`lighting_randomization` | 相机与灯光由场景 XML 定义 |
| 办公室背景（桌椅、盆栽） | `configs/scene.yaml` 的 `background`；背景仅用于视觉 | `assets/vendor/office/` 下的 OBJ、GLTF、纹理；来源见 `assets/vendor/office/ASSET_MANIFEST.md` |

场景运行示例：

```bash
./.venv/bin/python scripts/run_scene.py --headless --task color_block_stack \
  --pick-color yellow --place-color orange --seed 7 \
  --output artifacts/stack_yellow_on_orange
# 有图形桌面时也可用 --gui 查看，不写入 LeRobot 数据集
./.venv/bin/python scripts/run_scene.py --gui --task color_block_stack \
  --pick-color yellow --place-color orange
```

`--headless` 用于离屏运行并录制，`--gui` 用于交互查看；`--task` 当前只支持 `color_block_stack`。`--pick-color` 和 `--place-color` 分别指定被抓取块与目标块，可选 `yellow`、`orange`、`sky_blue`，且两者不能相同。`--seed` 控制方块随机位置和朝向；`--fixed-blocks` 使用配置中的固定位置。`--max-steps` 是仿真步数上限，默认 90000；`--warmup-frames` 是起始姿态保持并录制的帧数，默认 10。`--randomize-background`、`--randomize-lighting` 分别随机化背景位置和光照；`--ssgi` / `--no-ssgi` 覆盖配置中的屏幕空间全局光照开关。`--output` 为结果目录，默认 `artifacts/stack_color_blocks`；离屏运行在该目录下写入 `run.json`、`trace.json` 和 `lerobot_dataset/`。

## 2. 合成数据采集

采集脚本在随机初始方块位姿和光照下运行脚本化专家轨迹，只将成功且通过数据检查的 episode 保存为 LeRobot 数据集。先做小规模试采，再采集目标数量：

```bash
./.venv/bin/python scripts/collect_color_stack.py --seed 20260926 \
  --pick-color yellow --place-color orange --episodes 2 \
  --max-attempts 40 --output data/color_stack_smoke
./.venv/bin/python scripts/collect_color_stack.py --seed 20260926 \
  --pick-color yellow --place-color orange --episodes 100 \
  --max-attempts 1000 --output data/color_stack_v2
```

`--episodes` 是所需成功回合数，`--max-attempts` 是最多尝试次数；`--seed` 是生成各次尝试种子的主种子。颜色参数含义同场景运行命令。`--max-steps` 默认 90000，`--warmup-frames` 默认 10；`--randomize-background` 可增加背景变化，`--lighting-attempts` 可覆盖光照采样重试次数。默认两路 RGB 以图像特征保存；`--save-video` 改为视频特征，需要可用的 H.264 NVENC 编码器，`--video-gop`、`--video-bframes`、`--video-qp`、`--video-preset` 控制视频编码。

`--output` 本身就是 LeRobot 数据集路径（示例为 `data/color_stack_v2/`），包含 20 Hz 的全局及腕部 RGB、`observation.state`、`action` 和语言任务。其同级目录生成 `color_stack_v2_attempts.jsonl`、`color_stack_v2_episodes.jsonl`、`color_stack_v2_verification.json`，分别记录所有尝试、成功回合元数据和回读验证。输出目录必须为空；重复采集请换新路径。数据格式和采集细节见 [场景说明](README_SCENE.md)。

已发布的[训练数据集](https://huggingface.co/datasets/y-s-h/Motrix_Robochanllenge_Colorblock_v3.1/tree/main/train)位于 Hugging Face 数据集仓库的 `train/` 下。其中 `train/meta/`、`train/data/`、`train/videos/` 对应本地 `data/color_stack_v3.1_filter_vid/`；`train/color_stack_v3.1_filter_vid_episodes.jsonl` 对应本地数据集目录同级的逐回合采集日志。运行下文训练集种子评测时，请将数据集目录及日志放回上述本地位置。

## 3. 策略评测与结果

评测需要已启动的 OpenPI socket 推理服务及对应的模型 checkpoint；安装客户端依赖：

```bash
./.venv/bin/python -m pip install -r requirements-inference.txt
```

在 OpenPI 服务端启动自己的 checkpoint（服务命令及环境示例见 [场景说明](README_SCENE.md)），确保本机可通过 `--host`、`--port` 访问。评测分为以下两组，分别保存结果。

### 训练集种子：检查训练效果

从训练数据日志中抽取 20 个黄色叠橙色回合，复用其种子、初始 TCP 位姿与光照。数据集须至少有 20 个匹配回合：

```bash
./.venv/bin/python scripts/score_robochallenge_checkpoint.py \
  --host 127.0.0.1 --port 8000 --checkpoint 16000 \
  --training-dataset data/color_stack_v3.1_filter_vid --selection-seed 20260928 \
  --episodes 20 --max-steps 150 --output artifacts/score_16000_train20 --save-video
```

`--training-dataset` 指向 LeRobot 数据集，脚本还读取其同级 `<数据集名>_episodes.jsonl`；`--selection-seed` 固定抽样顺序。该组用于观察已见过的场景种子上的训练效果，不能当作未见种子的泛化结果。

20 轮训练集种子评测的[汇总、逐轮记录和双视角视频](https://huggingface.co/datasets/y-s-h/Motrix_Robochanllenge_Colorblock_v3.1/tree/main/evaluation/train20)已上传至数据集仓库 `evaluation/train20/` 目录，本地对应 `artifacts/score_color_stack_v3.1_filter_train20/`。汇总结果：checkpoint 标签 `16000`、任务成功率 **0/20（0%）**、平均得分 **3.45/10**、抓取阶段成功率 **100%**、放置阶段成功率 **0%**、平均仿真时长 **70.64 秒**。双视角 LeRobot 视频用于 Demo 展示。

### 非训练集种子：检查泛化能力

不传 `--training-dataset`，用 `--seed` 指定起始种子，连续运行 20 轮。已完成的评测使用 `20260928` 至 `20260947`；初始 TCP 位姿取采集脚本默认值。也可用 `--initial-position X Y Z`、`--initial-quaternion X Y Z W` 覆盖（世界坐标、米、xyzw 顺序）：

```bash
./.venv/bin/python scripts/score_robochallenge_checkpoint.py \
  --host 127.0.0.1 --port 8000 --checkpoint 18000 \
  --seed 20260928 --episodes 20 --max-steps 150 \
  --output artifacts/score_18000_heldout20_reproduction --save-video
```

[公开评测目录](https://huggingface.co/datasets/y-s-h/Motrix_Robochanllenge_Colorblock_v3.1/tree/main/evaluation/18000_heldout20)中的 [summary.json](https://huggingface.co/datasets/y-s-h/Motrix_Robochanllenge_Colorblock_v3.1/blob/main/evaluation/18000_heldout20/summary.json) 记录成功率 **15%**、平均得分 **4.25/10**、抓取成功率 **95%**、放置成功率 **15%** 和平均仿真时长 **62.25 秒**。`selection.json` 保存 20 个种子；`episode_*/` 保存逐轮结果；`lerobot_dataset/videos/` 保存固定视角与腕部视角视频。本地对应 `artifacts/score_18000_heldout20/`。

两组共用参数：`--episodes` 为评测轮数，`--max-steps` 为每轮最多执行的策略步数，`--warmup-frames` 默认 10。`--checkpoint` 仅是写入报告的标签，需自行核对服务端实际加载的权重。`--save-video` 可把评测轨迹保存到结果目录的 `lerobot_dataset/`，并可用视频编码参数调整输出。各自的结果目录中，`selection.json` 记录选中的种子和初始位姿，`episode_*/result.json` 与 `episode_*/steps.json` 记录单轮判定、动作和耗时，`summary.json` 汇总成功率、平均得分、平均仿真时长及抓取/放置阶段成功率。输出目录必须为空。评分采用抓取与放置各 5 分、每次重试扣 0.5 分的 **MotrixSim 本地代理判定**；物理阈值和初始场景并非 RoboChallenge 官方真机评测。评分口径见 [评测说明](docs/robochallenge_checkpoint_scoring_implementation.md)。

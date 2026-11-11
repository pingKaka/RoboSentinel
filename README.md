# RoboSentinel

RoboSentinel 是一个面向机器人操作过程的实时故障检测系统。它接收双相机图像和可选机器人状态，持续跟踪任务对象，并通过规则确认异常。

系统是独立的观测侧模块：检测接口不会生成或执行机器人动作，规则报告就是最终输出。

## 核心流程

```text
任务描述 + 初始双相机场景
          |
          v
Qwen3-VL 生成跟踪目标和可解释规则
          |
          v
GroundingDINO 初始化检测 + SAM2.1 在线分割跟踪
          |
          v
标准特征层 + 声明式规则引擎
```

每个同步观测都会进入跟踪器和规则引擎。自动规则或操作者人工报告产生的 `RuleReport` 会直接作为故障检测结果。

## 当前能力

- 使用 Qwen3-VL 根据任务和初始场景生成跟踪目标与规则包。
- 使用 GroundingDINO 和 SAM2.1 处理双相机图像流。
- 用受限 JSON Schema 和简单算子执行可审计规则，不执行 VLM 生成的任意代码。
- 根据任务选择特征并记录选择原因，首版覆盖目标丢失、抓取失败和放置失败。
- 输出版本化标准特征：目标可见性、归一化几何关系、TCP 位置和夹爪位置。
- 通过 HTTP/WebSocket 服务接入视频回放、相机流或 UR5e 遥操作。
- 在服务端和客户端分别记录规则报告与可选原始观测。
- 支持操作者人工报告故障，统一输出为确认的规则发现。

当前实现只负责检测，不生成机器人动作或恢复策略。规则阈值自适应和成功轨迹拟合尚未加入。

## 仓库结构

```text
RoboSentinel/
├── cfg.yaml                    # 日常配置和子配置入口
├── config/                     # 检测高级配置、遥操作硬件配置
├── failure_detection/          # 流水线、规则 DSL、运行时和网络传输
├── scripts/                    # 服务、回放和相机客户端入口
├── training/                   # 微调数据处理、校验和远程训练
├── ur5e_video_teleop/          # UR5e 双相机遥操作客户端
├── my_backend.py               # 真实模型后端装配入口
└── project_config.py           # 分层配置加载与路径解析
```

标准特征字段、单位和扩展约束见 [标准特征规范](docs/standard_features_zh.md)，规则格式与执行语义见 [可解释故障规则规范](docs/rule_schema_zh.md)。规则只能读取标准特征上下文，不能直接读取原始图像或机器人状态字典。

模型权重、视频、训练数据和运行结果不进入 Git，默认分别放在 `checkpoint/`、`video/`、`training/data/`、`ur5e_video_teleop/videos/` 和 `result/`。

## 环境准备

服务端推荐 Python 3.10。`environment.yml` 会安装 `torch>=2.5` 和 `transformers==5.16.1`；如默认 PyTorch 与主机 CUDA 不匹配，请在环境创建后按 PyTorch 官方说明替换为对应构建。

```bash
conda env create -f environment.yml
conda activate robosentinel
```

准备以下模型目录：

```text
checkpoint/
├── Qwen3-VL-8B-Instruct/
├── GroundingDINO-tiny/
└── SAM2.1-hiera-small/
```

仅运行遥操作客户端时，在原有机器人环境中安装：

```bash
pip install -r requirements-client.txt
pip install --no-deps -e ./ur5e_video_teleop
```

遥操作还依赖同级目录中的 UR5e 机器人与主臂驱动，详见 [UR5e Video Teleop](ur5e_video_teleop/README.md)。

## 配置

`cfg.yaml` 是统一入口，并按顺序合并两个子配置：

- `config/detection.yaml`：网络、模型后端、规则、回放和输出参数。
- `config/teleop.yaml`：UR5e、Dynamixel、夹爪和 RealSense 参数。
- `cfg.yaml`：任务、模型路径、客户端地址及本次运行覆盖项。

后加载的配置覆盖先加载的同名字段。运行前先检查配置和模型路径：

```bash
python failure_detection/run_pipeline.py
```

## 运行

启动检测服务：

```bash
python scripts/serve.py
```

使用成对视频回放；视频必须具有相同帧率并从同一事件开始：

```bash
python scripts/replay_detection.py \
  --external video/exterior.mp4 \
  --wrist video/wrist.mp4
```

使用两路本地相机：

```bash
python scripts/stream_client.py
```

启用 UR5e 遥操作检测：

```bash
ur5e-video-teleop --sentinel-server http://SERVER_IP:8000
```

## 开发与验证

运行单元测试：

```bash
python -m unittest discover -s failure_detection/tests -v
```

使用单张图像检查真实 GPU 后端：

```bash
python scripts/check_backend.py --image /path/to/image.jpg --mode full
```

使用 `video/` 中的成对视频检查标准视觉特征收集：

```bash
python scripts/collect_video_features.py --video-dir video
```

微调数据格式、准备、校验和远程训练命令见 [training/README.md](training/README.md)。

## 输出与版本控制

以下内容均为本地资产或可再生成产物，已由 `.gitignore` 排除：

- 模型权重和 Hugging Face 缓存。
- 采集视频、回放视频和训练数据。
- 服务端/客户端运行结果及标注帧。
- Python 缓存、构建元数据和训练输出。

提交代码前可用 `git status --short --ignored` 检查边界，避免把权重、数据集或实验结果加入版本库。

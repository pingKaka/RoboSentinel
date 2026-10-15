# 从零启动 RoboSentinel

这份教程按“先安装、再回放、最后真机”的顺序操作。你最终会得到一个本地窗口：双相机原图、模型跟踪画面、任务和目标信息、检测状态、中文故障分析。服务器负责三个模型的计算，本地负责遥操作、采集和显示。

已安装好的日常启动见第 9 节；每个参数的含义见 [配置说明](config_reference_zh.md)，规则和 prompt 调试见 [调试教程](rules_zh.md)。

## 1. 准备什么

| 位置 | 需要准备 |
| --- | --- |
| 本地电脑 | Ubuntu 图形桌面、Conda、Git；主臂、UR 机器人、夹爪、两台 RealSense；能访问机器人和服务器 |
| GPU 服务器 | NVIDIA GPU 和可用驱动、Conda、Git、足够的权重/日志存储空间；建议从 24GB 显存等级开始评估 |
| 模型 | Qwen3-VL-8B-Instruct、GroundingDINO-tiny、SAM2.1-hiera-small 的完整 Transformers 权重目录 |
| 测试素材 | 一对同帧率、时间对应的外部/腕部视频；仓库不附带视频和权重 |

8GB GPU 无法承载当前三模型同时驻留的完整配置；24GB 也应以实际加载和任务占用为准。先用 `nvidia-smi` 确认服务器没有其他进程占满显存。

本文假设你已有 Conda；若 `conda` 不存在，先按 Miniconda 官方说明安装，并重新打开终端。示例中的 `SERVER_IP`、`user` 要替换成你的实际值。两台机器都要有同一版代码，修改未推送到 Git 仓库时，服务器 `git pull` 无法获取。

**检测不控制机器人启停。** 启动遥操作程序后，即使还没按空格，机器人也可能跟随主臂。硬件标定、急停和主从臂对齐必须先完成，不能用这套故障检测代替机器人安全控制。

## 2. 取得代码

本地终端执行（已存在仓库则进入原目录，不要重复克隆）：

```bash
mkdir -p ~/ur_data_collection
cd ~/ur_data_collection
git clone https://github.com/pingKaka/RoboSentinel.git
cd RoboSentinel
```

服务器终端执行：

```bash
mkdir -p ~/ur_data_collection
cd ~/ur_data_collection
git clone https://github.com/pingKaka/RoboSentinel.git
cd RoboSentinel
```

后文命令默认在各自机器的 `~/ur_data_collection/RoboSentinel` 下执行。当前这台开发机的现有目录是 `/home/kakaping/ur_data_collection/RoboSentinel`。

## 3. 安装服务器环境

只在服务器安装推理环境，不要把根目录 `requirements.txt` 装到已有的本地遥操作环境里，它包含无界面的 OpenCV。

```bash
conda create -n robosentinel python=3.10 -y
conda activate robosentinel
nvidia-smi
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU unavailable')"
```

应显示 `CUDA: True` 和显卡名称。这里使用 CUDA 12.8 的 PyTorch 包，服务器驱动须支持；若驱动太旧，应由服务器管理员升级驱动或选择兼容的 PyTorch 包，不要在 CUDA 检查失败时继续启动模型。项目固定的 Transformers 版本在 `requirements.txt` 中，不建议随意换版本。

### 下载或复制权重

没有权重时，在服务器项目目录执行：

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir checkpoint/Qwen3-VL-8B-Instruct
hf download IDEA-Research/grounding-dino-tiny --local-dir checkpoint/GroundingDINO-tiny
hf download facebook/sam2.1-hiera-small --local-dir checkpoint/SAM2.1-hiera-small
```

`hf` 来自依赖中的 huggingface-hub；若命令不存在，执行 `python -m pip install -U huggingface-hub`。下载需要能访问 Hugging Face，下载失败先解决网络问题，不要留下半套权重直接运行。

如果本地已经有验证过的完整 `checkpoint/`，可以不用重新下载，从本地项目目录上传：

```bash
rsync -az --progress checkpoint/ user@SERVER_IP:~/ur_data_collection/RoboSentinel/checkpoint/
```

目录应包含配置、处理器/分词器文件、完整模型权重，不是只有一个 `.pth` 文件。SAM 使用 Transformers 的 `Sam2VideoModel` 格式；不要把其他实现的 SAM 权重直接放进来。

服务器 `cfg.yaml` 确认以下字段（修改已有字段，不要重复添加同名 YAML 段）：

```yaml
models:
  vlm: checkpoint/Qwen3-VL-8B-Instruct
  grounding_dino: checkpoint/GroundingDINO-tiny
  sam: checkpoint/SAM2.1-hiera-small
```

执行 `python failure_detection/run_pipeline.py` 检查配置和路径。目录可用只说明文件位置存在，不代表模型已经推理成功。

## 4. 安装本地环境

### 已有可用的 ur_data 环境

你当前这台电脑优先用这一条路径，保留已安装的驱动和标定：

```bash
conda activate ur_data
cd ~/ur_data_collection/RoboSentinel
python -m pip install -r requirements-client.txt
python -m pip install -e ./ur5e_video_teleop
ur5e-video-teleop --help
```

`--help` 只检查入口，不连接机器人。不要同时运行旧版遥操作和此程序，以免重复连接相机和机器人。

### 全新的本地电脑

本项目复用外部 LeRobot UR 驱动，仓库本身不包含所有驱动。按原驱动项目所指定的 LeRobot 版本安装：

```bash
conda create -n ur_data python=3.10 -y
conda activate ur_data
cd ~/ur_data_collection
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
python -m pip install -e .
cd ~/ur_data_collection
git clone https://github.com/scy-v/lerobot_ur5e_isoteleop.git
cd lerobot_ur5e_isoteleop
python -m pip install -r requirements.txt
cd ~/ur_data_collection/RoboSentinel
python -m pip install -r requirements-client.txt
python -m pip install -e ./ur5e_video_teleop
```

目录排列应是三个兄弟目录：`RoboSentinel/`、`lerobot/`、`lerobot_ur5e_isoteleop/`。外部驱动会安装 RTDE、RealSense、Pinocchio、Dynamixel 等依赖。现有设备若依赖本地修改过的驱动，要保留该版本及其配置，不能假定最新远程版本与它完全相同。

### 中文界面与硬件检查

Ubuntu 安装字体和常用图形库：

```bash
sudo apt-get update
sudo apt-get install -y fonts-noto-cjk libgl1 libglib2.0-0
```

本地必须使用有 GUI 的 OpenCV。若安装混乱或提示 `imshow` 未实现，在本地 `ur_data` 环境修复：

```bash
python -m pip uninstall -y opencv-python-headless opencv-python opencv-contrib-python opencv-contrib-python-headless
python -m pip install opencv-python
```

在本地桌面终端运行，不要在无图形桌面的普通 SSH 会话中启动窗口。Wayland 下键盘监听若无响应，切换可用的 X11 桌面会话。

检查依赖（不发送机器人动作）：

```bash
python -c "import cv2, PIL, pyrealsense2, pinocchio, rtde_receive, dynamixel_sdk; print('imports OK')"
ur5e-video-teleop --help
```

硬件首次配置请完成外部驱动的 [设备配置与标定教程](https://github.com/scy-v/lerobot_ur5e_isoteleop/blob/main/README_zh.MD)：机器人 Remote Control、网络、相机序列号、主臂和夹爪串口权限、主从臂硬件零点偏置及关节周期偏置。USB/udev 权限按该设备实际安装规则设置，不要仅复制别人的串口名称。此步骤需要真实设备，安装 Python 包不能替代标定。

## 5. 配置本次实验

两台机器各有自己的根目录 `cfg.yaml`。服务器设置模型路径、规则和服务端输出；本地设置任务、网络、硬件和界面。

本地首先修改：

| 字段 | 填什么 |
| --- | --- |
| `task.id` | 本次任务的简短编号 |
| `task.description` | 哪个物体、做什么、放到哪里；由客户端发给服务器 |
| `client.server` | 内网 `http://SERVER_IP:8000`，或隧道 `http://127.0.0.1:8000` |
| `client.fps` | 初次保留 5；这是最大发送频率，不是模型实际速度保证 |
| `client.record_observations` | 保留 true，记录发送样本以便复现 |
| `video_teleop.preview` | true，打开可视化 |
| `video_teleop.sentinel.enabled` | true，启用检测 |
| `video_teleop.robot.ip` | 实际 UR 机器人地址，和服务器地址不是一回事 |
| `video_teleop.teleop.dynamixel_config` | 主臂串口、关节方向和实际标定值 |
| `video_teleop.robot.gripper_port` | 实际夹爪串口 |
| `video_teleop.cameras` | 实际腕部/外部相机序列号和支持的分辨率 |

例如任务文字可写：`Pick the white-capped reagent bottle and place it in the far-right hole of the large test-tube rack.`
不用自己填写跟踪目标 `name`，它由 VLM 从图像和任务中生成。任务文字写了“最右孔”不等于系统已经具备插错孔判定能力。

保留已标定的力控增益、参考系、关节偏置；不要为了测试视觉修改这些硬件参数。URDF 路径由机器人驱动解析。所有参数解释见 [配置说明](config_reference_zh.md)。改配置后重启对应程序，不会热更新。

## 6. 启动服务器并连通网络

服务器终端 A：

```bash
conda activate robosentinel
cd ~/ur_data_collection/RoboSentinel
python scripts/serve.py
```

等待 `READY` 和 `Application startup complete`，首次加载可能需要几分钟。三模型成功驻留后，服务器不退出就无需每轮重新加载权重。

服务器终端 B：

```bash
curl http://127.0.0.1:8000/health
```

应看到 `models_ready: true`、三项 `resident_models`，无人测试时 `active_tasks: 0`。

本地测试连接：

```bash
curl http://SERVER_IP:8000/health
```

如果只允许 SSH，另开一个本地终端保持以下命令运行：

```bash
ssh -N -L 8000:127.0.0.1:8000 user@SERVER_IP
```

随后把本地 `client.server` 改成 `http://127.0.0.1:8000`，用同一地址检查 `/health`。若本地 8000 被占用，可把 `-L` 第一个端口改为 18000，本地配置也对应改为 18000。服务没有鉴权/TLS，只在可信内网或 SSH 隧道使用，不直接暴露公网。

## 7. 先回放测试，再连接真机

本地 `cfg.yaml` 设置：

```yaml
replay:
  observations: null
  external: /你的路径/外部相机.mp4
  wrist: /你的路径/腕部相机.mp4
  seconds: 30
  simulate_fault_at: null
```

视频要包含任务目标、同帧率并时间对应。执行：

```bash
conda activate ur_data
python scripts/replay_detection.py
```

看到 `STREAMING`、返回结果、最后 `PASS` 代表链路跑通；还应检查结果里的目标和框是否正确。回放是命令行测试，不会打开遥操作窗口，也不操作机器人。

想验证故障分析，把 `replay.simulate_fault_at` 改为 `3`，使用超过 3 秒的视频并保留 `controller_fault` 规则，再运行一次；应出现 `DIAGNOSIS`。之后改回 null。这是模拟控制器故障，不证明视觉检测准确率。

没有视频可跳过回放，完成硬件检查后进入下一节；不要把“回放通过”当作真机标定通过。默认同时只允许一个任务，结束回放后再开遥操作。

## 8. 启动遥操作并完成一轮实验

先打开机器人 Remote Control，按原标定流程对齐主从臂，确认急停可用、工作区安全。然后在本地桌面终端运行：

```bash
conda activate ur_data
cd ~/ur_data_collection/RoboSentinel
ur5e-video-teleop
```

**这个命令启动的就是实际遥操作，机器人可能立即跟随；空格不是运动许可键。**

| 顺序 | 你的操作 | 预期现象 |
| --- | --- | --- |
| 1 | 暂不按键，保持主臂稳定 | 双相机原图可见，状态“未开始检测”，右侧显示当前任务 |
| 2 | 按一次空格 | 同时开始录像和检测；状态为初始化，首对图像发往服务器 |
| 3 | 等待初始化完成 | VLM 自动生成目标，DINO 定位、SAM 初始化；状态转为“检测中” |
| 4 | 执行任务 | 下方双视角更新框和分割轮廓，右侧显示各目标可见性 |
| 5 | 异常时按 F，或由规则自动触发 | 人工报告立即显示分析中；自动异常随服务器结果反馈，最终显示中文分析与建议 |
| 6 | 查看分析后按 R | 清除界面故障并重新允许自动报警；异常仍存在可能再次报警 |
| 7 | 再按空格 | 停止检测和录像，但遥操作仍运行 |
| 8 | 需要下一轮时再按空格 | 新建任务，重新理解场景；模型权重不重新加载 |
| 9 | 结束实验按 Esc | 退出程序、关闭设备连接和记录 |

按住 F 或 R 不会重复触发。手动分析尚未结束时再次按 F/R 会提示等待。R 不创建新任务、不重新选择目标，也不代表物理故障已经排除；明显换场景时用空格停止再开始。修改任务文字需重启客户端。

### 如何读界面

- 左上：双相机当前原图；左中：已处理帧的目标框和 SAM 简化分割轮廓。轮廓不等于精确完整稠密掩码。
- 左下：故障分析与恢复建议，鼠标在此区域滚轮查看长文本。
- 右侧：任务 ID/描述、VLM 生成的目标名称/角色、每个目标的外部/腕部可见性、连接和会话、录像、返回数量、丢弃旧观测数、耗时、规则、控制器状态、故障来源和结果目录。鼠标在右侧滚轮查看全部详情。
- 已处理帧时间和“距当前”说明结果有多旧；VLM 分析时跟踪回传可能暂停，但原图和遥操作继续。未返回的数据不会被伪装成实时模型结果。
- `unknown/未知` 表示证据不足，不代表正常；控制器状态缺失时也显示未知。

## 9. 以后每天怎么启动

无需每天安装依赖或下载模型。通常只需要三个终端：

1. 服务器：激活 `robosentinel`，进入项目目录，`python scripts/serve.py`，等 READY。
2. 本地可选隧道：`ssh -N -L 8000:127.0.0.1:8000 user@SERVER_IP`。
3. 本地：激活 `ur_data`，进入项目目录，`ur5e-video-teleop`。

随后就是：**空格开始 → 等检测中 → 操作 → F 报故障 → 看分析 → R 恢复 → 空格结束 → Esc 退出。**
结束本地程序不会卸载服务器模型。全部实验结束后可在服务器终端 Ctrl+C 释放显存。更新代码时，两端先停止程序，`git pull --ff-only`，按各自环境更新依赖再重启；有本地配置修改时先备份并处理 Git 冲突，不要强制覆盖标定。

## 10. 得到了什么文件

| 内容 | 默认位置 |
| --- | --- |
| 服务端结构化结果 | `result/server/<会话ID>/results.jsonl` |
| 本地结构化结果 | `result/client/teleop/<时间_随机ID>/run_001/results.jsonl` |
| 本地发送的图像/机器人状态 | 同目录 `observations.jsonl`（record_observations 为 true 时） |
| 每轮双视角 MP4 | `ur5e_video_teleop/videos/` |
| 服务端带框 JPG（可选） | 开启 `outputs.save_annotated_frames` 后保存在会话目录 |

每按空格开始新一轮，目录会变为 `run_002` 等；右侧会话 ID 用于对照服务端文件。服务器先保存成功再回传，本地再保存。初始化结果含目标规划，后续结果含跟踪、轮廓、规则和可选诊断。无新故障时 `diagnosis: null` 正常。退出时不会自动写整场实验总结。

记录的是发送样本，不是机器人控制循环的全部帧；MP4 是原始相机录像，不是窗口录屏。手动故障和 R 恢复会写入相应记录。现有回放默认不重新执行历史手动触发/恢复命令，也不会执行机器人动作。

要复现图像与状态，将 `replay.observations` 指向对应 `observations.jsonl` 后运行回放脚本；切回视频时设回 null。文件可能很大，实验后自行归档和清理。

## 11. 遇到问题先查这里

| 问题 | 排查方法 |
| --- | --- |
| 服务器没有 READY | 看启动报错；检查 CUDA、权重是否完整、模型类型和显存；不要重复启动多个进程 |
| health 本机可访问，本地不可访问 | 检查 IP、端口、防火墙或 SSH 隧道；0.0.0.0 是监听地址，不是客户端目标地址 |
| 409 Task limit | 结束另一个回放/遥操作任务并等待清理，再开始；不要直接删除他人任务 |
| 503 / 模型输出解析失败 | 看服务器终端；检查权重、prompt 输出 JSON、磁盘权限，按调试教程单图排查 |
| 空格后一直初始化 | 初始 VLM 可能较慢；看服务器终端和 GPU 占用，检查目标是否清楚；网络超时默认 600 秒 |
| 没窗口、中文方块、键盘失灵 | 本地桌面、GUI OpenCV、中文字体、X11/pynput 权限；不要在无显示的 SSH 会话运行 |
| 串口或相机连接失败 | 检查权限、序列号、USB、是否有其他程序占用；与模型服务器无关 |
| 检测框错误或报警不合理 | 看任务文字、VLM 目标、双视角可见性；再按 [规则与 prompt 教程](rules_zh.md) 调试 |
| R 后立刻再次故障 | 自动规则仍满足异常条件；恢复键不是关闭规则，也不是物理修复 |
| 画面延迟大 | 看模型处理/请求往返/帧滞后；先降低 client.fps，不要为了视觉延迟改变机器人力控参数 |
| 本地没有 MP4 | 是否按空格、视频目录可写、视频编码器可用；检测 JSONL 与录像是不同文件 |

可在服务器环境运行 `python -m unittest discover -s failure_detection/tests -v` 验证代码；其中界面渲染测试需要可用中文字体。测试通过不等于在你这台机器人、服务器上完成了联调。完整 GPU 服务和真机实验仍需按以上步骤实测。

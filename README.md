# RoboSentinel

RoboSentinel 是一套部署在 GPU 服务器上的机器人实时故障检测系统。它读取外部相机、腕部相机以及可选的 UR5e 状态，只判断任务是否发生故障，不生成机器人动作，也不修改机器人控制程序。

## 系统做什么

一次任务的处理过程如下：

```text
任务文字 + 两路初始图像
        |
        v
Qwen3-VL 选择跟踪目标并编写一条可解释监控规则
        |
        v
GroundingDINO 找到目标，SAM2.1 逐帧输出目标掩码
        |
        v
标准特征层计算可见性、掩码关系、TCP 和夹爪特征
        |
        v
确定性规则引擎逐帧返回 clear / confirmed / unknown
```

VLM只在任务初始化时使用。实时阶段运行的是固定规则，不会逐帧调用 VLM。VLM输出的简洁规则会先被编译、校验，再进入规则引擎；系统不会直接执行 VLM 生成的 Python。

## 仓库结构

```text
RoboSentinel/
├── cfg.yaml                  # 日常任务、模型路径和客户端地址
├── config/
│   ├── detection.yaml       # 服务、模型后端和规则参数
│   ├── teleop.yaml          # UR5e、夹爪和相机参数
│   └── manual_monitoring_plan.example.json
├── failure_detection/       # 跟踪、特征、规则、服务和传输代码
├── prompt/                  # 当前在线提示词
├── scripts/                 # 服务、回放、相机和测试入口
├── docs/                    # 特征、规则、机器人状态专题文档
├── training/                # VLM微调工具，主流程不依赖
└── ur5e_video_teleop/       # UR5e双相机遥操作客户端
```

模型权重、视频和运行结果分别放在 `checkpoint/`、`video/` 和 `result/`，均不提交 Git。

## 一、服务器准备

### 1. 硬件和软件

建议使用 Linux、Python 3.10 和 NVIDIA GPU。Qwen3-VL 默认使用 4-bit 量化，但总显存还要容纳 GroundingDINO、SAM2.1 和中间张量；显存是否足够应以目标服务器实际启动结果为准。

先检查驱动：

```bash
nvidia-smi
```

### 2. 创建环境

```bash
cd /path/to/RoboSentinel
conda env create -f environment.yml
conda activate robosentinel
```

确认 PyTorch 能看到 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

第二行必须是 `True`。如果为 `False`，先安装与服务器驱动匹配的 PyTorch CUDA 构建，再继续。

### 3. 放置模型权重

运行时只读取本地文件，不会自动联网下载。目录必须与 `cfg.yaml` 一致：

```text
checkpoint/
├── Qwen3-VL-8B-Instruct/
├── GroundingDINO-tiny/
└── SAM2.1-hiera-small/
```

服务器可以访问 Hugging Face 时，在已激活的环境中执行：

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir checkpoint/Qwen3-VL-8B-Instruct
hf download IDEA-Research/grounding-dino-tiny \
  --local-dir checkpoint/GroundingDINO-tiny
hf download facebook/sam2.1-hiera-small \
  --local-dir checkpoint/SAM2.1-hiera-small
```

如果提示没有 `hf` 命令，先安装 `huggingface_hub`：

```bash
python -m pip install -U huggingface_hub
```

离线服务器应在联网机器完成下载，再把三个完整目录传到服务器的 `checkpoint/`。不要只传 Git LFS 指针或模型配置文件。

每个目录不能只有配置文件，还必须包含完整权重。需要更换目录时修改：

```yaml
# cfg.yaml
models:
  vlm: checkpoint/Qwen3-VL-8B-Instruct
  grounding_dino: checkpoint/GroundingDINO-tiny
  sam: checkpoint/SAM2.1-hiera-small
```

## 二、配置任务

初次部署通常只需修改 `cfg.yaml`：

```yaml
task:
  id: white_bottle_to_black_cup
  description: "Pick the white-capped reagent bottle and place it in the black cup."

client:
  server: "http://127.0.0.1:8000"
  fps: 10
```

任务描述应明确写出操作对象和目标位置。跨机器运行时，把 `client.server` 改为 GPU 服务器地址，例如 `http://192.168.1.20:8000`。

规则有三种模式，在 `config/detection.yaml` 中设置：

```yaml
stages:
  rules:
    mode: generated  # generated / manual / merged
    manual_plan: null
```

- `generated`：VLM根据任务和初始图像生成规则，推荐先使用。
- `manual`：只使用人工 JSON，不调用 VLM写规则。
- `merged`：同时使用 VLM规则和人工规则。

人工规则写法见 [规则教程](docs/rules_pipeline_tutorial_zh.md)。

## 三、启动前检查

### 1. 检查配置和模型目录

```bash
python failure_detection/run_pipeline.py --config cfg.yaml
```

输出中的三个模型都应为 `"available": true`，最终状态应为 `"configured"`。

### 2. 运行单元测试

```bash
python -m unittest discover -s failure_detection/tests -v
```

这一步不加载真实大模型，主要检查规则、特征、网络协议和机器人状态接口。

### 3. 单独测试 VLM 编写规则

仓库自带五组成对视频。以下命令只读取每组视频第 0 帧，让 VLM生成规则并验证能否编译，不执行故障检测：

```bash
python scripts/test_vlm_rule_generation.py \
  --config cfg.yaml \
  --video-dir video \
  --output-dir result/vlm_rule_test
```

最后应看到 `first_pass`、`final_pass` 和 `total`。详细输入、原始输出及校验错误保存在输出目录。

### 4. 单独检查真实模型后端

准备一张能清楚看到目标的图片：

```bash
python scripts/check_backend.py --config cfg.yaml --image /path/to/image.jpg --mode qwen
python scripts/check_backend.py --config cfg.yaml --image /path/to/image.jpg --mode tracker --target "white bottle"
python scripts/check_backend.py --config cfg.yaml --image /path/to/image.jpg --mode full
```

`qwen` 只测规则生成，`tracker` 只测 GroundingDINO/SAM，`full` 测完整初始化。

## 四、启动服务器

```bash
python scripts/serve.py --config cfg.yaml
```

服务器启动时会把三个模型预加载到 GPU。首次加载可能需要几十秒。不要使用多个 Uvicorn worker，因为每个进程都会重复占用一份显存。

另开终端检查：

```bash
curl http://127.0.0.1:8000/health
```

正常结果应包含：

```json
{
  "status": "ok",
  "models_ready": true,
  "resident_models": ["vlm", "grounding_dino", "sam"]
}
```

服务默认监听 `0.0.0.0:8000`。跨机器使用时需开放服务器 TCP 8000 端口。当前服务本身不提供 TLS 和身份认证，应部署在可信内网，或放在带鉴权的反向代理后面。

## 五、用视频打通完整流程

先用视频验证服务器、网络和三个模型，再连接机器人：

```bash
python scripts/replay_detection.py \
  --server http://127.0.0.1:8000 \
  --external video/exterior_success.mp4 \
  --wrist video/wrist_success.mp4 \
  --description "Pick the white-capped reagent bottle and place it in the black cup" \
  --fps 10 \
  --seconds 30
```

两段视频必须来自同一次操作、起始时刻对应且 FPS 一致。命令末尾出现：

```text
PASS: ... responses; output=...
```

表示 HTTP 建立任务、WebSocket 图像流、VLM规划、目标跟踪、特征提取和规则返回已经打通。`PASS` 只代表系统正常运行，不代表视频一定被判为成功。

纯视频没有 TCP 和夹爪状态。依赖机器人特征的规则会得到 `unknown`，这不是程序错误。要测试这些规则，应回放包含 `robot_state` 的观测记录或连接 UR5e。

结果默认写入：

```text
result/server/   # 服务器规则结果
result/client/   # 客户端响应和可选原始观测
```

## 六、接入实时相机或 UR5e

### 1. 只接两路相机

```bash
python scripts/stream_client.py \
  --server http://SERVER_IP:8000 \
  --external 0 \
  --wrist 2 \
  --description "Pick the bottle and place it in the cup" \
  --fps 10
```

相机编号可在 `config/detection.yaml` 的 `camera_client` 中配置。不要同时让两个程序占用同一相机。

### 2. 接入 UR5e

机器人电脑只需安装轻量客户端，不需要安装三个模型：

```bash
pip install -r requirements-client.txt
pip install --no-deps -e ./ur5e_video_teleop
```

检查 `config/teleop.yaml` 中的 UR5e IP、RealSense 序列号、主臂和夹爪参数，然后运行：

```bash
ur5e-video-teleop --sentinel-server http://SERVER_IP:8000
```

UR5e 状态适配器只向标准特征层提供：

- TCP位置：`robot.tcp_x`、`robot.tcp_y`、`robot.tcp_z`，单位米；
- 夹爪位置：`robot.gripper_position_normalized`，范围 `[0,1]`。

机器人接口与检测核心解耦，具体格式和扩展方法见 [机器人状态接口](docs/robot_state_provider_zh.md) 和 [UR5e客户端说明](ur5e_video_teleop/README.md)。

## 七、常用参数

主要参数位于 `config/detection.yaml`：

| 参数 | 作用 | 初次部署建议 |
|---|---|---|
| `backend.quantize_vlm` | Qwen是否使用4-bit量化 | 显存有限时保持 `true` |
| `backend.vlm_image_max_side` | VLM输入图像最长边 | 默认 `768` |
| `backend.box_threshold` | GroundingDINO框阈值 | 先用 `0.3` |
| `backend.min_mask_pixels` | 小掩码过滤阈值 | 先用 `16` |
| `backend.reseed_frames` | 定期重新检测间隔 | 默认 `60` |
| `runtime.buffer_seconds` | 时序特征历史长度 | 必须覆盖规则最大窗口 |
| `client.jpeg_quality` | 客户端图像压缩质量 | 网络不足时降低 |
| `client.fps` | 发送频率 | 先用 `5-10` FPS |
| `outputs.save_annotated_frames` | 保存审验图像 | 实时运行建议关闭 |

标准特征、单位及掩码关系定义见 [标准特征规范](docs/standard_features_zh.md)。

## 八、常见问题

### `torch.cuda.is_available()` 为 `False`

PyTorch与服务器驱动不匹配。重新安装正确的 CUDA 构建；只安装 CUDA Toolkit 不一定能解决 PyTorch构建错误。

### 启动时显存不足

确认 `quantize_vlm: true`，关闭其他 GPU进程，保持单 worker，并降低 `vlm_image_max_side`。仍不足时需要更大显存或更小的 VLM。

### 配置显示 `missing_models`

检查 `cfg.yaml` 路径是否相对于仓库根目录，模型目录是否完整，以及启动命令是否使用了正确的 `--config`。

### GroundingDINO找不到目标

把任务中的目标写成具体、可见的英文短语，例如 `white-capped reagent bottle`，不要只写 `object`。还可适当降低 `box_threshold`，但阈值过低会增加误检。

### 视频回放一直是 `unknown`

先看规则是否需要 TCP 或夹爪数据。纯 MP4 不含机器人状态；视觉目标丢失、掩码不可用或时间窗口尚未积累完成也会产生 `unknown`。

### 视频回放无法开始

确认外部和腕部视频都能被 OpenCV读取、FPS已写入文件且两者 FPS一致。两路视频必须同时起始。

### 客户端连接不到服务器

先在客户端执行 `curl http://SERVER_IP:8000/health`，再检查防火墙、地址、端口和服务器是否监听 `0.0.0.0`。

## 九、文档入口

- [标准特征规范](docs/standard_features_zh.md)
- [规则格式与执行语义](docs/rule_schema_zh.md)
- [人工规则和 Pipeline 教程](docs/rules_pipeline_tutorial_zh.md)
- [机器人状态接口](docs/robot_state_provider_zh.md)
- [VLM提示词调研与最终复测](docs/vlm_prompt_research_and_retest_2026-11-11_zh.md)
- [训练工具](training/README.md)

## 十、开发检查

提交前运行：

```bash
python -m unittest discover -s failure_detection/tests -v
git diff --check
git status --short
```

不要提交模型权重、视频、运行结果、机器人地址或密钥。

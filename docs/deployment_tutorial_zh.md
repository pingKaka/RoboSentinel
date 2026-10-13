# RoboSentinel 服务器部署教程

## 1. 系统分工

```text
本地设备                         RTX 4090服务器
采集双相机图像 ───────────────→ VLM理解任务
采集机器人状态（可选）────────→ GroundingDINO定位
                                 SAM实时跟踪
                                 规则检测
接收故障和恢复建议 ←────────── VLM故障分析
```

本地负责采集和发送，服务器负责模型计算。急停和碰撞保护仍应在本地完成。

## 2. 部署服务器

先把代码上传到GitHub，具体步骤见 [GitHub上传教程](github_upload_zh.md)。在本地终端登录服务器并克隆代码：

```bash
ssh user@SERVER_IP
git clone https://github.com/YOUR_NAME/RoboSentinel.git /data/RoboSentinel
exit
```

将本地模型权重单独上传。权重不进入GitHub：

```bash
rsync -az --progress checkpoint/ user@SERVER_IP:/data/RoboSentinel/checkpoint/
```

再次登录服务器并安装环境：

```bash
ssh user@SERVER_IP
cd /data/RoboSentinel
conda create -n robosentinel python=3.10 -y
conda activate robosentinel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

检查GPU和模型：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python failure_detection/run_pipeline.py
```

三个模型都应显示 `available: true`。

以后代码有更新时，先在本地提交并推送，再在服务器执行：

```bash
cd /data/RoboSentinel
git pull --ff-only
```

## 3. 接入真实模型

网络服务已经实现，但真实模型需要通过工厂函数接入。在根目录创建 `my_backend.py`：

```python
from failure_detection.pipeline import StreamingRoboSentinel
from failure_detection.pipeline.rule_engine import StreamingRuleEngine
from failure_detection.stages import (
    EventTriggeredVLMDiagnoser,
    OnlineGroundingSAMTracker,
    VLMScenePlanner,
)


def create_pipeline(config):
    qwen = load_qwen_once(config["models"]["vlm"])
    tracking_backend = create_online_tracker(
        config["models"]["grounding_dino"],
        config["models"]["sam"],
    )

    planner = VLMScenePlanner(qwen.plan_tracking_targets)
    tracker = OnlineGroundingSAMTracker(tracking_backend)
    rules = StreamingRuleEngine([
        # 加入经过验证的故障规则
    ])
    diagnoser = EventTriggeredVLMDiagnoser(qwen.diagnose_failure)

    return StreamingRoboSentinel(
        planner, tracker, rules, diagnoser,
        buffer_seconds=config["runtime"]["buffer_seconds"],
    )
```

需要实现三个部分：

- `load_qwen_once`：加载Qwen3-VL；
- `create_online_tracker`：加载GroundingDINO和SAM；
- 故障规则插件。

然后修改 `cfg.yaml`：

```yaml
deployment:
  host: "0.0.0.0"
  port: 8000
  pipeline_factory: "my_backend:create_pipeline"
```

模型只加载一次。GroundingDINO负责初始化和重新定位，SAM逐帧更新，VLM只在初始化和故障确认时调用。

## 4. 启动服务器

```bash
conda activate robosentinel
cd /data/RoboSentinel
python scripts/serve.py
```

检查服务：

```bash
curl http://SERVER_IP:8000/health
```

预期：

```json
{"status":"ok","pipeline_factory_configured":true,"active_tasks":0}
```

## 5. 本地发送实时图像

本地安装依赖：

```bash
pip install numpy opencv-python-headless httpx websockets
```

查看相机编号：

```bash
ls /dev/video*
```

启动客户端：

```bash
python scripts/stream_client.py \
  --server http://SERVER_IP:8000 \
  --external 0 \
  --wrist 2 \
  --task-id bottle_task \
  --description "Pick the bottle and place it in the rack" \
  --fps 10
```

运行过程：

```text
初始画面 → VLM生成跟踪对象 → GroundingDINO定位 → SAM初始化
实时画面 → SAM更新 → 规则检测
规则确认异常 → VLM返回故障分析和恢复建议
```

相机参数也可以填写RTSP地址。

## 6. 接入机器人状态

客户端目前发送 `robot_state: null`。接入后改成普通JSON：

```json
{
  "timestamp_sec": 3.42,
  "tcp_pose": [0.4, 0.1, 0.3, 0.0, 3.14, 0.0],
  "tcp_speed": [0.01, 0, 0, 0, 0, 0],
  "joint_position": [0, -1.2, 1.4, -1.7, -1.57, 0],
  "gripper_width": 0.021,
  "controller_state": "running"
}
```

机器人状态应与图像使用同一时间基准。它是规则和VLM的附加证据，不影响视觉主流程。

## 7. 常见问题

- 创建任务返回503：检查 `pipeline_factory`、`my_backend.py`和模型加载错误。
- 显存不足：Qwen使用4-bit量化，只运行一个任务，先降低到5 FPS。
- 延迟持续增加：降低FPS，服务器只保留最新帧。
- SAM跟丢目标：重新调用GroundingDINO定位。
- 网络中断：本地停止任务或进入安全状态。

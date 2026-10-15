# RoboSentinel 服务器部署教程

从全新环境开始的两端完整操作请看 [从零启动教程](start_from_zero_zh.md)。

日常运行请优先看 [精简运行教程](system_walkthrough_zh.md)。参数已统一到根目录 `cfg.yaml`，见 [参数说明](config_reference_zh.md)；本文命令行参数仍可用作临时覆盖。服务端默认将结果保存至 `result/server/<会话ID>/results.jsonl` 后再回传，本地默认自动创建独立运行目录。

遥操作接入、常驻模型、视频回放和实时测试请按 [实时检测测试流程](realtime_test_zh.md) 执行。

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

三个模型都应显示 `available: true`。这只检查目录存在，实际 GPU 推理检查见下一节。

以后代码有更新时，先在本地提交并推送，再在服务器执行：

```bash
cd /data/RoboSentinel
git pull --ff-only
pip install -r requirements.txt
```

## 3. 已接入的真实模型和规则

仓库已包含根目录的 `my_backend.py`，无需自行创建。默认 `cfg.yaml` 已配置：

```yaml
deployment:
  host: "0.0.0.0"
  port: 8000
  pipeline_factory: "my_backend:create_pipeline"
```

模型后端位于 `failure_detection/backends.py`。Qwen3-VL 默认采用 NF4 4-bit 量化，同一进程复用模型；GroundingDINO 负责定位，SAM2.1 Video 使用独立的双相机时序状态逐帧更新。默认每 60 帧重新定位并重建跟踪状态以限制内存增长；跟丢时每 10 帧尝试重新定位。模型路径相对于配置文件目录解析，只读取本地权重。

默认规则位于 `failure_detection/rules.py`：

- `target_lost`：被操作对象在两个相机中持续不可见 2 秒时报告 `target_unobservable`。这表示观测丢失，不等于确认掉落；超过 `max_gap_seconds` 的采样间隔会重置持续时间。
- `controller_fault`：机器人状态为 `fault`、`error`、`protective_stop` 或 `emergency_stop` 时报告故障；未提供状态时返回未知。

规则确认异常后调用 VLM，持续异常不重复诊断；规则明确恢复正常后才重新触发。恢复建议只返回给客户端，不执行机器人动作。抓取失败、指定孔位放置等任务规则仍需要阶段信号或标定信息，不属于默认规则。

可在 `stages.rules.plugins` 注册内置规则或 `package.module:RuleClass`，并通过 `options` 传入构造参数。自定义类实现 `update(context) -> RuleFinding`。

先使用一张包含目标的真实图片验证完整 GPU 推理（单张图片重复作为双相机输入，仅用于检查接线，不验证运动跟踪精度）：

```bash
python scripts/check_backend.py --image /path/to/bottle.jpg --description "Pick up the bottle"
```

也可以使用 `--mode qwen` 单独检查规划和诊断，或使用 `--mode tracker --target bottle` 检查 GroundingDINO 和 SAM2 视频更新。自检会创建独立模型进程，请在服务器服务未运行时执行。默认服务器在启动时加载三个模型并保持 GPU 常驻，可能需要数分钟；客户端默认等待 600 秒。

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
{"status":"ok","pipeline_factory_configured":true,"active_tasks":0,"models_ready":true,"resident_models":["vlm","grounding_dino","sam"],"device":"NVIDIA GeForce RTX 4090","allocated_mb":0}
```

`allocated_mb` 会显示实际加载后显存值，上面的 0 仅为格式示意。`models_ready` 表示三个模型已加载并检查参数位于 GPU，不证明任务检测精度。默认最多一个任务；连接断开后释放任务跟踪状态，模型仍驻留供下次复用。创建后未连接的任务可通过 `DELETE /tasks/{task_session_id}` 删除。更新代码和依赖后需重启服务。

## 5. 本地发送实时图像

本地安装依赖：

```bash
pip install -r requirements-client.txt
pip install opencv-python
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

独立相机客户端 `stream_client.py` 发送 `robot_state: null`；遥操作桥接已自动发送实际状态，见实时测试流程。其他运行环境可发送普通JSON：

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
- 延迟过大：降低FPS；客户端每发送一对图像就等待一次响应，不会主动无限堆积请求。相机驱动自身的缓冲需单独配置。
- SAM跟丢目标：重新调用GroundingDINO定位。
- 网络中断：本地停止任务或进入安全状态。

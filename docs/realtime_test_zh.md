# 遥操作与服务器实时检测测试

日常运行请优先看 [精简运行教程](system_walkthrough_zh.md)。参数已统一到根目录 `cfg.yaml`，见 [参数说明](config_reference_zh.md)；本文命令行参数仍可用作临时覆盖。服务端默认将结果保存至 `result/server/<会话ID>/results.jsonl` 后再回传，本地默认自动创建独立运行目录。

## 1. 数据怎么走

```text
本地：主臂 -> 原有遥操作 -> UR5e/UR7e
                         |
                   已有 get_observation()
                         |
                双相机 RGB + 机器人状态
                         |
                 后台客户端（最新观测）
                         |
服务器：常驻 Qwen3-VL + GroundingDINO + SAM2.1
                         |
                跟踪 -> 规则 -> 故障诊断
                         |
本地：终端结果 + results.jsonl + observations.jsonl
```

检测不会向机器人发送动作。遥操作是当前动作来源；未来换为 VLA 时，检测仍接收相同的 `ObservationPacket`。网络、模型推理和 JSONL 写入在后台线程，主循环只采样、复制待发送观测。队列容量为一，慢服务器会跳过旧观测，不会累积历史帧。

## 2. 服务器准备（第一次或更新后）

本地改动需先提交并推送到服务器使用的 Git 仓库。已有三个权重目录不需要重新传输。

```bash
conda activate robosentinel
cd /data/RoboSentinel
git pull --ff-only
pip install -r requirements.txt
python scripts/serve.py
```

启动时会显示 `Loading three models onto CUDA`，依次加载三个模型，然后显示 `READY` 和 `Application startup complete`。默认 Qwen 使用 NF4 4-bit，另外两个模型也保留在 GPU。启动失败时检查完整报错；显存不足时应关闭其他 GPU 进程或使用目标 24GB 4090。

模型通过进程内缓存保留。新建任务、断开客户端、重新开始测试只重建跟踪状态和规则状态，不重新加载权重。不要启用多 worker 或自动 reload；每个进程都会各自加载一份模型。SAM 视频历史放在 CPU，模型参数仍在 GPU。

在另一个服务器终端检查：

```bash
curl -s http://127.0.0.1:8000/health
nvidia-smi
```

预期 `/health` 包含 `models_ready: true`、`resident_models: ["vlm", "grounding_dino", "sam"]` 和显卡名称；`allocated_mb` 是加载后的显存快照。尚在加载时 HTTP 服务可能还没有开始监听。此检查证明权重已加载，下一步验证实际推理。

服务运行期间不要另外启动 `check_backend.py`，它是独立进程，会再加载一套模型。

## 3. 本地环境与网络

本地继续使用已能遥操作的 `ur_data` 环境，不安装服务器整套依赖，以免替换 GUI OpenCV 或机器人依赖：

```bash
conda activate ur_data
cd /home/kakaping/ur_data_collection/RoboSentinel
pip install -r requirements-client.txt
pip install --no-deps -e ./ur5e_video_teleop
curl -s http://SERVER_IP:8000/health
```

把所有命令中的 `SERVER_IP` 替换成服务器地址。已有 `lerobot_robot_ur5e`、`lerobot_teleoperator_ur5e`、RTDE、RealSense、Pinocchio 和主臂依赖继续复用。子项目移动后的 editable 安装需要执行上面的安装命令，确保运行的是本仓库版本。

若服务器仅能 SSH 访问，本地另开终端：

```bash
ssh -N -L 8000:127.0.0.1:8000 user@SERVER_IP
```

后续客户端地址改成 `http://127.0.0.1:8000`。

## 4. 第一步：视频回放，不连接机器人

先用子项目已有的成对视频验证检测；这一命令不导入机器人驱动，也不连接机器人：

```bash
python scripts/replay_detection.py \
  --server http://SERVER_IP:8000 \
  --external ur5e_video_teleop/videos/exterior_1.mp4 \
  --wrist ur5e_video_teleop/videos/wrist_1.mp4 \
  --seconds 30 --fps 5 \
  --description "Pick the white-capped reagent bottle and place it in the test-tube rack." \
  --output result/test_video
```

视频文件保留在本地，不通过 Git 上传。两个视频应是同一次录制且帧率相同；回放按帧号配对，这不等于硬件时间同步。

客户端先等初始场景规划完成，再按视频时间播放。看到 `STREAMING` 后会持续打印：

```text
[Sentinel] t=2.00 visible=4/4 failure=None server=230.5ms dropped=1
```

`visible` 是各相机中可见的目标数，`failure=None` 表示证据不足（纯视频没有控制器状态），不是程序报错。`server` 为服务器本轮处理耗时，包含触发时的诊断；`dropped` 是被新观测替换的旧观测数。5 FPS 是发送上限，实际吞吐由模型和网络决定，回放结束会等最后一帧结果，再输出 `PASS`。

查看 `result/test_video/results.jsonl`：应有 `perception.tracks`、`rules`、`diagnosis`、时间戳和耗时。只有确认异常时 `diagnosis` 才非空。

## 5. 第二步：验证故障诊断通路

无需人为制造真实机器人故障。只在回放数据中注入一个标记为 `synthetic: true` 的控制器故障：

```bash
python scripts/replay_detection.py \
  --server http://SERVER_IP:8000 \
  --external ur5e_video_teleop/videos/exterior_1.mp4 \
  --wrist ur5e_video_teleop/videos/wrist_1.mp4 \
  --seconds 15 --fps 3 --simulate-fault-at 3 \
  --output result/test_diagnosis
```

视频需长于 3 秒。预期规则报告 `controller_fault`，终端出现 `DIAGNOSIS`，结果包含中文分析及恢复建议。持续同一故障不会逐帧重复诊断。合成状态会进入 VLM 证据，建议不被执行；这一步验证调用链，不评估真实故障检测精度。

## 6. 第三步：真实遥操作在线检测

使用原来已经验证的主从臂标定、控制模式、串口、机器人 IP 和相机序列号。不要同时启动旧遥操作程序或另一个相机客户端，以免争用设备。

```bash
ur5e-video-teleop \
  --config ur5e_video_teleop/config/video_teleop.yaml \
  --sentinel-server http://SERVER_IP:8000 \
  --task-description "Pick the white-capped reagent bottle and place it in the far-right hole of the large test-tube rack." \
  --sentinel-output result/test_teleop
```

该命令会启动真实遥操作，沿用原有动作路径。先保持初始场景稳定，看到 `STREAMING` 后再开始本次任务。空格仍只控制原有 MP4 录制；检测在启用后持续运行，不依赖录制键。预览显示 Sentinel 的连接状态。`Esc` 退出遥操作并关闭检测连接。

发送的数据包括双相机 RGB、关节位置/速度、TCP 速度/力、基座 TCP 位姿、安全状态和夹爪归一化位置。夹爪归一化位置不是米制开口宽度。原有相对末端位姿另存为 `teleop_ee_pose`，不冒充基座 TCP 位姿。

时间使用本地单调时钟；`capture_span_sec` 记录一次观测读取耗时，`state_read_delay_sec` 记录附加状态读取延迟。目前原驱动顺序读取相机且没有曝光时间戳，因此这是软件近似配对，不宣称硬件同步。

默认文件：

| 文件 | 内容 |
| --- | --- |
| `results.jsonl` | 每次服务器响应、检测和诊断、耗时 |
| `observations.jsonl` | 实际发送的 JPEG 图像、机器人状态、时间信息，可复现回放 |

默认自动生成独立输出目录。图像 JSONL 会占用磁盘；长时间运行可在根目录 `cfg.yaml` 中将 `client.record_observations` 设为 `false`。它记录的是发送样本，不是主循环全部帧。下文显式指定输出目录时，请自行避免重复使用同一路径。

网络或检测失败会显示 `error`，遥操作继续按原逻辑运行，检测不会自动急停。恢复网络后重启本次检测运行；不自动换场景继续旧任务。原有本地急停和安全操作仍适用，`debug: True` 也不代表夹爪完全不动作。

## 7. 重放刚采集的真实观测

退出遥操作后执行：

```bash
python scripts/replay_detection.py \
  --server http://SERVER_IP:8000 \
  --observations result/test_teleop/observations.jsonl \
  --seconds 30 --output result/test_teleop_replay
```

使用与录制时相同的 `--description`。重放会保留实际机器人状态并重映射到回放时间，绝不执行记录的 action。

## 8. 将来接入 VLA

在 VLA 主循环得到观测后，调用同一个客户端；检测模块不依赖动作来自主臂还是策略：

```python
import time
from failure_detection.transport.live_client import LiveDetectionClient, ObservationPacket

client = LiveDetectionClient("http://SERVER_IP:8000", "Your task", task_id="vla", fps=5).start()
origin = time.monotonic()
try:
    # observation_stream 由已有 VLA 环境提供；图像为 uint8 HWC RGB。
    for observation in observation_stream:
        timestamp = time.monotonic() - origin
        state = observation.get("robot_state")
        if state is not None:
            state = {**state, "timestamp_sec": timestamp}
        client.submit(ObservationPacket(timestamp, observation["frames"], state))
finally:
    client.close()
```

`robot_state.timestamp_sec` 应使用同一时间原点，位姿必须注明坐标系和单位。VLA 的 action 下发仍留在原环境中；`client.latest_result` 只供监测/显示，不自动转成动作。

## 9. 验收与排查

- 权重常驻：启动只出现一次加载进度；依次做两个回放任务不再加载权重；断开客户端后 `nvidia-smi` 仍有显存占用。
- 数据接入：实时结果中 `robot_state_received` 为 `true`，观测文件含两路图像和实际状态。
- 检测通路：回放有连续跟踪结果；合成故障测试出现一次诊断；真实测试不以合成故障作为精度结论。
- `409`：默认只允许一个任务，先关闭其他客户端。创建后未连接的遗留任务可以通过 `DELETE /tasks/{task_session_id}` 删除；活动任务需先断开流。
- `503`：查看服务器完整报错，常见原因是场景规划 JSON 不合法或模型推理失败。
- 跟踪不可见：先检查对象是否真的在画面内、任务描述是否准确；不把目标持续不可见等同于抓取失败或掉落。
- 推理太慢：先降到 2 FPS，比较 `server_processing_ms` 与 `client_roundtrip_ms`；不要把 5 FPS 配置当成性能保证。
- 单轮采样间隔超过 `target_lost.max_gap_seconds` 会重置失踪计时，按实测吞吐调整该参数；缺帧不用于确认连续故障。

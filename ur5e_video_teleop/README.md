# UR5e Video Teleop

本项目已集成到 RoboSentinel。双相机和状态上传、服务器常驻模型、无机器人视频回放及实时测试入口见 [项目总览](../README.md)。

在 RoboSentinel 根目录安装 `requirements-client.txt` 并执行 `pip install --no-deps -e ./ur5e_video_teleop` 后，可用 `--sentinel-server http://SERVER_IP:8000` 启用后台检测。不开启该参数且配置 `sentinel.enabled: False` 时保留原有视频遥操作行为。检测只接收观测并输出规则报告，不控制机器人。

`ur5e_video_teleop` 是一个轻量版 UR5e/UR7e 同构遥操作项目。它复用原项目
`lerobot_robot_ur5e` 和 `lerobot_teleoperator_ur5e` 的机器人控制、Dynamixel
主臂读取、Pinocchio 正运动学、夹爪控制和 RealSense 相机接入逻辑，因此遥操作行为与
`ur5e-record` 保持一致。

本项目和原始数据集录制项目的区别是：不创建 LeRobotDataset，不保存 action/observation
数据，不写 Hugging Face 数据集，只保存两路设定相机的视频画面。

## 1. 功能概览

- 使用 Dynamixel 主臂遥操作 UR5e/UR7e 从臂。
- 支持原项目相同的控制空间：
  - `joint`
  - `joint_to_tcp_force`
  - `tcp_force`
  - `tcp_position`
- 支持 DH gripper 夹爪控制。
- 支持两路 RealSense RGB 相机：
  - `wrist_image`
  - `exterior_image`
- 第一次按下录制键时同时开始保存两路相机视频。
- 再次按下录制键时同时结束两路视频文件。
- 每次开始录制都会为两路相机各生成一个新的 `.mp4` 文件。
- 可选择是否显示本地预览窗口。

## 2. 项目结构

```text
ur5e_video_teleop/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── video_teleop.yaml
├── videos/
└── ur5e_video_teleop/
    ├── __init__.py
    └── run_video_teleop.py
```

关键文件：

- `config/video_teleop.yaml`: 运行配置文件。
- `ur5e_video_teleop/run_video_teleop.py`: 主程序入口。
- `videos/`: 默认视频输出目录。

## 3. 环境准备

先确保原始项目环境已经可用，并且 Pinocchio、UR RTDE、Dynamixel、RealSense、夹爪库都能正常导入。

推荐在已有 `ur_data` 环境中运行：

```bash
conda activate ur_data
```

进入新项目目录：

```bash
cd /home/kakaping/ur_data_collection/RoboSentinel/ur5e_video_teleop
```

安装本项目：

```bash
pip install -r requirements.txt
```

`requirements.txt` 会以 editable 方式安装：

- `../../lerobot_ur5e_isoteleop/lerobot_robot_ur5e`
- `../../lerobot_ur5e_isoteleop/lerobot_teleoperator_ur5e`
- 当前项目本身

安装完成后应能看到命令帮助：

```bash
ur5e-video-teleop --help
```

## 4. 配置文件

默认配置文件是：

```bash
/home/kakaping/ur_data_collection/RoboSentinel/ur5e_video_teleop/config/video_teleop.yaml
```

集成版本默认读取父项目根目录 `cfg.yaml`；上面的旧文件仅通过 `project_config` 指向根配置。硬件参数集中在根配置的 `video_teleop` 下，检测连接与输出参数在 `client` 下。也可以通过 `--config` 指定其他配置：

```bash
ur5e-video-teleop --config /path/to/video_teleop.yaml
```

### 4.1 顶层运行配置

```yaml
video_teleop:
  fps: 15
  debug: False
  record_key: "space"
  exit_key: "esc"
  output_dir: "videos"
  preview: True
```

字段说明：

- `fps`: 遥操作循环和视频写入目标帧率。
- `debug`: 传给 `UR5eConfig.debug`。为 `True` 时从臂不会执行机械臂 action，但夹爪仍可能被控制。
- `record_key`: 按一次开始录制，再按一次停止录制。
- `exit_key`: 退出程序的按键。
- `output_dir`: 视频保存目录。根配置中的相对路径基于配置文件目录；命令行路径基于当前工作目录。
- `preview`: 是否显示 OpenCV 预览窗口。

默认行为：

- 第一次按下 `space` 同时开始录制 `wrist_image` 和 `exterior_image`。
- 再次按下 `space` 同时停止两路视频。
- 按 `esc` 退出程序。

### 4.2 主臂 Dynamixel 配置

```yaml
teleop:
  control_mode: "isoteleop"
  dynamixel_config:
    port: "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0"
    hardware_offsets: [-225.791, -178.417, -267.541, -234.14, -108.02, -140.45]
    use_gripper: True
    joint_ids: [1, 2, 3, 4, 5, 6]
    joint_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    joint_signs: [1, 1, -1, 1, 1, 1]
    gripper_config: [7, -0.612058, 0.374291]
```

字段说明：

- `port`: Dynamixel 主臂串口路径，建议使用 `/dev/serial/by-id/...` 这种固定路径。
- `hardware_offsets`: 主臂和从臂硬件零点偏差，单位是度。
- `use_gripper`: 是否启用主臂第 7 个 Dynamixel 作为夹爪输入。
- `joint_ids`: 6 个主臂关节的 Dynamixel ID。
- `joint_offsets`: 主从臂关节角周期偏移，单位是弧度。
- `joint_signs`: 主从臂关节方向符号。
- `gripper_config`: `[id, min, max]`，用于把主臂夹爪位置映射到 `[0, 1]`。

这些字段应和原项目 `scripts/config/cfg.yaml` 中已经标定好的值保持一致。

### 4.3 UR5e/UR7e 从臂配置

```yaml
robot:
  ip: "192.168.131.12"
  use_gripper: True
  close_threshold: 0.7
  gripper_bin_threshold: 0.98
  gripper_port: "/dev/ur5e_left_gripper"
  gripper_reverse: False
  gripper_force: 70
  gripper_speed: 60
  control_space: "tcp_force"
```

字段说明：

- `ip`: UR 控制柜 IP。
- `use_gripper`: 是否启用从臂夹爪。
- `close_threshold`: 主臂夹爪输入二值化阈值。
- `gripper_bin_threshold`: 从臂夹爪观测二值化阈值。
- `gripper_port`: DH gripper 串口路径。
- `gripper_reverse`: 如果夹爪开合方向相反，改为 `True`。
- `gripper_force`: 夹爪力控命令。
- `gripper_speed`: 夹爪速度命令。
- `control_space`: 从臂控制模式。

支持的 `control_space`：

```yaml
control_space: "joint"
control_space: "joint_to_tcp_force"
control_space: "tcp_force"
control_space: "tcp_position"
```

各模式含义：

- `joint`: 主臂输出 6 个关节角，从臂通过 `servoJ` 关节位置控制跟随。
- `joint_to_tcp_force`: 主臂输出关节角，经 URDF 正运动学得到目标末端位姿，从臂通过 `forceMode` 跟随。
- `tcp_force`: 主臂目标末端位姿和从臂当前末端位姿相减，生成 TCP delta，从臂通过 `forceMode` 跟随。
- `tcp_position`: 生成 TCP delta 后，从臂通过 `servoL` 笛卡尔位置控制跟随。

### 4.4 TCP 坐标系配置

```yaml
tcp_force:
  reference_frame: "tcp"
tcp_position:
  reference_frame: "base"
```

仅在对应控制模式下生效。

- `base`: TCP delta 在机器人 base 坐标系下表达。
- `tcp`: TCP delta 在当前末端坐标系下表达。

### 4.5 力控参数

```yaml
force_mode:
  kp: 2000
  kd: 200
  kp_rot: 4000
  kd_rot: 800
  rtde_freq: 125
  select_vector: [1, 1, 1, 1, 1, 1]
  force_limit: [2, 2, 2, 2, 2, 2]
  pos_delta: 0.2
  vel_delta: 0.2
  gain_scale: 1.5
```

这些参数直接传给原项目 `UR5e` 控制逻辑，用于 `joint_to_tcp_force` 和 `tcp_force`。
如果原项目 `ur5e-record` 的遥操作手感已经合适，保持相同参数即可。

### 4.6 关节控制参数

```yaml
joint_mode:
  look_ahead_time: 0.2
  dt: 0.002
  gain: 100
```

这些参数用于 `joint` 模式下的 UR `servoJ` 控制。

### 4.7 相机配置

```yaml
cameras:
  wrist_cam_serial: "349522070893"
  exterior_cam_serial: "213522072373"
  width: 640
  height: 480
```

字段说明：

- `wrist_cam_serial`: 腕部 RealSense 相机序列号。
- `exterior_cam_serial`: 外部 RealSense 相机序列号。
- `width`: RGB 图像宽度。
- `height`: RGB 图像高度。

可以用原项目命令检查相机：

```bash
tools-check-rs
```

## 5. 运行教程

### 5.1 启动前检查

1. 激活环境：

```bash
conda activate ur_data
```

2. 确认 UR 示教器处于 Remote Control。

3. 确认主臂串口路径存在：

```bash
ls /dev/serial/by-id/
```

4. 确认夹爪串口路径存在：

```bash
ls /dev/ur5e_left_gripper
```

5. 确认 RealSense 相机序列号正确：

```bash
tools-check-rs
```

6. 确认根目录 `config/teleop.yaml` 中的标定参数和原项目一致：

- `hardware_offsets`
- `joint_offsets`
- `joint_signs`
- `gripper_config`

### 5.2 启动默认遥操作

```bash
cd /home/kakaping/ur_data_collection/RoboSentinel/ur5e_video_teleop
ur5e-video-teleop
```

终端出现类似以下信息后即可操作：

```text
====== [READY] Press 'space' once to start recording both cameras, press it again to stop. Press 'esc' to exit. ======
```

### 5.3 录制视频

第一次按下 `space`：

- 机器人继续实时遥操作。
- 程序同时开始写入 `wrist_image` 和 `exterior_image` 两路视频。
- 预览窗口显示两路相机画面，并标记 `REC`。
- 终端分别打印两个视频文件路径。

再次按下 `space`：

- 两个视频文件同时关闭。
- 机器人继续实时遥操作。
- 下一次按下 `space` 会创建新的一对视频文件。

按 `esc`：

- 停止程序。
- 如果正在录制，会先关闭当前视频。
- 断开机器人、主臂、相机连接。

### 5.4 修改录制按键

临时把录制切换键改为 `r`：

```bash
ur5e-video-teleop --record-key r
```

第一次按下 `r` 开始同时录制两路相机，第二次按下 `r` 停止录制。

或修改配置：

```yaml
record_key: "r"
```

特殊按键使用 `pynput` 的名称，例如：

- `space`
- `shift`
- `ctrl`
- `alt`
- `enter`
- `esc`

### 5.5 修改视频保存目录

```bash
ur5e-video-teleop --output-dir /home/kakaping/ur_videos
```

或修改配置：

```yaml
output_dir: "/home/kakaping/ur_videos"
```

### 5.6 关闭预览窗口

如果机器没有桌面环境，或者不想显示 OpenCV 窗口：

```bash
ur5e-video-teleop --no-preview
```

或修改配置：

```yaml
preview: False
```

## 6. 视频文件格式

默认保存为 `.mp4`，编码器为 OpenCV `mp4v`。

文件名格式：

```text
<camera_name>_<YYYYMMDD_HHMMSS>_<segment_index>.mp4
```

示例：

```text
videos/wrist_image_20260909_153012_001.mp4
videos/wrist_image_20260909_153045_002.mp4
```

说明：

- 每次开始录制都会创建两个新文件，一个对应 `wrist_image`，一个对应 `exterior_image`。
- 再次按下录制键后，两个 writer 都会被 release，视频可直接播放。
- 如果一次运行中多次开始/停止，会生成多组视频片段。

## 7. 和 `ur5e-record` 的区别

相同点：

- 使用同一套 `UR5e` 从臂类。
- 使用同一套 `UR5eTeleop` 主臂类。
- 使用同一套 Dynamixel 标定参数。
- 使用同一套 URDF/Pinocchio 正运动学。
- 使用同一套 UR RTDE 控制路径。
- 使用同一套 RealSense 相机配置方式。

不同点：

- 不创建 `LeRobotDataset`。
- 不保存 action。
- 默认不保存 observation/state；启用 Sentinel 后可记录实际发送的图像和状态 JSONL。
- 不保存 episode/task/meta 信息。
- 不上传 Hugging Face Hub。
- 不使用 LeRobot 的 `record_loop` 和键盘事件。
- 同时保存 `wrist_image` 和 `exterior_image` 两路视频画面。

## 8. 安全注意事项

运行前请确认：

- UR 示教器处于 Remote Control。
- 急停按钮可用。
- 主臂和从臂姿态大致对齐。
- `hardware_offsets` 和 `joint_offsets` 已正确标定。
- `control_space` 与你预期一致。
- `debug` 在真实控制时为 `False`，调试时可设为 `True`。
- 工作空间内没有障碍物。
- 按 `esc` 或 `Ctrl+C` 可以退出程序。

如果只是测试相机写视频，不想让机械臂执行 action，可以先设：

```yaml
debug: True
```

注意：原项目的夹爪线程在 debug 模式下仍可能响应夹爪 action。

## 9. 常见问题

### 9.1 启动后没有按键响应

`pynput` 依赖当前桌面/终端环境捕获键盘事件。请确认：

- 程序运行在本地桌面会话中。
- 当前终端窗口处于焦点状态。
- 如果通过 SSH 运行，可能无法捕获本地键盘按下/松开事件。

无桌面环境下建议使用 `--no-preview`，但键盘监听仍需要当前环境支持。

### 9.2 OpenCV 预览窗口打不开

如果没有图形界面，运行：

```bash
ur5e-video-teleop --no-preview
```

### 9.3 视频文件打不开或长度为 0

通常原因：

- 开始录制后持续时间太短，没有写入足够帧。
- 程序异常退出时视频 writer 没来得及 release。
- 输出目录无写权限。

建议：

```bash
mkdir -p videos
ls -lh videos
```

然后按下录制键开始录制，等待至少 2 秒后再次按下录制键停止。

### 9.4 RealSense 相机连接失败

检查相机序列号：

```bash
tools-check-rs
```

然后更新：

```yaml
cameras:
  wrist_cam_serial: "<your_wrist_serial>"
  exterior_cam_serial: "<your_exterior_serial>"
```

### 9.5 Dynamixel 主臂连接失败

检查串口：

```bash
ls /dev/serial/by-id/
```

确认配置中的：

```yaml
port: "/dev/serial/by-id/..."
```

必要时修改权限：

```bash
sudo chmod 666 /dev/serial/by-id/<your_device>
```

### 9.6 夹爪连接失败

检查固定映射是否存在：

```bash
ls /dev/ur5e_left_gripper
```

如果不存在，需要重新映射夹爪串口：

```bash
cd /home/kakaping/ur_data_collection/lerobot_ur5e_isoteleop
map_gripper.sh ur5e_left_gripper
```

### 9.7 遥操作方向不对

优先检查：

- `joint_signs`
- `hardware_offsets`
- `joint_offsets`
- `gripper_reverse`

这些参数应直接复用原项目中已经验证过的配置。

## 10. 开发说明

主循环在：

```text
ur5e_video_teleop/run_video_teleop.py
```

核心流程：

```text
读取配置
创建 UR5e 和 UR5eTeleop
连接机器人、主臂、相机
启动键盘监听
循环执行：
  action = teleop.get_action()
  robot.send_action(action)
  observation = robot.get_observation()
  如果录制键按下：
  如果录制状态开启：
    同时写 wrist_image 和 exterior_image 视频帧
  如果录制状态关闭：
    关闭两路视频文件
退出时：
  关闭视频 writer
  停止键盘监听
  断开机器人、主臂、相机
```

因为没有数据集写入，本项目不会生成 LeRobot 的 episode、metadata、parquet 或 video dataset
目录；默认输出普通 `.mp4` 文件。启用 Sentinel 后还可输出检测结果和观测 JSONL，供测试与回放。

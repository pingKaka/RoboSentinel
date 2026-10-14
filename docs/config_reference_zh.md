# cfg.yaml 参数说明

所有主工作流默认读取仓库根目录 `cfg.yaml`，用 `--config` 切换。显式命令行参数优先于配置。修改后重启对应服务/客户端，不热加载。两台机器的配置独立：任务文字以客户端发送值为准。
配置中的模型、回放、输出路径相对配置文件目录；命令行路径相对当前目录。硬件驱动资源路径（如 URDF）仍遵循驱动规则。子项目旧配置只是指向根配置的兼容入口。

## 服务与模型

| 参数 | 作用 |
| --- | --- |
| version | 配置结构版本，目前 4 |
| task.id / description | 任务编号 / 自然语言目标；客户端每次启动创建新会话 |
| models.vlm / grounding_dino / sam | Qwen3-VL / GroundingDINO / SAM2.1 本地权重目录 |
| deployment.host / port | 服务监听地址 / 端口；0.0.0.0 表示所有网卡 |
| deployment.max_tasks | 同时存在的任务数，建议保持 1，受显存限制 |
| deployment.pipeline_factory | 每个任务构建流水线的 Python 工厂路径 |
| deployment.preload_factory | 启动时加载并检查三模型驻留 CUDA 的工厂；不要置空来跳过检查 |
| backend.quantize_vlm | Qwen 使用 NF4 量化，降低显存开销 |
| backend.vlm_max_new_tokens | 每次 Qwen 生成的最大 token 数；过小可能截断 JSON |
| backend.vlm_image_max_side | Qwen 输入图像最长边像素，控制视觉开销 |
| backend.box_threshold / text_threshold | DINO 框 / 文本置信度阈值，范围 0~1 |
| backend.min_mask_pixels | SAM 可见目标所需的最少掩码像素，正整数 |
| backend.reseed_frames | 每隔多少次跟踪更新重新 DINO 定位并初始化 SAM，正整数 |
| backend.redetect_frames | 目标丢失后尝试 DINO 重定位的更新间隔，正整数 |
| runtime.buffer_seconds | 缓存感知历史的秒数，不是原始视频缓存 |
| stages.rules.plugins | 规则名称或 name/options 列表，见 [扩展规则](rules_zh.md) |
| target_lost 的 options.seconds | 双视角持续不可见的确认时长（秒） |
| target_lost 的 options.max_gap_seconds | 相邻观测超过此间隔时重新计时，避免把丢帧判为持续不可见 |

`stages.planner.backend`、`stages.tracker.backend`、`stages.diagnosis.backend` 是描述性标签，不会动态切换模型，实际实现由工厂决定。
`streams.external_camera.provider`、`streams.wrist_camera.provider`、`streams.robot_state.provider` 是预留元信息；实时输入来自客户端。
`streams.robot_state.enabled` 仅用于原配置摘要，不是实时状态上传开关。`streams.max_timestamp_delta_sec` 目前也不执行实时同步拒收；实际采集跨度随数据上传。不要把这些字段当作已实现的安全校验。

## 结果与客户端

| 参数 | 作用 |
| --- | --- |
| outputs.server_dir | 服务端结果根目录，每个会话独立子目录；null 禁用磁盘保存 |
| outputs.console | 服务端输出完整 RESULT JSON 日志 |
| outputs.save_annotated_frames | 额外保存每个已处理观测的双视角带框 JPG，增加磁盘开销 |
| outputs.fsync | 每条 JSON 写入后 fsync，降低断电丢失风险，增加延迟；不是跨文件事务 |
| client.server | 客户端连接地址，不能填 0.0.0.0 |
| client.fps | 最大观测发送频率 Hz，不是模型吞吐保证；繁忙时取最新观测，不累积队列 |
| client.timeout_sec | 网络请求等待秒数；回放也用它限制等待初始化/结果的时长 |
| client.jpeg_quality | 上传 JPEG 质量，1~100 |
| client.ws_max_size | WebSocket 接收消息大小上限，字节 |
| client.ping_timeout_sec | WebSocket 心跳等待秒数，模型诊断慢时需留余量 |
| client.output_dir | 本地结果根目录，自动追加模式/时间与随机 ID，避免混合多次实验 |
| client.record_observations | 保存已发送观测（图像和状态）供回放；会增加磁盘用量 |
| client.log_level | 本地日志等级，例如 INFO、DEBUG、WARNING |

服务端先写 results.jsonl 再回传，客户端再写自己的 results.jsonl。首条 initialized 含规划与初始感知，后续 update 含感知、规则、诊断。`server_processing_ms` 是流水线处理时间，不包含存盘/网络耗时。
图片路径回传但图片不自动下载；默认不导出稠密掩码和原始 logits。写盘失败返回错误并终止会话，不发送成功结果。没有自动轮转清理，需监控磁盘空间。

## 测试输入

| 参数 | 作用 |
| --- | --- |
| replay.observations | 之前保存的 observations.jsonl；非 null 时优先于配置的视频对 |
| replay.external / wrist | 双视角视频路径；显式 CLI 输入覆盖配置输入 |
| replay.seconds | 回放最大时长（秒），视频可能提前结束 |
| replay.simulate_fault_at | 回放到该秒后注入模拟控制器故障，null 禁用；不作用于真机 |
| camera_client.external / wrist | 独立 OpenCV 客户端设备编号或视频/流地址，不是 RealSense 序列号 |
| check.image | 单图后端自检输入，必须设置或通过 --image 提供 |
| check.mode | qwen / tracker / full；选择单模型组或完整后端检查 |
| check.target | tracker 模式的英文目标短语 |
| check.frames / interval_sec | 自检更新次数 / 构造的观测时间间隔（秒），不是实际采集频率 |

## 遥操作 video_teleop

以下参数保留原程序语义和已有标定，修改前确认硬件驱动要求。检测只是旁路监测，没有运动许可联锁，也不会自动急停。

| 相对 video_teleop 的参数 | 作用 |
| --- | --- |
| fps | 遥操作主循环目标频率 Hz，与 client.fps 分离 |
| debug | 原调试模式，不保证所有执行器不动作 |
| record_key / exit_key | MP4 录制切换键 / 退出键 |
| output_dir / preview | MP4 保存目录 / 是否显示相机预览 |
| sentinel.enabled | 是否将遥操作观测送到检测服务；网络等参数统一取 client |
| teleop.control_mode | 主臂控制模式，目前 isoteleop |
| teleop.dynamixel_config.port | 主臂串口路径 |
| teleop.dynamixel_config.hardware_offsets | 主臂硬件标定偏置数组，保留原标定值和单位 |
| teleop.dynamixel_config.use_gripper | 是否读取主臂夹爪 |
| teleop.dynamixel_config.joint_ids | 主臂 6 个关节的电机 ID |
| teleop.dynamixel_config.joint_offsets / joint_signs | 关节映射偏置 / 方向符号，必须与标定一致 |
| teleop.dynamixel_config.gripper_config | [电机 ID, 最小位置, 最大位置]，映射到归一化夹爪输入 |
| robot.ip | 从臂 IP |
| robot.use_gripper / gripper_port | 夹爪开关镜像 / 从臂串口；当前程序共用 dynamixel_config.use_gripper，YAML 锚点保持两者一致，不要单独改镜像字段 |
| robot.close_threshold / gripper_bin_threshold | 主臂闭合指令阈值 / 从臂观测二值化阈值 |
| robot.gripper_reverse | 反转夹爪方向 |
| robot.gripper_force / gripper_speed | 夹爪驱动力/速度设置，范围和单位由夹爪驱动定义 |
| robot.control_space | tcp_force / tcp_position / joint，选择从臂控制方式 |
| robot.tcp_force.reference_frame | 力控制参考坐标系，如 tcp |
| robot.tcp_position.reference_frame | 位置控制参考坐标系，如 base |
| robot.tcp_position.speed / acceleration | TCP 位置控制速度/加速度参数，按底层 RTDE 接口解释 |
| robot.tcp_position.servo_time | 伺服周期（秒） |
| robot.tcp_position.lookahead_time / gain | 前瞻时间（秒）/ 伺服增益 |
| robot.robot_urdf_path | 驱动使用的机器人 URDF 资源路径 |
| robot.force_mode.kp / kd | 平移比例 / 微分增益 |
| robot.force_mode.kp_rot / kd_rot | 旋转比例 / 微分增益 |
| robot.force_mode.rtde_freq | 力控制 RTDE 循环频率 Hz |
| robot.force_mode.select_vector | 6 轴柔顺选择向量 |
| robot.force_mode.force_limit | 6 轴限制参数，具体速度/偏差语义由原驱动和选轴决定，不应直接理解为六个牛顿值 |
| robot.force_mode.pos_delta / vel_delta | 原力控驱动的位置/速度变化限制参数 |
| robot.force_mode.gain_scale | 力控增益缩放 |
| robot.joint_mode.look_ahead_time / dt / gain | 关节伺服前瞻秒数 / 周期秒数 / 增益 |
| cameras.wrist_cam_serial / exterior_cam_serial | 腕部 / 外部 RealSense 序列号 |
| cameras.width / height | 相机图像宽 / 高（像素） |

硬件参数的底层解释见子项目 README 和所安装的机器人驱动；不要为提高检测速度修改力控增益。VLA 迁移只需提供同样的图像和状态观测，不要求复用遥操作控制参数。

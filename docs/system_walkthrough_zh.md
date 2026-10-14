# RoboSentinel 运行教程

## 1. 只改一份配置

统一入口是项目根目录 `cfg.yaml`，逐项解释见 [参数说明](config_reference_zh.md)。
服务器和本地各有一份配置：本地填写服务器地址和真实硬件参数，服务器填写权重路径。
任务说明由本地 `task.description` 随创建请求发送，不会被服务器同名字段覆盖。

首先确认：
- `models.*`：服务器上三套已下载权重的目录。
- `client.server`：例如 `http://服务器IP:8000`。
- `task.id/description`：本次操作的编号和文字描述。
- `replay.external/wrist`：本地一对视频；示例视频不随 git 下载，需要自己准备。
- `video_teleop`：真机 IP、串口、相机序列号、主从臂标定。不要照抄陌生硬件的标定。

## 2. 启动服务器

在项目目录、已配置 CUDA 的 Python 环境中：
```bash
git pull
pip install -r requirements.txt
python scripts/serve.py
```
等待 `READY` 和 `Application startup complete`。另开终端：
```bash
curl http://127.0.0.1:8000/health
```
应看到 `models_ready: true` 和三项 `resident_models`。
模型常驻显卡直到进程退出；新任务只重建任务状态。不要启动多个服务进程或同时运行模型自检。
安装细节见 [部署说明](deployment_tutorial_zh.md)。git pull 不会下载模型或安装 CUDA，8GB 显存不足以验证三模型同时驻留。

## 3. 先用视频测试，不接机器人

本地项目目录执行：
```bash
pip install -r requirements-client.txt
python scripts/replay_detection.py
```
正常顺序：首对图像 → 创建任务 → Qwen 选择目标 → DINO 定位、SAM 初始化 → `STREAMING` → 持续返回结果。
先检查结果中有可见目标，而不是只看连接成功。
将 `replay.simulate_fault_at` 改为 `3` 再运行，可在回放第 3 秒注入模拟控制器故障，检查规则和 Qwen 诊断链路；测试后改回 `null`。
这只验证链路，不证明视觉检测准确率。缺少图像/状态时结果可能是 `unknown`，不代表正常。

## 4. 再接遥操作

在原来可以运行遥操作的环境中安装并启动：
```bash
pip install -r requirements-client.txt
pip install -e ./ur5e_video_teleop
ur5e-video-teleop
```
确认 `video_teleop.sentinel.enabled: true`。程序复用原相机和机器人状态，不额外连接设备。
**此命令会驱动真实机器人。现有循环先发送动作再采集观测，没有检测就绪联锁。**
按原流程对齐主从臂；初始化时主动保持主臂、场景稳定，看到 `STREAMING` 后再做实验。
空格只切换 MP4 录制，不创建检测任务；Esc 退出。不要同时启动其他占用相机的程序。
`debug` 不是硬件安全开关，夹爪仍可能响应。检测和诊断不会自动停止机器人。

## 5. 什么时候创建任务、如何检测

后台收到第一对图像就发起一次 `POST /tasks`，不是启动服务器时，也不是按空格时。
同一连接中的多次抓取不会自动分成新任务。换任务说明或重置场景后，应退出客户端再启动。
每次更新：SAM 跟踪（周期性 DINO 重定位）→ 规则判断 → 仅新确认故障时调用 Qwen 诊断。
默认规则：操作对象双视角持续不可见、控制器故障。持续故障不反复诊断，规则明确恢复 `clear` 后才重新触发。
不可见不等于确定掉落，框检测也不能单独证明插孔成功。扩展方法见 [规则说明](rules_zh.md)。

## 6. 在哪里看结果

- 服务器：`result/server/<会话ID>/results.jsonl`，同时输出 `RESULT` 日志。
- 本地：`result/client/<模式>/<时间_随机ID>/results.jsonl`，每次运行独立目录。
- 初始化记录包含任务、Qwen 目标规划和初始感知；更新记录包含框、可见性、规则证据、可选诊断和模型处理耗时。
- 顺序是“处理 → 服务端保存 → 回传 → 本地保存”；无故障时 `diagnosis: null` 是正常的。
- `outputs.save_annotated_frames: true` 可保存双视角带框 JPG；回传的是服务器路径，不会自动下载图片。
- 默认不保存原始 logits 或稠密 SAM mask；`client.record_observations` 保存图像/状态，供以后重放。
- `outputs.fsync: true` 加强断电持久性，但更慢。结果文件和图像需要自行清理、监控磁盘空间。

服务未加鉴权/TLS，只在可信内网或 SSH 隧道使用。结束客户端不卸载服务器模型。
以后接 VLA 时，替换观测来源并继续提交 `ObservationPacket`，沿用相同任务与结果协议。

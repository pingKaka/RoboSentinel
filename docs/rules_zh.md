# 检测规则、Prompt 与调试教程

这份文档集中说明“检测不对时改哪里、怎么改、怎么验证”。启动步骤见 [运行教程](system_walkthrough_zh.md)，参数全集见 [配置说明](config_reference_zh.md)。

第一次写规则，先看下面的“白话入门”和“可用数据”，不用先读完整 Python 代码。后面的第 5 节是需要动手实现时再查的代码教程。

## 白话入门：规则就是一句带条件的话

比如你看到机器人末端移动太快，会判断“可能有问题”。把这句话写成程序，需要说清楚三件事：

> **看什么：** 末端速度。
>
> **什么情况算异常：** 速度超过 0.25 m/s，并且持续至少 0.5 秒。
>
> **发现后怎么办：** 报告“持续超速”，再让 VLM 看当前图像、状态和证据，解释原因。

这里的阈值只是演示，不是要求你真的让机器人超速。规则只负责判断，VLM 负责解释，机器人不会因为规则返回异常就自动停下。

### 用六条观测理解一条规则

假设你的规则是上面的“持续超速”：

| 收到数据的时刻 | 末端速度 | 程序应该说什么 | 为什么 |
| --- | --- | --- | --- |
| 0 秒 | 0.10 m/s | 正常 | 没有超速 |
| 1 秒 | 0.30 m/s | 暂时不能确定 | 刚开始超速，先记下起点 |
| 1.3 秒 | 0.30 m/s | 暂时不能确定 | 只持续了 0.3 秒 |
| 1.6 秒 | 0.30 m/s | 发现异常 | 已持续 0.6 秒，超过要求 |
| 1.8 秒 | 没收到速度值 | 暂时不能确定 | 没有数据不能当作恢复正常 |
| 2 秒 | 0.10 m/s | 正常 | 确实收到恢复正常的测量 |

代码里这三句话分别写成：正常 `clear`、不能确定 `unknown`、发现异常 `confirmed`。它们不是模型置信度，只是你写的规则给出的判断。

**先把上面这种表列出来，再写代码。** 如果连每行应该返回什么都说不清，先不要急着调模型。

### 你可以先从这三种规则开始

| 想判断什么 | 需要看什么 | 容易理解的判断方式 |
| --- | --- | --- |
| 目标持续看不见 | 双视角的目标可见性 | 两个视角都看不见，持续超过 2 秒就报告 |
| 末端移动太快 | 末端实际速度 | 速度超过阈值，并持续一段时间就报告 |
| 机器人保护停机 | 控制器状态 | 状态是 protective_stop 就报告 |

第一种和第三种已经默认启用，第二种已有示例但默认未启用。先理解这三种，再考虑“抓空”“插错孔”等需要多种证据的规则。

## 现在到底有哪些数据能用

下面按**当前实际代码**列出。最重要的区别是：有些来自相机和模型，有些来自机器人；有字段不等于每次都有有效数值。

### A. 图像和物体信息：知道“看到了什么”

| 数据 | 通俗含义 | 规则里从哪里取 | 可以做什么 |
| --- | --- | --- | --- |
| 当前双相机图像 | 外部和腕部各一张彩色图 | `context.observation.frames["external"].image`、`["wrist"].image` | 自己增加图像分析；这是图像数组，不是文字描述 |
| 目标 ID | 同一任务中给每个目标起的编号 | `context.plan.targets` 中的 `target_id` | 把同一个目标在两个相机、不同时间的记录对应起来 |
| 目标名称 | 如白盖试剂瓶，由初始 VLM 输出 | 同一目标的 `name` | 查看模型选中了什么；不要假定名称永远固定 |
| 目标角色 | 被操作物体、目的地或夹爪 | 同一目标的 `role` | 优先检查 `manipulated_object`，不用写死“瓶子” |
| 是否可见 | 跟踪器当前是否找到这个目标 | `context.perception.tracks` 每条记录的 `visible` | 做目标持续不可见规则 |
| 属于哪个相机 | external 或 wrist | 每条 track 的 `camera` | 区分单视角遮挡和双视角均不可见 |
| 目标框 | 目标在图像中占据的矩形范围 | 每条 track 的 `box_xyxy` | 算图像中的位置、大小、移动趋势 |
| 目标轮廓 | SAM 分割结果的简化边界 | 每条 track 的 `contours` | 看形状/轮廓变化，或显示分割区域 |
| 感知状态 | 本轮感知输出是否完整 | `context.perception.status` | 判断数据是否可用于规则；当前后端成功更新时返回 complete |

目标框 `[x1, y1, x2, y2]` 的单位是**像素**，原点在图像左上角。中心可以自己算成 `((x1+x2)/2, (y1+y2)/2)`。框中心移动 20 像素，不等于物体移动 20 毫米。

外部和腕部图像是两套不同坐标，不能直接拿两张图的 x、y 相减来算实际距离。腕部相机会随机器人运动，所以图中目标移动也不一定代表物体实际移动。

`visible: true` 只代表跟踪器给出了有效结果，不保证身份绝对正确。轮廓最多保留 8 个简化外轮廓，不包含内部孔洞，不能当成精确测量工具。

### B. 机器人状态：知道“机器人实际怎么样”

以下字段都从 `state = context.observation.robot_state or {}` 读取，例如 `state.get("tcp_speed")`。缺少有效数据时应返回 unknown。

| 字段 | 通俗含义和单位 | 可以用于什么规则 | 注意事项 |
| --- | --- | --- | --- |
| `tcp_pose` | 工具中心点的位置和姿态；前三项 x/y/z 为米，后三项为旋转向量，单位弧度 | 已标定的工作区越界、末端位置异常 | 机器人基坐标系；不是图像坐标，也不是欧拉角 |
| `tcp_speed` | 工具中心点实际速度；前三项 m/s，后三项 rad/s | 超速、长时间不动 | 线速度大小要取前三项的向量模长 |
| `tcp_force` | 六维力/力矩；前三项 N，后三项 N·m | 持续受力过大、接触异常的候选规则 | 需确认 RTDE 坐标约定、负载/重力补偿和零点；数值大不自动等于碰撞 |
| `joint_position` | 6 个关节实际角度，弧度 | 关节工作范围检查 | 当前 UR 驱动直接取 RTDE 关节角，不是角度制 |
| `joint_velocity` | 6 个关节实际角速度，rad/s | 某关节超速或异常停止 | 不是末端的直线速度 |
| `gripper_position_normalized` | 夹爪读回的归一化位置 | 闭合程度、位置变化 | 不是米制开口宽度，也不是夹持力；开闭方向先按实际设备验证 |
| `gripper_action_bin` | 驱动记录的二值夹爪命令 | 知道程序是否发出了开/合意图 | **不是成功抓住物体的反馈**，也不是实际接触传感器 |
| `controller_state` | normal、protective_stop、emergency_stop、fault 等 | 保护停机、急停、控制器异常 | 来自安全模式映射；normal 不等于任务成功 |
| `safety_mode` | RTDE 原始安全模式数字 | 需要区分更具体的安全状态 | 使用前对照 UR 定义，不凭数字猜含义 |

当前遥操作适配器会尝试上传这些值。机器人未提供、夹爪未启用或读取失败时，值可能是 null，部分字段可能根本不存在。普通双视频回放没有这些真实机器人状态；重放遥操作保存的 `observations.jsonl` 才能带上原来记录的状态。

### C. 动作和坐标说明：知道“想让机器人做什么”

| 字段 | 含义 | 怎么用 |
| --- | --- | --- |
| `commanded_action` | 这一轮发给机器人驱动的动作字典 | 和实际运动对照，但先理解控制模式、字段、单位和延迟；指令不等于执行成功 |
| `control_space` | joint、tcp_force、tcp_position 等控制方式 | 决定如何理解动作字典 |
| `teleop_ee_pose` | 原驱动输出的末端位姿 | 可能是相对起点的位姿，不要直接当作 tcp_pose |
| `teleop_ee_pose_frame` | 这个位姿使用的坐标系 | 例如 base 或 episode_reference |
| `teleop_ee_pose_representation` | 姿态采用哪种表达 | xyz_m_euler_rad 或 xyz_m_rotation_vector_rad |
| `tcp_pose_frame` | tcp_pose 的坐标说明 | 当前为 base_xyz_m_rotation_vector_rad |
| `source`、`state_read_error` | 数据来源、读取出错说明 | 排查缺数据，state_read_error 只在相应读取异常时出现 |

“命令要求运动，但实际一直不动”可以作为后续规则方向，不过需要自己保存前后状态、考虑控制模式和延迟，当前没有开箱即用的“卡住”判定。

### D. 时间和历史：知道“持续多久了”

| 数据 | 从哪里取 | 实际含义 |
| --- | --- | --- |
| 本次观测时间 | `context.observation.timestamp_sec` | 本地采集的相对秒数，规则一般用它计时；不是服务器墙钟 |
| 机器人状态采样时间 | `state.get("timestamp_sec")` | 状态读取时的相对秒数，可能比图像观测稍晚 |
| 帧编号/时间 | `context.observation.frames[camera].frame_id` / `.timestamp_sec` | 服务端分配的处理序号/观测时间，不是相机硬件帧号 |
| 采集跨度 | `context.observation.synchronization.get("capture_span_sec")` | 本轮获取观测用时，不是精确的双相机曝光时差 |
| 状态读取延迟 | 同一 synchronization 中的 `state_read_delay_sec` | 读取附加状态距本轮图像采集结束的时间 |
| 历史跟踪结果 | `context.history` | 默认最近 15 秒的已处理感知结果，包括本次结果；由 runtime.buffer_seconds 控制 |

历史里有之前的目标框、可见性和轮廓，**没有历史原图、历史机器人位置/速度或历史动作**。如果想比较“上一帧的速度”和“当前速度”，要像超速示例保存 `self.previous` 一样，在规则实例里自己保存所需数值。

处理的是抽样观测，不是机器人控制循环的每一帧。只有 1 秒和 4 秒两个测量点，不能直接断言中间 3 秒一直异常。当前同步信息也不是硬件同步保证。

### E. 任务文字：知道“原本要做什么”

`context.task.task_id` 是任务编号，`context.task.description` 是任务文字。它们可以帮助选择规则或解释证据，但不会自动变成“抓取中/搬运中/插入中”的阶段信号。当前没有自动提供阶段进度、任务完成百分比或成功标签。

### 这些信息目前没有，不能直接拿来写规则

| 你可能想用的东西 | 目前情况 |
| --- | --- |
| 相机深度图、点云、物体三维位置 | 当前只传彩色图，未传深度/点云；需要额外接入与标定 |
| 每个孔的编号、三维坐标、插入深度 | 当前未提供；试管架整体框不能替代孔位数据 |
| 夹爪实际开口毫米数、真实夹持力、是否碰到物体 | 当前适配器未提供；归一化位置和开合命令不能替代 |
| 关节力矩、关节加速度、工具加速度 | 当前本地驱动有部分相关读数，但 teleop_adapter 尚未上传；不能直接在服务器规则里读取，需要先扩展适配器 |
| 稠密 SAM mask | TrackState 虽有 mask 字段，但当前后端返回 None；使用 contours 只能得到简化轮廓 |
| 每帧检测置信度 | TrackState 虽有 quality 字段，但当前后端未填有效值；不能据此设置信度阈值 |
| 稳定的目标属性字典 | attributes 在接口中存在，规划适配器也能接收，但当前 prompt 未要求模型输出它，通常为空；不能假定已有颜色、材质等固定字段 |
| 抓取成功、放置成功、当前操作阶段 | 尚未作为独立测量/信号提供，需要你实现对应判断或接入控制器阶段信息 |
| VLM 每帧的场景描述、历史故障分析 | 不在当前规则输入中；VLM 故障分析发生在规则之后 |

### 如何确认你自己的实验真的记录到了数据

打开本地 `result/client/teleop/<本次运行>/run_001/observations.jsonl`，查看非 initial 记录里的 `robot_state`；初始任务创建记录可能没有机器人状态，不能只看第一行就认为全程没有。

不想翻看很长的图像编码，可以在项目根目录运行下面命令，把最后一个路径换成实际记录路径。它只打印第一条含 robot_state 的更新记录，不打印图像：

```bash
python - result/client/teleop/你的运行目录/run_001/observations.jsonl <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    for line in source:
        item = json.loads(line)
        if item.get("kind") == "update" and item.get("robot_state"):
            print(json.dumps(item["robot_state"], ensure_ascii=False, indent=2))
            break
    else:
        print("没有找到带机器人状态的更新记录。")
PY
```

看到 `"tcp_speed": null`，意思是“这个样本没拿到速度”，不是“速度为零”。同样，检查视觉规则时，应在 `results.jsonl` 中看 `perception.tracks`，不是把原图中的肉眼判断当作跟踪器已经输出的结果。

## 1. 先搞清楚：谁负责什么

可以把系统理解成四个连续步骤：

```text
任务文字 + 第一对图像
    → Qwen：理解初始场景和任务，输出需要跟踪的物体（含 name）
    → GroundingDINO 定位，SAM 持续跟踪
    → Python 规则：判断是否出现异常
    → 仅新确认异常时，Qwen 解释原因并给建议
```

**当前不是 Qwen 每帧判断是否失败。** 自动模式需要规则确认异常才调用故障诊断；也可以在遥操作中按 F 人工报告故障，直接触发同一个分析 VLM。

## 快速调试：按 F 手动报告故障

规则还不完善时，可以由你判断“现在发生了故障”，让模型只负责分析原因：

1. 更新并重启服务器，再启动本地 `ur5e-video-teleop`，按空格开始检测，等待日志出现 `STREAMING`。
2. 看到异常时按一下 **F**（无需 Shift）。程序提交最近一轮采集的双视角图像和机器人状态，日志显示 `Manual failure queued with current observation`。
3. 服务器更新该观测的跟踪结果，**跳过该帧的规则判断，直接调用故障分析 VLM**。仍需任务初始化与模型服务正常可用。
4. 等待本地出现 `DIAGNOSIS`，在结果 JSONL 中检查 `manual_failure: true`、`rules.findings` 中的 `manual_failure` 和 `diagnosis`。报告明确注明人工来源，不声称自动规则已验证故障。
5. 分析完成后，中文故障分析在本地界面保留（滚轮查看长文本）。按 R 清除提示并重新允许自动报警；异常未消失时可能再次报警。松开再按 F 也可再次手动分析。普通帧仍按原规则运行。

按键可在本地 `cfg.yaml → video_teleop.sentinel.manual_failure_key` 修改，默认 `f`。不能与录制键、退出键或预览退出键 `q` 冲突。检测功能需保持 `sentinel.enabled: true`；键盘监听仍需要原遥操作所用的本地桌面会话。

按住只触发一次；已有手动请求待处理或正在分析时，再按会被拒绝并提示等待。初始化未完成、断线或检测关闭时也会明确拒绝。手动观测独立保留，不会被普通最新帧覆盖，但仍需等正在进行的服务器推理完成，因此不是零延迟。

这里的“此刻”是按键被主循环处理时最近采集到的观测，并非硬件同步的按键瞬间。请求不会重放过期任务或在连接失败后自动重试。退出客户端可能取消尚未返回的请求。

**F 只请求分析，不暂停遥操作、不停止机器人，也不会执行模型建议。** 模型仍应区分可见事实与人工报告，不能因人工触发而臆造原因。

手动标记随观测和结果保存；当前 `replay_detection.py` 默认不重放历史手动触发，重放仍按自动规则运行。

正式流程只有两处 VLM prompt：① 初始场景与任务理解；② 故障分析。`targets[].name` 是第一处 VLM 调用的输出，不是用户预先填写的物体清单，也不是第三处 VLM prompt。用户提供任务文字和图像，模型生成目标名称，程序再把名称交给 GroundingDINO 定位。

| 想调整什么 | 改哪里 | 在哪台机器改 |
| --- | --- | --- |
| 本次机器人要做什么 | `cfg.yaml → task.description` | 本地客户端 |
| Qwen 选错了跟踪对象 | `failure_detection/backends.py → QwenBackend.plan_tracking_targets()` 中的 prompt | 服务器 |
| 框不准、容易跟丢 | `cfg.yaml → backend` 中的检测与跟踪参数 | 服务器 |
| 多久算目标丢失、启用哪些规则 | `cfg.yaml → stages.rules.plugins` | 服务器 |
| 新增一种异常判断 | `failure_detection/rule_plugins/` 下新增插件并注册 | 服务器 |
| 故障解释太笼统、建议不合适 | `failure_detection/backends.py → QwenBackend.diagnose_failure()` 中的 prompt | 服务器 |
| 希望诊断看到额外数据 | `failure_detection/stages/diagnosis.py` 中传给模型的 payload | 服务器，必要时同时改客户端采集 |

配置和源码都不热更新。改服务器规则、阈值或 prompt 后重启服务，再启动客户端创建新任务；只改本地任务说明时，重启客户端即可。服务器重启会重新加载模型，单纯新建任务不会。

## 2. 最简单的调整：写清任务

例如，修改本地 `cfg.yaml`：

```yaml
task:
  id: white_bottle_to_right_hole
  description: "Pick the white-capped reagent bottle and place it in the far-right hole of the large test-tube rack."
```

尽量写清“哪个物体、做什么、放到哪里”，例如颜色、外观、可见的相对位置。不要只写“完成任务”，也不要要求识别图像中看不见的物体。

本地会把任务文字发给服务器。修改服务器的同名字段不会覆盖客户端已发送的任务。
遥操作按空格开始检测与录像后，客户端取得第一对图像时创建任务。再按空格停止，下一次开始会创建新任务；同一轮里的多次抓取不会自动分任务。回放客户端仍自动启动检测。

注意：写了“最右侧孔”并不意味着已经实现插错孔检测。文字说明、视觉定位和规则判断是三件不同的事。

## 3. 修改 Qwen 的两处 Prompt

### 3.1 第一处：初始场景与任务理解

打开 [backends.py](../failure_detection/backends.py)，找到 `plan_tracking_targets()`。传给 `self.infer()` 的第二个参数就是 prompt；第三个参数会自动附上任务文字，不需要把某个任务写死在 Python 中。

VLM 根据初始双视角图像和任务文字，自行生成跟踪对象的名称和角色。下面只是某个场景中可能产生的模型输出示例，不是需要写入配置或提前固定在 prompt 中的目标列表：

```json
{
  "targets": [
    {"target_id": "bottle", "name": "white-capped reagent bottle", "role": "manipulated_object"},
    {"target_id": "rack", "name": "large test-tube rack", "role": "destination"}
  ]
}
```

如果模型总是选择背景，可以在原 prompt 末尾追加以下英文句子，保留原来的 JSON 格式要求：

```python
'Prioritize the object being moved and its destination. '
'Assign manipulated_object to the object being moved, not the gripper or the rack. '
'Use short English names describing visible color, shape, and object category. '
```

在 Python 中，相邻字符串会自动拼接；把它们放在原 prompt 最后一句之后、第三个参数之前，注意逗号仍应位于完整 prompt 的最后。

不能随意更改的约定：
- 返回 JSON 对象，顶层字段是 `targets`，当前只接受 1～4 个对象。
- 每个对象必须有非空的 `target_id`、`name`、`role`，ID 不能重复。
- `role` 只能是 `manipulated_object`、`destination`、`gripper`。
- 模型输出的 `name` 会直接作为 GroundingDINO 的定位文本；prompt 可以要求模型生成简短、可见、具体的英文名词短语，但不应写死某次实验的目标名称。
- 默认丢失规则只检查 `manipulated_object`。角色分错时，规则可能监测错对象或一直 unknown。

如果确实要增加目标数量或角色，除了 prompt，还需要同步修改本函数的校验及相关规则，并评估显存和速度。

### 3.2 第二处：故障分析

同一文件中找到 `diagnose_failure()`。它会收到当前双视角图像、任务文字、跟踪结果、规则报告和当前机器人状态。

这是检测链路末端的故障分析，不是等整个机器人任务结束才分析；规则新确认异常或人工按键报告故障时调用它。人工路径复用这一处 prompt，没有增加第三处 VLM prompt。

可以在原 prompt 中追加：

```python
'In failure_analysis, distinguish observed facts from possible causes. '
'Name the triggered rule and cite available evidence. '
'If the evidence is insufficient, state what cannot be determined. '
'Keep recovery_suggestions short and require human verification before any motion. '
```

输出结构必须保留，例如：

```json
{
  "failure_analysis": "目标持续不可见，但现有图像不足以区分遮挡与掉落。",
  "recovery_suggestions": ["由操作人员检查双视角画面，确认目标实际位置。"]
}
```

`failure_analysis` 必须是字符串，`recovery_suggestions` 必须是字符串列表。不要要求输出 Markdown 表格来代替 JSON，否则解析会失败。

目前诊断**只看当前图像和当前证据**，不会自动看完整历史视频。虽然规则能读取感知历史，诊断 payload 并没有自动传递这些历史。要让模型分析一段变化，应在 `stages/diagnosis.py` 中明确增加时间摘要或选择历史证据，而不是只在 prompt 中写“请看之前的视频”。

### 3.3 哪些不是额外的 VLM prompt

- GroundingDINO 用的是初始理解阶段由 VLM 返回的 `name`，属于检测器的文本输入，不是额外的 VLM prompt。
- SAM 接收图像和定位框，不通过自然语言判断故障。
- `backend.vlm_max_new_tokens` 控制输出长度，太小可能截断 JSON；`vlm_image_max_side` 控制输入分辨率。
- 当前两处 prompt 写在 Python 中，尚未提供 `cfg.yaml` 的 prompt 配置字段。

## 4. 不写代码，先调已有规则

服务器 `cfg.yaml` 的对应部分默认类似：

```yaml
stages:
  rules:
    plugins:
      - name: target_lost
        options:
          seconds: 2.0
          max_gap_seconds: 2.0
      - controller_fault
```

这是局部示例，只修改现有 `stages.rules`，不要删除其他 stage，也不要另建重复的 `stages` 键。

`target_lost`：被操作对象在双视角都持续不可见才确认。任一视角可见就不算丢失；图像记录不完整则 unknown。
`seconds` 越大，越不容易因为短暂遮挡误报，但确认也更慢。`max_gap_seconds` 是允许的最大相邻观测间隔；超过它就重新计时，避免把断流当作持续不可见。

`controller_fault`：读取 `robot_state.controller_state`，如 protective_stop、emergency_stop、fault、error 会确认异常；缺失或不认识的状态返回 unknown。普通视频没有真实机器人状态，不能靠视频证明控制器正常。

规则状态的意思：

| 状态 | 通俗含义 | 对诊断的影响 |
| --- | --- | --- |
| clear | 有证据说明这一项正常 | 允许下次异常再次触发 |
| unknown | 信息不足，或还没达到持续时长 | 不触发，也不解除已确认故障的去重状态 |
| confirmed | 满足规则定义的异常条件 | 首次确认时调用 Qwen |

例如 `confirmed → confirmed` 只诊断一次；`confirmed → unknown → confirmed` 仍不会重诊断；`confirmed → clear → confirmed` 才会再次诊断。换新任务也会重置规则状态。

## 5. 从零编写一条故障检测规则

下面用“TCP 持续超速”完整走一遍。TCP 可以理解为机器人末端工具的位置；我们检查的是末端移动速度，不是关节转速。

这条示例已放在仓库中，但默认没有启用。你可以先阅读、运行它，再照同样的步骤编写自己的规则。**不需要为了学习规则而让真实机器人超速。**

### 5.1 第一步：用一句话定义故障

先不要写代码，写出能用数据验证的条件：

> 当末端线速度连续超过 0.25 m/s 达到 0.5 秒，报告“持续超速”；恢复到阈值以内时，报告正常。

再规定边界：

| 问题 | 本例的处理 |
| --- | --- |
| 速度恰好等于 0.25 m/s | 正常，因为规则使用“大于”，不是“大于等于” |
| 只超速一帧 | unknown，还没有达到持续时长 |
| 没有速度数据、数据格式错误或包含 NaN | unknown，并清除本规则的计时 |
| 两帧时间相隔超过 2 秒 | 重新计时，不能假定中间一直超速 |
| 时间倒退或重复 | 重新计时 |
| 速度恢复正常 | clear，下一次超速从头计时 |
| 一直超速 | 持续返回 confirmed；流水线负责避免每帧调用 VLM |

这里的数值只是演示阈值，不是设备安全限值。“连续”是基于实际收到的离散观测判断，不能保证两帧之间完全没有变化；应根据采样率和任务选择允许的间隔。

### 5.2 第二步：确认输入数据够不够

本例只需要两项输入：

```text
context.observation.timestamp_sec       观测时间，单位秒
context.observation.robot_state.tcp_speed
                                       [vx, vy, vz, wx, wy, wz]
```

Python 中 `robot_state` 是字典，实际读取方式是 `state.get("tcp_speed")`。前三项是线速度（m/s），后三项是角速度，本规则不使用后三项。

线速度大小为 `sqrt(vx² + vy² + vz²)`。例如 `[0.3, 0.4, 0, 0, 0, 0]` 的线速度为 0.5 m/s，不是简单相加得到 0.7。

先检查已有 `observations.jsonl` 中是否真的包含这个字段。普通视频没有机器人状态，不能验证速度规则；遥操作适配器会尝试从驱动观测中提取速度，缺失时为 null。不要把“字段存在但为 null”当作有效测量。

### 5.3 第三步：实现插件类

文件位置是 [failure_detection/rule_plugins/tcp_speed.py](../failure_detection/rule_plugins/tcp_speed.py)。该文件已经存在，完整实现如下；创建其他规则时使用新的模块名、类名和唯一的 `rule_id`，不要覆盖已有插件。

```python
"""Example only: monitoring is not a robot safety controller."""
import math

from failure_detection.pipeline.contracts import RuleFinding


class TCPSpeedRule:
    rule_id = "tcp_speed"

    def __init__(self, limit_mps=0.25, seconds=0.5, max_gap_seconds=2.0):
        self.limit = float(limit_mps)
        self.seconds = float(seconds)
        self.max_gap = float(max_gap_seconds)
        if any(not math.isfinite(v) or v <= 0 for v in (self.limit, self.seconds, self.max_gap)):
            raise ValueError("Speed and time limits must be finite and positive")
        self.since = self.previous = None

    def update(self, context):
        now = context.observation.timestamp_sec
        velocity = (context.observation.robot_state or {}).get("tcp_speed")
        try:
            if len(velocity) < 3:
                raise ValueError("Missing linear velocity")
            speed = math.hypot(*(float(v) for v in velocity[:3]))
            if not math.isfinite(speed) or not math.isfinite(now):
                raise ValueError("Nonfinite observation")
        except (TypeError, ValueError, OverflowError):
            self.since = self.previous = None
            return RuleFinding(self.rule_id, "unknown")
        if self.previous is not None and (now <= self.previous or now - self.previous > self.max_gap):
            self.since = None
        self.previous = now
        evidence = {"speed_mps": speed, "limit_mps": self.limit}
        if speed <= self.limit:
            self.since = None
            return RuleFinding(self.rule_id, "clear", evidence=evidence)
        if self.since is None:
            self.since = now
        confirmed = now - self.since >= self.seconds
        return RuleFinding(self.rule_id, "confirmed" if confirmed else "unknown",
                           "tcp_overspeed" if confirmed else None, evidence=evidence)
```

按执行顺序理解这段代码：

1. `__init__()` 接收配置里的阈值，检查它们是有限正数，初始化计时变量。
2. `update()` 每收到一帧观测调用一次。先读取时间和速度，数据不够就返回 unknown。
3. 时间不连续时，清掉旧的起点 `since`。
4. 速度正常时返回 clear，并清掉起点。
5. 第一次超速只记录起点。后续超速观测达到持续时长后才返回 confirmed。

`previous` 是上一帧时间，`since` 是本次连续超速的起点。它们保存在规则实例中；每个任务创建新的规则实例，因此不同任务不会共用计时。

`RuleFinding` 是交给流水线的“判断单”：

| 字段 | 本例 | 意义 |
| --- | --- | --- |
| rule_id | tcp_speed | 哪条规则做出的判断，必须唯一 |
| status | clear / unknown / confirmed | 正常 / 尚不确定 / 确认异常 |
| failure_type | tcp_overspeed | 故障类别，只有确认时填写 |
| evidence | speed_mps、limit_mps | 可复核的数值，供日志和 VLM 分析使用 |

不要在规则内部加载 VLM、发送机器人命令、等待人工输入或调用 `sleep()`。规则应快速返回；VLM 调用由流水线统一处理。证据使用普通 JSON 类型，不要直接放图像、NumPy 数组或 GPU 张量。

### 5.4 第四步：注册到配置

只在服务器现有 `cfg.yaml → stages.rules.plugins` 列表中追加：

```yaml
      - name: failure_detection.rule_plugins.tcp_speed:TCPSpeedRule
        options:
          limit_mps: 0.25
          seconds: 0.5
          max_gap_seconds: 2.0
```

路径可以拆成三部分：

```text
failure_detection.rule_plugins.tcp_speed : TCPSpeedRule
              Python 模块路径               类名
```

`options` 的键必须与构造函数参数一致。比如把 `limit_mps` 写成 `speed_limit`，就会创建失败。阈值放配置里，不需要每次修改代码。

现有 `target_lost`、`controller_fault` 可以保留，任一规则确认异常都可能触发分析。只想隔离调试本例时，可以在专用测试配置中仅启用本例，不要无意中修改正式实验的规则组合。不要把同一 `rule_id` 注册两次。

不需要修改 `my_backend.py`、规则引擎或 VLM prompt，插件加载器已经支持上述格式。

### 5.5 第五步：不用机器人、也不用模型，先验证

下面的命令在项目根目录运行，仅构造数据并调用 Python 规则，不连接硬件、不加载模型。配置文件尚未启用插件也可以运行。

```bash
python - <<'PY'
from types import SimpleNamespace
from failure_detection.rules import create_rules

rule = create_rules({"plugins": [{
    "name": "failure_detection.rule_plugins.tcp_speed:TCPSpeedRule",
    "options": {"limit_mps": 0.25, "seconds": 0.5, "max_gap_seconds": 2.0},
}]})[0]

samples = [
    (0.0, [0.10, 0, 0], "clear"),
    (0.1, [0.30, 0, 0], "unknown"),
    (0.4, [0.30, 0, 0], "unknown"),
    (0.7, [0.30, 0, 0], "confirmed"),
    (0.8, [0.30, 0, 0], "confirmed"),
    (0.9, [0.10, 0, 0], "clear"),
    (1.0, [0.30, 0, 0], "unknown"),
    (1.6, [0.30, 0, 0], "confirmed"),
    (1.7, None, "unknown"),
    (1.8, [float("nan"), 0, 0], "unknown"),
    (2.0, [0.30, 0, 0], "unknown"),
    (5.0, [0.30, 0, 0], "unknown"),
    (4.0, [0.30, 0, 0], "unknown"),
    (4.6, [0.30, 0, 0], "confirmed"),
    (4.7, [0.25, 0, 0], "clear"),
]
for timestamp, speed, expected in samples:
    context = SimpleNamespace(observation=SimpleNamespace(
        timestamp_sec=timestamp, robot_state={"tcp_speed": speed},
    ))
    finding = rule.update(context)
    print(timestamp, finding.status, finding.evidence)
    assert finding.status == expected, (timestamp, finding.status, expected)
print("PASS: speed rule")
PY
```

应以 `PASS: speed rule` 结束。这个示例故意覆盖正常、短暂超速、持续超速、恢复、再次故障、缺失数据、NaN、长间隔、时间倒退和阈值相等。

它只证明规则按预期返回状态，不验证 VLM、通信和机器人数据真实性。新规则正式保留时，应把这些场景写成 `failure_detection/tests/test_你的规则.py` 中的 unittest 测试，参考已有 [test_result_config.py](../failure_detection/tests/test_result_config.py)，再执行：

```bash
python -m unittest discover -s failure_detection/tests -v
```

### 5.6 第六步：验证完整检测链路

1. 把插件文件和配置部署到服务器，重启 `python scripts/serve.py`，等待 READY。
2. 选择含有效 `robot_state.tcp_speed` 的观测记录，在本地设置 `replay.observations` 并执行 `python scripts/replay_detection.py`。没有超过阈值的样本，不应出现本规则故障。
3. 查看每条 `rules.findings`：应出现 `rule_id: tcp_speed`、状态和速度证据。确认首次 confirmed 时有 `diagnosis`，恢复 clear 后再次超速能够再次分析。
4. 最后才进行正常速度的真机观测测试：空格开始检测，右侧查看 tcp_speed 的规则状态，核对记录中的速度是否与实际运动一致。

不要通过危险超速制造样本。需要验证 confirmed 分支时，先用上面的构造数据测试；已有离线记录也可在独立测试配置中选择适合该记录的演示阈值，明确标注这只是软件调试。

**F 不能验证你的规则是否有效。** F 会绕过该帧规则，直接调用 VLM；`replay.simulate_fault_at` 注入的是 controller_fault，也不验证速度规则。要确认本插件生效，必须在结果中看到 tcp_speed 自己的判断和证据。

**R 只重置流水线的诊断去重，并清除界面故障提示，不重建规则实例或清掉其计时。** 如果仍满足超速条件，下一次观测可能马上再次 confirmed。要让整个规则从空白状态开始，用空格停止再开始新任务。

### 5.7 再看一个视觉规则例子：目标持续不可见

已有实现位于 [rules.py](../failure_detection/rules.py) 的 `TargetLostRule`。它与速度规则的计时方法相似，但输入换成目标和双视角跟踪：

```text
从 VLM 的 plan.targets 中选择 role == manipulated_object 的目标
    → 检查这个 target_id 在 external 和 wrist 两个视角的 tracks
    → 任一视角可见：清除该目标的丢失计时
    → 两个视角记录完整、但均不可见：开始/继续计时
    → 不可见达到指定秒数：confirmed
    → 缺少某个视角或感知不完整：unknown，不把缺数据当丢失
```

这里不写死 `name == "bottle"`：目标名称由 VLM 生成，规则按角色选择对象，再用 `target_id` 关联跟踪结果。这样换成杯子等操作对象时不必重写规则。

需要检查多个目标时，使用以 `target_id` 为键的字典分别计时；一个目标恢复可见，只清除它自己的计时。默认规则的 confirmed 表示“持续不可观测”，不等于已经证明“掉落”。

### 5.8 写其他规则时可以读取什么

字段定义见 [contracts.py](../failure_detection/pipeline/contracts.py)：

| 输入 | 用途 |
| --- | --- |
| `context.task` | 任务 ID 和文字 |
| `context.plan.targets` | VLM 生成的目标 ID、名称和角色 |
| `context.observation.timestamp_sec` | 观测时间，用于持续时长判断 |
| `context.observation.robot_state` | 当前机器人状态字典，可能为 None |
| `context.observation.frames` | 当前双视角图像，服务端为 OpenCV BGR |
| `context.perception.tracks` | 当前各目标的 camera、visible、box_xyxy、contours 等 |
| `context.history` | 感知结果历史，不是历史视频或机器人状态序列 |

`box_xyxy` 是图像像素坐标，不是机器人坐标；`gripper_position_normalized` 是归一化值，不是米。不要把这些不同单位直接比较。

例如“抓空”可能需要夹爪反馈与物体运动证据；“插错孔”需要具体孔位识别、空间标定和操作阶段。先明确输入是否足够，再编写条件；任务文字或 VLM 的解释不能替代缺失的测量。

### 5.9 新规则写完后的检查表

- 条件是否能用已有数据复核，是否明确单位、阈值及持续时长？
- 缺失数据是否返回 unknown，而不是 clear 或 confirmed？
- 是否处理了时间倒退、重复时间、长间隔和非有限值？
- 是否有唯一 rule_id，配置的模块路径和构造参数是否正确？
- 证据是否足以让人理解为什么报警，且能保存为 JSON？
- 正常样本和异常样本是否都测过？是否验证恢复后可以再次触发？
- 是否明确它只是检测，不会自行停止机器人或执行恢复动作？

## 6. 一套固定调试流程

每次只改一种东西，使用相同视频或相同观测记录对比，保留修改前后的配置、prompt 和结果。

### 第一步：准备可查看的记录

服务器配置：
```yaml
outputs:
  server_dir: result/server
  console: true
  save_annotated_frames: true
  fsync: false
```

本地将 `client.record_observations` 设为 `true`，保留图像和状态以便复现。带框图片占空间，调试结束后可关闭并自行清理。

### 第二步：按问题选择测试方法

**只测试 prompt 或定位**：先停止服务器，避免额外加载一套模型挤满显存。在服务器设置 `check.image` 为真实场景图片，然后运行：

```bash
python scripts/check_backend.py --mode qwen
python scripts/check_backend.py --mode tracker --target "white-capped reagent bottle"
```

第一条会分别调用初始场景与任务理解、故障分析，打印结构化结果；目标名称由 VLM 生成。这里的诊断是人为构造的测试事件，不代表实际故障。
第二条是**独立排查 DINO/SAM 的特殊自检**：故意绕过 Qwen，才需要用 `--target` 提供测试短语。`check.target` 也只用于这种 tracker 自检模式，正式服务和完整回放不会用它预先指定目标。不要把该命令当作正式流程的配置步骤。
单图检查把同一张图当两个视角，也不会验证真实运动时的持续跟踪。

**测试完整工作流**：启动服务器后，本地配置双视角视频并执行：
```bash
python scripts/replay_detection.py
```

**重放真实状态**：将本地 `replay.observations` 指向之前保存的 `observations.jsonl`，再执行同一命令。此项非 null 时优先于配置的视频路径；切回视频测试时改回 null。重放不向机器人执行记录中的动作。

**检查故障诊断链路**：临时设置 `replay.simulate_fault_at: 3`，使用超过 3 秒的视频/记录，并保留 controller_fault 插件。回放会注入模拟故障，测试后改回 null。这只能验证触发和回传，不能验证视觉准确率。

### 第三步：按顺序查看结果

服务器记录位于 `result/server/<会话ID>/results.jsonl`。每行是一条 JSON，可以用编辑器打开；实时查看可执行：
```bash
tail -f result/server/实际会话ID/results.jsonl
```

依次看：
1. 首条初始化记录（`record_type: initialized`）的 `tracking_plan.targets`：是否选对物体，角色是否正确？
2. `perception.tracks` 和带框 JPG：框是否落在正确物体上，两个视角的 visible 是否合理？
3. `rules.findings`：是哪条规则、什么状态、evidence 是否支持该结论？
4. `diagnosis`：解释是否忠于证据？没有新确认事件时为 null 是正常的。

服务器保存成功后才回传；本地副本在 `result/client/<模式>/<运行目录>/results.jsonl`。带框图片回传的只是服务器路径，不会自动下载到本地。
当前记录保存的是结构化模型结果，不包含 Qwen 原始生成文本；遇到 JSON 解析失败，可以临时在 `infer()` 的 `return parse_json(text)` 前增加日志，调试后移除，并注意不要泄露敏感场景信息。

### 第四步：回归后再上真机

```bash
python -m unittest discover -s failure_detection/tests -v
```

新规则至少测试：正常、达到阈值、尚未达到持续时长、缺失数据、时间倒退、长间隔、故障恢复、再次故障。可参考 `failure_detection/tests/test_result_config.py` 和 `test_backend_rules.py`。这些测试通过只说明代码行为符合测试，不代表真实检测准确率。

最后才在受控条件下做低速真机实验。检测不负责运动许可或急停，Qwen 建议也不会自动执行；不要通过制造真实保护停机或危险动作来验证故障链路。

## 7. 常见问题速查

| 现象 | 优先检查 |
| --- | --- |
| 一开始就盯错物体 | 本地任务文字、规划 prompt、targets 的名称和角色 |
| 目标选对但一直没有框 | 初始画面、DINO 英文短语、box_threshold/text_threshold；先单图 tracker 测试 |
| 框跳到同类物体上 | 名称是否有辨别特征；当前每个目标取最高分框，没有额外的跨物体身份校验 |
| 短暂遮挡也报警 | 先确认框正确，再适当增加 target_lost.seconds |
| 肉眼看见丢失却不报警 | 是否另一视角仍可见、角色是否正确、观测间隔是否超过 max_gap_seconds |
| 规则一直 unknown | 是否缺状态、缺目标、持续时间不足或时间间隔太大；unknown 不等于正常 |
| confirmed 但本条没有诊断 | 是否此前已诊断且尚未 clear；查看前面的结果 |
| 诊断说得很肯定但证据不足 | 修改诊断 prompt 强调不确定性，同时检查规则证据是否可靠 |
| 修改后没生效 | 是否改错机器、是否重启服务和客户端、是否被 CLI 参数覆盖 |
| VLM JSON 报错 | 是否改坏字段/类型、输出是否被 max_new_tokens 截断；必要时查看原始生成文本 |
| 处理太慢 | 看 server_processing_ms；降低发送 FPS、Qwen 图像尺寸，减少不必要目标；更频繁 DINO 重定位通常更慢 |

调参不是越敏感越好：降低检测阈值可能增加错误框，缩短故障时长可能增加遮挡误报。每次修改都应同时回放“正常”和“故障”样本。

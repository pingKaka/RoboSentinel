# 检测规则、Prompt 与调试教程

这份文档集中说明“检测不对时改哪里、怎么改、怎么验证”。启动步骤见 [运行教程](system_walkthrough_zh.md)，参数全集见 [配置说明](config_reference_zh.md)。

## 1. 先搞清楚：谁负责什么

可以把系统理解成四个连续步骤：

```text
任务文字 + 第一对图像
    → Qwen：理解初始场景和任务，输出需要跟踪的物体（含 name）
    → GroundingDINO 定位，SAM 持续跟踪
    → Python 规则：判断是否出现异常
    → 仅新确认异常时，Qwen 解释原因并给建议
```

**当前不是 Qwen 每帧判断是否失败。** 如果规则没有确认异常，诊断 prompt 写得再详细也不会调用故障诊断。

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
任务是在客户端取得第一对图像时创建，不是按空格录制时创建；同一连接里的多次抓取不会自动变成新任务。

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

这是检测链路末端的故障分析，不是等整个机器人任务结束才分析；当前代码在规则新确认异常时调用它。

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

## 5. 扩展自己的检测规则

### 5.1 先启用现成示例

已有 [TCP 持续超速插件](../failure_detection/rule_plugins/tcp_speed.py)，默认没有启用。把以下项追加到现有 `stages.rules.plugins` 列表中：

```yaml
      - name: failure_detection.rule_plugins.tcp_speed:TCPSpeedRule
        options:
          limit_mps: 0.25
          seconds: 0.5
          max_gap_seconds: 2.0
```

含义：读取 `robot_state.tcp_speed` 的前三项，算出 TCP 线速度；持续超过 0.25 m/s 达到 0.5 秒才确认。缺少速度则 unknown，时间倒退或间隔过大重新计时。

这些数值只是示例，不是机器人的安全认证限值。只有含真实速度的观测才能验证该规则，普通视频回放不行。

### 5.2 自己写一个插件

最方便的方法是参考这个示例新建模块。需要遵守四个约定：

1. 类提供唯一、非空的 `rule_id`，例如 `grasp_empty`；不要与已有规则重复。
2. 构造函数接收 YAML 的 `options`，例如阈值和持续时长。每个任务会创建独立实例，计时状态可以存在实例变量中。
3. 实现 `update(context)`，每次收到观测都会被调用，返回 `RuleFinding`。
4. 在配置里用 `包.模块:类名` 注册，并确保服务器 Python 能导入模块。

返回值形式如下，实际判断条件需要自己实现：

```python
from failure_detection.pipeline.contracts import RuleFinding

# 数据不足时：
return RuleFinding(self.rule_id, "unknown")
# 已确认正常时：
return RuleFinding(self.rule_id, "clear")
# 满足异常条件时：
return RuleFinding(
    self.rule_id, "confirmed", "your_failure_type",
    evidence={"measured_value": measured_value, "threshold": self.threshold},
)
```

这些是 `update()` 内不同分支的示意，不是可以独立运行的完整插件。`evidence` 应放可复核的数值或文字，使用能转成 JSON 的普通类型，不要塞图像或 GPU 张量。

### 5.3 规则能读取什么

字段定义见 [contracts.py](../failure_detection/pipeline/contracts.py)：

| 输入 | 用途 |
| --- | --- |
| `context.task` | 任务 ID 和文字 |
| `context.plan.targets` | 目标 ID、名称和角色 |
| `context.observation.timestamp_sec` | 本次观测时间，用于持续时间判断 |
| `context.observation.robot_state` | 机器人状态，可能为 None |
| `context.observation.frames` | 当前双视角图像；传输解码后为 OpenCV BGR |
| `context.perception.tracks` | 各目标的 camera、visible、box_xyxy 等 |
| `context.history` | 缓存的感知结果，不是历史视频或机器人状态序列 |

使用观测时间计算持续时长，不要用服务器当前墙钟。时间倒退、长间隔、NaN、缺失状态都要处理。
`box_xyxy` 是图像像素坐标，不是机器人坐标；`gripper_position_normalized` 是归一化值，不是米。

### 5.4 不要把无法观测的事硬写成规则

“插错孔”通常需要孔位识别、空间标定或操作阶段信息；“抓空”通常需要夹爪反馈与物体运动证据；“掉落”也不能仅用某个视角看不见来判断。

建议先写一句可验证的定义，例如“目标在两个视角均不可见至少 2 秒”，再检查现有输入是否足够。不够时先补采集和标定，不要只改 prompt 让模型给出更肯定的结论。

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

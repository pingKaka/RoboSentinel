# RoboSentinel 规则编写与 Pipeline 教程

本文教你从零写出第一条人工规则。部署和视频回放请先阅读仓库根目录的 `README.md`。

第一次阅读只看第 1 到第 5 节，后面的内容用于查参数。

## 1. 用一句话理解规则

一条规则就是四句话：

```text
观察谁？
读取什么数据？
什么时候开始检查？
什么情况算故障？
```

例如：

```text
观察瓶子和杯子。
读取两个掩码的 IoU。
两者都可见时开始检查。
IoU 连续超过 50% 达 0.5 秒，并且至少出现 3 帧，算故障。
```

JSON 只是把这四句话写成机器能够严格检查的格式。

## 2. 最快写出第一条人工规则

复制现成示例：

```bash
cp config/manual_monitoring_plan.example.json config/my_monitoring_plan.json
```

这个示例已经是一条完整规则：“瓶子和杯子的掩码 IoU 不得持续超过 50%”。

先不要修改，直接验证：

```bash
python scripts/validate_monitoring_plan.py config/my_monitoring_plan.json
```

看到以下内容表示文件可用：

```json
{
  "status": "valid",
  "rule_ids": ["object_destination_overlap_too_high"]
}
```

然后在 `config/detection.yaml` 中选择人工计划：

```yaml
rules:
  mode: manual
  schema_version: "1.0.0"
  manual_plan: config/my_monitoring_plan.json
  max_generation_attempts: 2
  max_rules: 6
```

重启服务器后，这条规则就会进入实时 Pipeline。`manual` 模式不会调用 Qwen3-VL 写规则。

## 3. 只改四个地方就能复用示例

打开 `config/my_monitoring_plan.json`。初学者先只改以下内容：

### 改目标名称

```json
"name": "white capped reagent bottle"
```

这是 GroundingDINO 用来找物体的英文短语。写颜色、形状、材质等可见特征。

### 改规则名称

```json
"rule_id": "object_destination_overlap_too_high"
```

它只是规则的唯一名字，使用小写字母、数字和下划线。

### 改阈值

```json
"condition": {"op": "gt", "feature": "overlap", "value": 0.5}
```

`0.5` 就是 50%。如果要改成 30%，写 `0.3`。

### 改确认时间

```json
"duration_sec": 0.5,
"minimum_samples": 3
```

表示异常必须连续持续 0.5 秒，并且至少出现 3 帧。

每次修改后都运行验证命令。验证通过只代表格式和引用正确，不代表阈值一定合理。

## 4. 用人话拆解完整规则

下面是示例中的关键部分：

```json
"feature_bindings": [
  {
    "alias": "overlap",
    "feature": "relation.mask_iou",
    "entities": ["object_1", "destination"],
    "camera": "external",
    "reason": "衡量两个掩码的整体重叠程度。"
  }
],
"activation": { ... },
"condition": {"op": "gt", "feature": "overlap", "value": 0.5}
```

翻译成自然语言：

```text
把“外部相机中物体1和目标区域的掩码IoU”取名为 overlap。
activation 成立时开始检查。
overlap 大于 0.5 时认为出现异常。
```

`alias` 只是一个短名称。后面的表达式使用 `overlap`，不用重复写完整特征路径。

## 5. 一帧是怎样被判断的

每一帧按这个顺序处理：

```text
1. GroundingDINO/SAM 给出目标掩码
2. 标准特征层算出 overlap 等数值
3. 检查 activation
4. activation 成立后检查 condition
5. condition 成立后开始计时和数帧
6. 时间和帧数都达到要求后返回 confirmed
```

结果含义：

- `clear`：规则不适用，或者条件明确正常；
- `unknown`：看不清、缺数据、历史不足，或者正在等待持续确认；
- `confirmed`：异常持续达到要求。

缺数据不会被当成故障。

## 6. 完整 Pipeline（查阅）

```text
任务描述 + 初始 external/wrist 图像
    -> 选择规则来源 generated/manual/merged
    -> 得到 TrackingPlan + RulePackage
    -> GroundingDINO 用英文目标名称寻找初始框
    -> SAM2.1 根据初始框逐帧输出二值掩码
    -> StandardFeatureExtractor 计算标准特征
    -> DeclarativeRule 解释执行 activation 和 condition
    -> StreamingRuleEngine 汇总 RuleFinding
    -> RuleReport.failure_detected
```

VLM 只在任务初始化时运行。流式阶段不会逐帧询问 VLM，也不会执行 VLM 生成的 Python。

## 7. 三种规则来源

在 `config/detection.yaml` 设置：

```yaml
stages:
  rules:
    mode: generated
    schema_version: "1.0.0"
    manual_plan: null
    max_generation_attempts: 2
    max_rules: 6
```

### generated

Qwen3-VL 根据任务描述和初始双相机画面生成目标与规则。`manual_plan` 保持 `null`。适合开放任务和快速实验。

### manual

完全使用人工 JSON，Qwen3-VL 不参与任务规划。配置示例：

```yaml
mode: manual
manual_plan: config/my_monitoring_plan.json
```

适合目标固定、规则已经过验证、希望结果完全可重复的部署。服务器仍可常驻三模型，但本次任务不会调用 VLM。

### merged

人工计划提供固定目标和必须执行的规则；VLM 必须保留相同目标，再生成补充规则。最终顺序为“VLM 规则在前，人工计划在后”。

```yaml
mode: merged
manual_plan: config/my_monitoring_plan.json
```

若目标 ID/名称/角色被 VLM 改动、规则 ID 重复或总规则数超过 `max_rules`，任务初始化直接失败。

## 8. 人工文件结构

先复制示例：

```bash
cp config/manual_monitoring_plan.example.json config/my_monitoring_plan.json
```

文件顶层必须包含 `targets` 和 `rule_package`：

```json
{
  "targets": [],
  "rule_package": {
    "schema_version": "1.0.0",
    "task_type": "pick_and_place",
    "rules": []
  }
}
```

写完后离线校验，不需要 GPU：

```bash
python scripts/validate_monitoring_plan.py config/my_monitoring_plan.json
```

## 9. 跟踪目标参数

```json
{
  "target_id": "object_1",
  "name": "white capped reagent bottle",
  "role": "manipulated_object",
  "attributes": {}
}
```

- `target_id`：规则内部稳定 ID，只能使用小写字母开头的小写字母、数字和下划线，最长 64 字符。
- `name`：GroundingDINO 使用的英文视觉短语。必须描述可见属性，不能写“目标物体”这类抽象词。
- `role`：`manipulated_object`、`destination` 或 `gripper`。
- `attributes`：可选元数据，当前跟踪器和规则执行器不读取。

目标总数必须为 1 到 4。人工规则中出现的实体 ID 必须在这里声明。

## 10. RulePackage 参数

- `schema_version`：当前固定为 `1.0.0`。
- `task_type`：`pick`、`transport`、`place`、`pick_and_place` 或 `other`。
- `rules`：规则数组；数量不能超过配置的 `max_rules`。

每条规则包含：

- `rule_id`：唯一规则 ID。
- `failure_type`：确认故障后输出的机器可读类型。
- `constraint_type`：`during_execution` 或 `upon_completion`，当前用于解释和审计，不自动产生阶段信号。
- `description`：用通俗语言描述正常约束。
- `feature_bindings`：声明规则读取哪些标准特征。
- `activation`：什么时候需要检查该约束。
- `condition`：什么条件表示约束被违反。
- `duration_sec`：违反条件必须连续持续的秒数，可为 0。
- `minimum_samples`：违反条件必须连续出现的最少帧数，范围 1 到 300。
- `unknown_policy`：当前固定为 `unknown`。
- `explanation_template`：确认时使用的解释前缀；实际值、条件、时间和帧数会自动追加。

## 11. FeatureBinding 参数

```json
{
  "alias": "overlap",
  "feature": "relation.mask_iou",
  "entities": ["object_1", "destination"],
  "camera": "external",
  "reason": "衡量两个掩码的整体重叠程度。"
}
```

- `alias`：本条规则表达式使用的局部名称。
- `feature`：标准特征名。
- `entity`：单实体特征使用。
- `entities`：关系特征使用，必须是两个不同目标 ID。顺序影响 `coverage_a/coverage_b` 和 `delta_x/delta_y`。
- `camera`：`entity.camera.*` 和 `relation.*` 必填，当前通常是 `external` 或 `wrist`。
- `reason`：为什么该特征能支持这条约束，最终会进入 evidence。

### 单实体特征

- `entity.visible_any`：任意相机可见。
- `entity.missing_all_cameras`：所有相机都明确不可见。
- `entity.visible_camera_count`：可见相机数量。
- `entity.camera.visibility`：某相机中的 `visible/not_visible/unknown`。
- `entity.camera.mask_centroid_x/y`：掩码质心归一化坐标。
- `entity.camera.mask_area_ratio`：掩码面积占整幅图像比例。
- `entity.camera.mask_perimeter_normalized`：轮廓周长除以图像对角线。
- `entity.camera.mask_component_count`：掩码连通域数量。
- `entity.camera.mask_compactness`：掩码紧致度。
- `entity.camera.tracking_quality`：最近一次 GroundingDINO 初始化置信度。

### 关系特征

- `relation.delta_x/y`：实体 B 质心减实体 A 质心。
- `relation.centroid_distance`：两个归一化质心的欧氏距离。
- `relation.mask_iou`：交集面积除以并集面积。
- `relation.mask_intersection_ratio`：交集面积除以整幅图像面积。
- `relation.mask_coverage_a`：交集除以 `entities[0]` 的掩码面积。
- `relation.mask_coverage_b`：交集除以 `entities[1]` 的掩码面积。

关系特征只在同一相机内两个目标都有有效 SAM 掩码时存在。否则结果为 `unknown`。

### 机器人特征

- `robot.tcp_x/y/z`：TCP 在所声明坐标系中的三维位置，单位米。
- `robot.gripper_position_normalized`：夹爪归一化位置 `[0,1]`。项目不假设 0 或 1 必然代表张开，使用前必须确认具体驱动方向。

## 12. 所有算子

### eq

```json
{"op": "eq", "feature": "visible", "value": true}
```

实际值与 `value` 完全相等时为真，适合布尔值和字符串。

### gt / lt

```json
{"op": "gt", "feature": "overlap", "value": 0.5}
```

分别表示严格大于和严格小于，不包含边界。

### between

```json
{"op": "between", "feature": "tcp_z", "min": 0.1, "max": 0.3}
```

表示 `min <= value <= max`，包含两端。

### changed

```json
{
  "op": "changed",
  "feature": "tcp_z",
  "delta": 0.05,
  "direction": "increase",
  "window_sec": 1.0
}
```

取窗口首尾有效值之差。`direction` 可为 `increase`、`decrease` 或 `any`。它不是累计路程。历史不足完整窗口时为 `unknown`。

### stable

```json
{"op": "stable", "feature": "tcp_z", "tolerance": 0.01, "window_sec": 1.0}
```

窗口内 `最大值-最小值 <= tolerance` 时为真。历史不足时为 `unknown`。

### all / any

```json
{
  "op": "all",
  "args": [
    {"op": "eq", "feature": "a_visible", "value": true},
    {"op": "gt", "feature": "overlap", "value": 0.5}
  ]
}
```

`all` 是三值 AND：存在 False 即 False，否则存在 unknown 即 unknown。`any` 是三值 OR：存在 True 即 True，否则存在 unknown 即 unknown。

## 13. activation 与 condition

`activation` 表示规则适用阶段，`condition` 表示故障。不要把正常条件写入 `condition`。

例如“两个物体重叠不得超过 50%”：

```json
"activation": {
  "op": "all",
  "args": [
    {"op": "eq", "feature": "object_visible", "value": true},
    {"op": "eq", "feature": "destination_visible", "value": true}
  ]
},
"condition": {"op": "gt", "feature": "overlap", "value": 0.5}
```

若 activation 为 False，规则是 `clear`；activation 或 condition 无法计算时是 `unknown`；两者均满足后才进入持续确认。

## 14. 持续确认

违反条件首次出现时开始计时并累计样本。只有同时满足：

```text
elapsed >= duration_sec
sample_count >= minimum_samples
```

才返回 `confirmed`。条件恢复、规则不再激活或证据缺失都会清空计时，防止断续异常被错误累积。

## 15. RuleReport 汇总

- 任意规则 `confirmed`：`failure_detected=true`。
- 无确认但至少一条 `unknown`：`failure_detected=null`。
- 全部 `clear`：`failure_detected=false`。

每条 evidence 包含约束、特征值、阈值、特征原因、持续时间、样本数和解释。

## 16. 常见错误

- 用框面积或框 IoU 写规则：标准层只允许掩码特征。
- 把“正常条件”写成 condition：condition 必须表达故障。
- 关系特征没有指定 camera：Schema 会拒绝。
- `coverage_a` 实体顺序写反：检查 `entities[0]` 是否是被覆盖对象。
- 用 `changed` 判断累计运动：它只看窗口首尾差。
- 夹爪 0/1 方向未经设备确认：可能导致 activation 相反。
- `duration_sec` 大于历史缓冲：时序证据会长期不足。`runtime.buffer_seconds` 必须覆盖最长窗口。

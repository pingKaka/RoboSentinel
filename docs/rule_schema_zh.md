# 可解释故障规则规范（v1.0.0）

本项目借鉴 Code-as-Monitor 的“由 VLM 根据任务生成监控约束、运行时确定性执行”思路，但不执行 VLM 生成的任意 Python。VLM直接声明 activation 和 condition 所需的特征、对象、算子、阈值和原因；确定性编译器补全内部 alias 与 binding，再由严格 Schema 校验。

机器可读结构定义位于 `failure_detection/rule_dsl/rule_schema_v1.json`；Python 校验器在此基础上继续检查目标 ID、特征绑定和引用关系。

## 执行边界

1. 任务开始时，VLM 查看初始多相机图像和任务描述，一次性生成跟踪目标与规则包。
2. 编译器把简洁的 VLM 输出转换成运行时 DSL；Schema 校验目标 ID、标准特征、算子、阈值和解释字段，失败最多重新生成两次。
3. GroundingDINO 和 SAM2.1 持续输出跟踪掩码，标准特征层逐帧计算特征。
4. 规则引擎逐帧执行固定规则，只输出 `clear`、`confirmed` 或 `unknown`。
5. 特征缺失或证据不足一律为 `unknown`，不作为故障。

## 规则结构

每条规则必须包含：

- `rule_id`：规则唯一 ID。
- `failure_type`：确认时输出的故障类型。
- `constraint_type`：`during_execution` 或 `upon_completion`。
- `description`：自然语言约束。
- `feature_bindings`：特征别名、实体/相机绑定及选择原因。
- `activation`：规则何时生效。
- `condition`：生效后什么情况属于违反约束。
- `duration_sec`、`minimum_samples`：时间与样本数确认条件。
- `unknown_policy`：v1 固定为 `unknown`。
- `explanation_template`：确认故障时的解释。

```json
{
  "schema_version": "1.0.0",
  "task_type": "pick_and_place",
  "rules": [
    {
      "rule_id": "grasp_failed",
      "failure_type": "object_not_grasped",
      "constraint_type": "during_execution",
      "description": "夹爪闭合后，目标物体应保持在夹爪附近。",
      "feature_bindings": [
        {
          "alias": "opening",
          "feature": "robot.gripper_position_normalized",
          "reason": "用于识别夹爪已经执行闭合动作。"
        },
        {
          "alias": "object_gripper_distance",
          "feature": "relation.centroid_distance",
          "entities": ["object", "gripper"],
          "camera": "external",
          "reason": "用于判断目标是否随夹爪一起运动。"
        }
      ],
      "activation": {"op": "lt", "feature": "opening", "value": 0.2},
      "condition": {"op": "gt", "feature": "object_gripper_distance", "value": 0.15},
      "duration_sec": 0.5,
      "minimum_samples": 3,
      "unknown_policy": "unknown",
      "explanation_template": "夹爪已闭合，但目标与夹爪距离持续超过阈值。"
    }
  ]
}
```

## 简单算子

- 比较：`eq`、`gt`、`lt`、`between`
- 时序：`changed`、`stable`
- 组合：`all`、`any`

运行证据同时保存实际特征值、阈值表达式、特征选择原因、持续时间和样本数。解释来自这些实际证据，不在视频流中再次调用 VLM。

## v1 范围

首版用于 `pick`、`transport`、`place` 和 `pick_and_place`，优先生成 `target_lost`、`grasp_failed`、`place_failed`。规则自适应与阈值拟合不属于 v1，待积累正确任务轨迹后单独加入。

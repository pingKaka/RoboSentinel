# 项目工作流程

## 1. 场景和任务理解

任务开始时，VLM读取任务描述和初始双相机画面，输出需要跟踪的对象，例如：

```json
{
  "targets": [
    {"target_id":"object","name":"white-capped bottle","role":"manipulated_object"},
    {"target_id":"destination","name":"test-tube rack","role":"destination"}
  ]
}
```

VLM只回答“跟踪什么”，不负责判断故障。

## 2. 感知和规则判断

GroundingDINO根据对象名称找到初始位置。之后每收到一组新图像，SAM利用内部记忆更新mask和轨迹；目标丢失时可再次调用GroundingDINO初始化。

规则插件读取统一的 `RuleContext`：

- 任务描述；
- VLM生成的跟踪计划；
- GroundingDINO和SAM的当前轨迹及历史窗口；
- 可选机器人状态。

项目不内置规则。每项实验可以选择几何规则、统计模型、学习型异常检测器或它们的组合。

## 3. 故障分析和恢复建议

只有规则产生确认的异常事件时，VLM才接收当前画面、当前对象状态、完整规则报告和可选机器人状态，输出：

```json
{
  "failure_analysis":"故障原因或证据不足说明",
  "recovery_suggestions":["恢复建议一","恢复建议二"]
}
```

规则没有可靠结果时，VLM必须保留不确定性，不能自行补造故障。

## 机器人状态接口

机器人状态由 `RobotStateProvider` 提供。关闭时不影响视觉流程；启用时可以提供TCP、关节、夹爪、力觉或控制器状态。具体机器人和执行策略不属于主流程。

## 代码对应关系

- `pipeline/contracts.py`：阶段之间的数据格式；
- `pipeline/interfaces.py`：Planner、Tracker、Rule、Diagnoser和机器人状态接口；
- `pipeline/orchestrator.py`：逐帧串联整个流程；
- `pipeline/rule_engine.py`：加载和执行规则插件；
- `stages/planning.py`：VLM规划适配器；
- `stages/perception.py`：在线GroundingDINO+SAM适配器；
- `stages/diagnosis.py`：VLM诊断适配器；
- `state/`：可选机器人状态接口。

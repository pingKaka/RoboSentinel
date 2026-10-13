# 接口说明

主接口位于 `failure_detection.pipeline.interfaces`。

- `ScenePlanner`：任务和初始场景画面变成跟踪计划；
- `OnlineObjectTracker`：初始化后逐帧更新对象轨迹；
- `StreamingFailureRule`：每次新观测到达时更新规则状态；
- `RuleEngine`：汇总所有规则；
- `FailureDiagnoser`：规则报告变成故障解释和恢复建议；
- `RobotStateProvider`：可选的机器人状态来源。

所有组件只通过 `pipeline/contracts.py` 中的数据对象通信。

# RoboSentinel 工作进度记录（2026-11-10）

## 1. 当前目标

项目已收缩为独立的机器人操作故障检测系统：接收双相机视频和少量机器人状态，逐帧提取标准特征，执行可解释规则并输出故障报告。

当前明确不包含：

- VLA 控制、动作规划或重规划；
- 故障恢复动作；
- 在线 VLM 逐帧判断；
- 规则自动拟合和自我迭代。

## 2. 今天完成的工作

### 2.1 收缩故障检测流水线

当前主流程为：

```text
任务描述 + 初始双相机场景
        -> Qwen3-VL 生成跟踪目标和规则包
        -> GroundingDINO 初始化目标
        -> SAM2.1 在线掩码跟踪
        -> 标准特征层逐帧提取特征
        -> 声明式规则引擎
        -> RuleReport
```

已删除或停用事后 VLM 诊断、恢复建议和机器人动作相关链路。`RuleReport` 是自动故障检测的最终输出，操作者手动上报也会转换成统一的确认结果。

### 2.2 建立标准特征层 v2.0.0

机器人特征仅保留：

- `robot.tcp_position_m`：TCP 三维位置；
- `robot.tcp_frame`：TCP 坐标系；
- `robot.gripper_position_normalized`：归一化夹爪开合位置。

视觉特征改为直接来自 SAM 掩码，不再使用检测框近似：

- `mask_centroid_xy`
- `mask_area_ratio`
- `mask_perimeter_normalized`
- `mask_component_count`
- `mask_compactness`
- `tracking_quality`

同相机实体关系包括：

- `delta_xy`
- `centroid_distance`
- `mask_iou`
- `mask_intersection_ratio`
- `mask_coverage_a`
- `mask_coverage_b`

GroundingDINO 的框仅用于初始化 SAM 和界面显示，不进入标准规则特征。

主要文件：

- `failure_detection/features/contracts.py`
- `failure_detection/features/catalog.py`
- `failure_detection/features/extractor.py`
- `docs/standard_features_zh.md`

注意：`result/feature_collection/` 下的旧结果属于 v1 Schema，不能作为 v2 特征结果使用。今天按照要求没有重新运行视频特征验证。

### 2.3 调研并借鉴 Code-as-Monitor

参考工作：Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection，CVPR 2025，arXiv `2412.04455v3`。

CaM 的主要做法：

- VLM 根据任务生成执行中约束和完成时约束；
- 将相关实体抽象成点、线、面等几何元素；
- 为每个子目标生成 NumPy/Python 监控代码；
- 监控器读取当前及历史实体位置、末端轨迹和完成状态；
- 通过白盒路径测试检查生成代码，错误时重新生成；
- 阈值来自任务知识库或 VLM 先验。

本项目借鉴的是“VLM 生成约束，运行时确定性执行”的方法。不同之处是：本项目不执行 VLM 生成的任意 Python，而是生成受限 JSON DSL。这是针对当前工程的安全性和可维护性适配，不是 CaM 原始实现。

### 2.4 建立可解释规则 Schema v1.0.0

规则包包含：

- `task_type`
- `rule_id` 和 `failure_type`
- 自然语言约束描述
- `during_execution` 或 `upon_completion`
- 所选特征、实体/相机绑定以及每个特征的选择原因
- 激活条件 `activation`
- 违反条件 `condition`
- 持续时间和最少样本数
- 缺失证据策略
- 解释模板

当前允许的简单算子：

- 比较：`eq`、`gt`、`lt`、`between`
- 时序：`changed`、`stable`
- 组合：`all`、`any`

机器可读定义：

- `failure_detection/rule_dsl/rule_schema_v1.json`

语义校验和执行代码：

- `failure_detection/rule_dsl/contracts.py`
- `failure_detection/rule_dsl/schema.py`
- `failure_detection/rule_dsl/runtime.py`
- `docs/rule_schema_zh.md`

### 2.5 接入 VLM 规则生成

任务初始化时，Qwen3-VL 一次性生成：

1. 需要跟踪的目标；
2. 与目标 ID 一致的规则包；
3. 针对当前任务和场景的特征选择理由。

生成结果会检查：

- 目标 ID、角色和重复项；
- 特征是否在允许目录内；
- 实体是否确实存在于跟踪计划；
- 关系特征是否绑定两个有效实体；
- 算子和阈值是否合法；
- 规则引用的特征别名是否存在。

校验失败时最多重新生成两次。VLM 只在初始化时调用，不参与逐帧判断。

主要文件：

- `failure_detection/backends.py`
- `failure_detection/stages/planning.py`
- `failure_detection/pipeline/contracts.py`
- `my_backend.py`

### 2.6 实现确定性规则执行

规则引擎逐帧读取 `FeatureContext`，返回：

- `clear`：规则未激活或约束当前满足；
- `confirmed`：违反条件满足持续时间和样本数要求；
- `unknown`：掩码、关系、机器人状态或历史样本不足。

缺失证据不会被当作故障。确认结果中的 evidence 包含：

- 实际特征值；
- 激活条件和违反条件；
- 阈值；
- 特征选择原因；
- 持续时间和样本数；
- 最终解释文本。

生成的完整规则包会写入服务端任务初始化记录，方便后续审计和复现。

旧的静态 `TargetLostRule/create_rules` 路径已经移除，部署中只有一条规则来源：VLM 规则包。

## 3. 当前配置

`config/detection.yaml` 已切换为：

```yaml
stages:
  features:
    schema_version: "2.0.0"
  rules:
    mode: "generated"
    schema_version: "1.0.0"
    max_generation_attempts: 2
    max_rules: 6
```

VLM 输出上限提高为 `1536` tokens，以容纳目标、规则、特征理由和解释字段。

首版面向 `pick`、`transport`、`place` 和 `pick_and_place`，优先生成：

- `target_lost`
- `grasp_failed`
- `place_failed`

这些名称是推荐项，不是硬编码的固定规则；具体特征和阈值由 VLM 根据任务与场景选择。

## 4. 验证状态

使用环境：`robofac-infer`，Python 3.10。

已完成：

- Python 编译检查；
- JSON Schema 格式检查；
- 项目配置和模型路径检查；
- 32 项单元及集成测试全部通过；
- `git diff --check` 通过。

重点覆盖：

- 非法特征、实体和算子被拒绝；
- `changed/stable/all/any` 等算子行为；
- 缺失特征传播为 `unknown`；
- 持续时间和最少样本数确认；
- evidence 包含实际值和阈值；
- VLM 非法输出触发重新生成；
- 生成规则能够被规则引擎编译执行；
- 原有网络传输、人工故障和标准特征测试没有回归。

本轮没有执行真实 Qwen3-VL、GroundingDINO 和 SAM2.1 的端到端 GPU 视频推理。模型目录在配置检查中均存在，但仍需用真实场景检查 VLM 输出质量和规则阈值合理性。

测试命令：

```bash
conda run -n robofac-infer \
  python -m unittest discover -s failure_detection/tests -v

conda run -n robofac-infer \
  python failure_detection/run_pipeline.py
```

## 5. 已知限制

1. 当前规则阈值来自 VLM 先验，没有从成功轨迹中拟合。
2. 当前没有显式任务阶段状态；`pick/transport/place` 阶段主要由夹爪状态、TCP 变化和视觉关系间接判断。
3. 视觉关系只在同一相机同时获得两个有效掩码时存在，否则为 `unknown`。
4. 没有相机标定和三维重建，距离与重叠关系目前是二维归一化掩码特征。
5. 尚未对白盒规则做真实视频回放评估，也没有统计误报率和漏报率。
6. 尚未实现规则版本存储、成功轨迹数据集和自适应更新审批机制。

## 6. 下一步建议

### 第一步：真实初始化验证

从 `video/` 或真实双相机场景截取初始帧，运行 Qwen3-VL 规则生成，检查：

- 是否选择了正确目标；
- 是否引用了可采集的特征；
- 是否能稳定通过 Schema；
- 是否生成适合任务阶段的规则；
- 阈值是否处于合理范围。

先只检查生成规则，不急于评价故障检测准确率。

### 第二步：规则回放评估

准备少量成功和已知失败视频，逐帧保存：

- 标准特征；
- 每条规则的激活状态；
- 条件计算结果；
- `clear/unknown/confirmed`；
- 实际值、阈值和解释。

据此确认 `target_lost/grasp_failed/place_failed` 是否能在正确时间触发。

### 第三步：建立成功轨迹记录

为后续自适应保存：

- 任务和场景标识；
- 规则包版本；
- 完整特征时间序列；
- 人工确认的成功标签；
- 每条规则的裕量分布和未知比例。

### 第四步：加入受控自适应

只调整数值参数，例如阈值、`duration_sec` 和 `minimum_samples`，第一阶段不要让 VLM 自动改写规则结构。每次更新应保留旧版本，并先离线回放比较误报率、漏报率和未知率，再决定是否启用。

## 7. 后续接手时首先阅读

1. `README.md`
2. `docs/standard_features_zh.md`
3. `docs/rule_schema_zh.md`
4. `failure_detection/rule_dsl/rule_schema_v1.json`
5. `failure_detection/rule_dsl/schema.py`
6. `failure_detection/rule_dsl/runtime.py`
7. `failure_detection/stages/planning.py`
8. `failure_detection/pipeline/orchestrator.py`
9. `failure_detection/tests/test_rule_dsl.py`

继续工作时应保持两个边界：规则只能读取标准特征，VLM 只能在任务初始化阶段生成受限规则。

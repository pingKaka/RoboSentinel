# VLM 规则提示词调研与复测

## 1. 参考方法

### Code-as-Monitor

论文：*Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection*，CVPR 2025，arXiv 2412.04455。

原文附录 F.5 的提示词有四个关键特点：

1. 先说明具体工作：为机器人子目标编写 Python 监控函数。
2. 明确函数输入的数据类型和维度，包括末端轨迹、元素历史位置和完成标志。
3. 给出任务分阶段示例，区分执行中约束与完成时约束。
4. 规定可用库、返回值和函数骨架，但把几何计算留给模型编写。

论文还在附录 A.2 说明：生成代码先进行白盒路径测试；发生执行错误就重新生成。VLM只在子目标开始时调用，避免运行中频繁调用。

### 通用结构化生成经验

本次采用的原则是：

- 给模型一个与运行时概念一致的 API，而不是要求重复填写等价信息。
- 用完整名称代替 `x/y/z` 一类缩写，防止模型照抄成不存在的枚举值。
- 同时提供正例和会误触发的负例。
- 让模型内部完成任务分解，但最终只输出可验证结果。
- 将机械转换交给确定性程序，将场景判断和规则选择交给 VLM。

## 2. 原 DSL 接口的问题

旧格式要求同一测量写两次：

```text
feature_bindings[].feature = robot.tcp_z
activation.feature = tcp_z_alias
```

这个 alias 间接层不是规则语义的一部分，也不是 CaM 的做法，却成为主要失败来源。提示词越强调这一点，模型需要同时记忆的格式约束越多。

## 3. 调整后的方法

VLM现在直接为 activation 和 condition 声明：

```json
{
  "feature": "relation.mask_coverage_a",
  "entities": ["bottle", "cup"],
  "camera": "external",
  "op": "lt",
  "value": 0.5,
  "reason": "Less than half of the bottle mask is covered by the cup mask."
}
```

确定性编译器随后生成 alias 和 binding，并交给原有严格 DSL 解析器。运行时规则引擎没有放宽，也没有执行任意模型代码。

此外增加两项保护：

- 接受 `entity_a/entity_b` 和 `entities` 两种等价关系参数写法。
- 拒绝 activation 与 condition 完全相同的规则，避免正常初始状态误触发。

## 4. 测试结果

输入仍是五组视频的双相机第 0 帧，模型不知道文件名和真实故障标签。本测试只检查规则编写，不执行故障判断。

| 场景 | 首轮可编译 | 自动纠错 |
|---|---:|---:|
| grasp_failure | 是 | 未使用 |
| success | 是 | 未使用 |
| target_offset | 是 | 未使用 |
| task_stalled | 是 | 未使用 |
| wrong_object | 是 | 未使用 |

首轮通过率为 `5/5`。原始结果位于：

```text
result/vlm_direct_evidence_v3_retest_2026-11-11/
```

五组结果均选择：

- activation：`robot.tcp_z` 在 1 秒内变化 0.05 米；
- condition：瓶子掩码被杯子掩码覆盖不足 50%；
- failure：放置失败。

该规则结构一致、可执行且解释与测量相符。TCP z 变化只能表示放置阶段候选事件，不能严格证明放置已经完成，这是后续结合夹爪事件或任务阶段状态优化的方向。

## 5. 是否切换为 Python 代码

当前调整后已达到 `5/5` 首轮可编译，因此暂不切换到任意 Python 代码生成。若后续跨任务测试再次出现持续低成功率，可以参考 CaM，改为生成受限函数代码，但应保留 AST 白名单、隔离执行、超时、只读输入和路径测试，不能直接 `exec` 未检查的 VLM 输出。

## 参考链接

- CaM 论文：https://arxiv.org/abs/2412.04455
- CaM PDF：https://arxiv.org/pdf/2412.04455
- Qwen3-VL 模型说明：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

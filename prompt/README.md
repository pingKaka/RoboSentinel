# Prompt 目录

运行时和训练时使用的文字模板统一保存在本目录，避免提示词散落在 Python 代码中。

- `monitoring_plan.txt`：Qwen3-VL 生成跟踪目标和故障规则的主提示词。
- `monitoring_plan_correction.txt`：Schema 校验失败后的完整重生成要求。
- `request_envelope.txt`：把主提示词与运行时 evidence JSON 组合成最终请求。
- `training_sft_zh.txt`：训练数据准备使用的 SFT 模板。

修改在线提示词后，至少执行一次 `scripts/check_backend.py --mode qwen` 和规则 Schema 测试。提示词不能引入标准特征目录之外的字段。

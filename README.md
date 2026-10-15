# RoboSentinel

RoboSentinel用于分析机器人操作过程中的故障，整体流程固定为：

```text
VLM理解初始场景和任务并给出跟踪对象
        ↓
GroundingDINO初始化检测 + SAM实时跟踪图像流
        ↓
可插拔规则判断故障
        ↓
VLM结合规则和当前状态，输出故障分析与恢复建议
```

每个收到的图像观测都会进入跟踪和规则插件；规则确认异常后才调用VLM诊断。机器人状态可选，本地遥操作已接入图像和状态上传。

当前代码提供数据契约、三阶段编排器、Qwen3-VL / GroundingDINO / SAM2.1 真实模型后端和网络服务。默认启用目标持续不可见、机器人控制器故障两类监测规则，阈值在 `cfg.yaml` 配置，也可注册自定义规则。任务级抓取和放置判定需结合阶段信号或标定信息补充。

检查项目配置：

```bash
python failure_detection/run_pipeline.py
```

运行测试：

```bash
python -m unittest discover -s failure_detection/tests -v
```

详细流程见 [项目流程](docs/pipeline.md)。

首次安装请阅读 [从零启动完整教程](docs/start_from_zero_zh.md)，包含两端环境、权重下载、驱动与标定、联网、回放、可视化遥操作及结果查看。已有环境的简短操作见 [运行速查](docs/system_walkthrough_zh.md)。

运行参数统一在根目录 `cfg.yaml`，见 [逐项参数说明](docs/config_reference_zh.md)；增加检测逻辑见 [规则扩展](docs/rules_zh.md)。服务端先保存结果再回传，本地也自动保存独立实验记录。

完整安装、联网和运行步骤见 [服务器部署与运行教程](docs/deployment_tutorial_zh.md)。

下一步测试请直接按 [遥操作与服务器实时检测测试](docs/realtime_test_zh.md)：启动服务器常驻模型，先回放视频验证，再运行本地遥操作自动上传图像与状态。相同观测接口可用于后续 VLA 环境。

首次上传或更新GitHub见 [GitHub上传教程](docs/github_upload_zh.md)。

codex resume 01a13a7b-3c6c-7f62-810a-e89cf4873a0a

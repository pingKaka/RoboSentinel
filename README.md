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

每个新图像帧都会进入跟踪和规则插件；规则确认异常后才调用VLM诊断。机器人状态是可选的同步流，项目只保留接口。

当前代码提供完整的数据契约、三阶段编排器和模型适配器。具体规则通过插件加入，框架不预设故障类型、优先级或阈值。

检查项目配置：

```bash
python failure_detection/run_pipeline.py
```

运行测试：

```bash
python -m unittest discover -s failure_detection/tests -v
```

详细流程见 [项目流程](docs/pipeline.md)。

完整安装、联网和运行步骤见 [服务器部署与运行教程](docs/deployment_tutorial_zh.md)。

首次上传或更新GitHub见 [GitHub上传教程](docs/github_upload_zh.md)。

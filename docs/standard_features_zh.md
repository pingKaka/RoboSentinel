# 标准特征规范 v2.0.0

规则只能读取 `FeatureContext`，不能直接读取原始图像、模型输出或机器人状态字典。

## 数据结构

每个时间点生成一个 `FeatureSnapshot`：

```text
FeatureSnapshot
├── schema_version
├── timestamp_sec / delta_sec
├── quality
│   ├── vision_valid / robot_state_valid
│   ├── cameras[]
│   └── timing_sec
├── entities[]
│   ├── target_id / name / role
│   ├── cameras[]
│   │   ├── visibility
│   │   ├── mask_centroid_xy / mask_area_ratio
│   │   ├── mask_perimeter_normalized
│   │   ├── mask_component_count / mask_compactness
│   │   └── tracking_quality
│   ├── visible_camera_count
│   ├── visible_any
│   └── missing_all_cameras
├── relations[]
│   ├── delta_xy / centroid_distance
│   ├── mask_iou / mask_intersection_ratio
│   └── mask_coverage_a / mask_coverage_b
└── robot
    ├── tcp_position_m / tcp_frame
    └── gripper_position_normalized
```

## 语义约束

- `visibility` 只能是 `visible`、`not_visible` 或 `unknown`。
- `visible` 必须具有与当前图像同尺寸的有效 SAM 二值掩码，否则降级为 `unknown`。
- `mask_centroid_xy` 是前景像素质心，按图像宽高归一化。
- `mask_area_ratio` 是前景像素数除以整幅图像像素数，不是外接矩形面积。
- `mask_perimeter_normalized` 是掩码外轮廓总周长除以图像对角线长度。
- `mask_component_count` 是八邻域前景连通域数量；`mask_compactness` 为 `4*pi*area/perimeter^2`。
- 空间关系只在同一相机内、两个实体均有有效掩码时生成。
- `mask_iou`、交集面积和双方覆盖率直接在二值掩码像素上计算，不使用外接框近似。
- 不同文本目标获得 IoU 大于配置阈值的近重复初始框时，只保留先匹配的目标；未初始化目标记为 `unknown`。
- `tracking_quality` 当前表示最近一次 GroundingDINO 初始化置信度，不代表 SAM 每帧置信度。
- TCP 只保留 xyz 位置，单位为米，并显式记录坐标系。
- 夹爪只保留 `[0, 1]` 的归一化位置；当前设备未统一定义两个端点的开闭方向，因此规范不擅自生成 `is_open`。
- 不可用值使用 `null`，不能使用零代替。
- `quality.robot_state_valid=false` 表示 TCP 和夹爪特征均不可用；纯视频测试应为该状态。

## 扩展规则

- 新特征必须增加新的 catalog 路径、类型、单位、可空性和测试。
- v1 字段的含义和单位不可原地改变；不兼容变更必须提升主版本号。
- GroundingDINO 检测框只用于初始化 SAM，不属于标准特征。
- SAM dense mask 只作为当前帧特征计算输入，不进入规则历史；标准层保存上述 mask 派生量和精确 mask 关系。
- 原始图像、策略动作、VLA 内部状态和故障结论不属于标准特征。
- 三维视觉关系、任务阶段和额外机器人状态应通过后续版本或独立 provider 增加。

完整机器可读目录由 `failure_detection.features.feature_catalog()` 返回。视频收集验证命令：

```bash
python scripts/collect_video_features.py --video-dir video --frame-step 1 --max-frames 0
python scripts/render_video_feature_report.py
```

第二条命令读取已收集的 JSONL 和对应视频帧，在
`result/feature_collection/visual_report/index.html` 生成可视化审验报告；它不会重新运行模型。

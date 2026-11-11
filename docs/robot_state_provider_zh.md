# 机器人状态接入：以 UR5e 为例

本文说明 RoboSentinel 如何读取 UR5e 状态，以及以后如何接入其他机器人。当前功能只读取状态用于故障检测，**不会向机器人发送控制命令**。

## 1. 先运行

默认配置已经启用 UR5e 状态读取：

```yaml
video_teleop:
  sentinel:
    robot_state:
      enabled: true
      provider: ur5e_rtde
```

按原来的方式启动遥操作即可。程序会复用 UR5e 驱动已经建立的 RTDE Receive 连接，不会再连接一次机器人。

若只想使用视频检测，可关闭机器人状态：

```yaml
robot_state:
  enabled: false
  provider: none
```

## 2. 当前读取哪些特征

当前标准机器人特征保持最小集合：

| 标准字段 | 含义 | 单位 | UR5e 数据来源 |
|---|---|---|---|
| `tcp_position` | 工具中心点的 x、y、z 位置 | m | RTDE `actual_TCP_pose` 的前三项 |
| `tcp_position_frame` | TCP 位置使用的坐标系 | - | RTDE 结果固定为 `base` |
| `gripper_position_normalized` | 夹爪开合位置 | `[0,1]` | 现有驱动观测 `gripper_raw_position` |

UR 官方将 `actual_TCP_pose` 定义为实际 TCP 笛卡尔位姿 `[X,Y,Z,Rx,Ry,Rz]`，位置单位为米，旋转使用轴角向量。当前项目只保留前三项，因为规则层目前只需要 TCP 位置。

输出示例：

```json
{
  "source": "ur5e_rtde",
  "timestamp_sec": 12.34,
  "valid": true,
  "tcp_position": [0.42, -0.18, 0.31],
  "tcp_position_frame": "base",
  "gripper_position_normalized": 0.55,
  "diagnostics": {
    "rtde_receiver_available": true
  }
}
```

## 3. 程序如何工作

```text
UR5e 驱动已有 RTDE Receiver
        ↓
UR5eRTDEStateProvider（只处理 UR5e 细节）
        ↓
RobotStateSample（统一格式）
        ↓
ObservationPacket.robot_state（通用传输）
        ↓
StandardFeatureExtractor（标准特征）
        ↓
规则 DSL
```

关键原则：

1. 主循环不再访问 `robot._arm["rtde_r"]`。
2. UR5e 私有驱动结构只在 `failure_detection/robot_state/ur5e.py` 中出现。
3. 网络传输、特征提取和规则执行不知道机器人品牌。
4. RTDE 读取失败时返回无效或降级数据，不中断遥操作。
5. 缺失的机器人特征在规则中应得到 `unknown`，不能直接判为故障。

## 4. 读取失败时会怎样

正常顺序如下：

1. 优先调用已有 RTDE Receiver 的 `getActualTCPPose()`。
2. 若 RTDE 暂时读取失败，尝试使用驱动观测中的 `tcp_pose.x/y/z`。
3. 两种来源都不可用时，TCP 字段为 `null`。
4. 错误文字写入 `diagnostics`，方便排查，但检测流水线继续工作。

夹爪值不是数字或不在 `[0,1]` 时也会变成 `null`，避免错误值进入规则。

## 5. 文件职责

| 文件 | 职责 |
|---|---|
| `robot_state/contracts.py` | 定义所有机器人共同遵守的接口和采样结果 |
| `robot_state/ur5e.py` | 读取 UR5e RTDE 和夹爪观测，转成统一字段 |
| `robot_state/null.py` | 无机器人或纯视频模式 |
| `robot_state/factory.py` | 根据配置选择 Provider |
| `transport/teleop_adapter.py` | 组装图像、状态和时间信息，不包含机器人细节 |

## 6. 如何接入另一种机器人

例如接入 Franka，只需要：

1. 新建 `failure_detection/robot_state/franka.py`。
2. 实现 `read(observation, timestamp_sec)`。
3. 将 Franka 的原始状态转换为同样的 `tcp_position`、`tcp_position_frame` 和 `gripper_position_normalized`。
4. 在 `factory.py` 中加入 `franka` 配置分支。
5. 添加使用模拟对象的单元测试。

伪代码：

```python
class FrankaStateProvider:
    name = "franka"

    def read(self, observation, timestamp_sec):
        values = {
            "tcp_position": read_franka_tcp_xyz(),
            "tcp_position_frame": "base",
            "gripper_position_normalized": read_franka_gripper(),
        }
        return RobotStateSample(timestamp_sec, self.name, True, values)
```

主循环、服务器、特征提取器和规则代码都不需要修改。若未来要增加一种新的**标准特征**，则必须同步更新特征 schema、规则 DSL、提示词、测试和文档，不能只在某个机器人 Provider 中偷偷增加规则依赖字段。

## 7. 后续可扩展状态

UR RTDE 还提供 TCP 速度、关节位置与速度、TCP 力、机器人模式、安全状态和速度缩放等字段。当前没有把它们加入标准特征，原因是需要先明确故障规则如何使用、单位和缺失值语义。它们可在后续以兼容方式加入，而不改变本次 Provider 架构。

## 8. 官方资料

- Universal Robots, *Real-Time Data Exchange (RTDE) Guide*。
- Universal Robots, *RTDE Output Recipes*，其中定义了 `actual_TCP_pose` 等输出字段。

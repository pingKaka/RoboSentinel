from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from failure_detection.robot_state import (
    NullRobotStateProvider, UR5eRTDEStateProvider, create_robot_state_provider,
)


class RobotStateProviderTests(unittest.TestCase):
    def test_ur5e_maps_rtde_pose_and_gripper(self):
        receiver = Mock()
        receiver.getActualTCPPose.return_value = [0.1, 0.2, 0.3, 1, 2, 3]
        sample = UR5eRTDEStateProvider(receiver).read({"gripper_raw_position": 0.4}, 2.5)
        self.assertEqual(sample.values["tcp_position"], [0.1, 0.2, 0.3])
        self.assertEqual(sample.values["tcp_position_frame"], "base")
        self.assertEqual(sample.values["gripper_position_normalized"], 0.4)
        self.assertTrue(sample.valid)
        receiver.getActualTCPPose.assert_called_once_with()

    def test_read_failure_uses_observation_without_raising(self):
        receiver = Mock()
        receiver.getActualTCPPose.side_effect = RuntimeError("offline")
        observation = {"tcp_pose.x": 1, "tcp_pose.y": 2, "tcp_pose.z": 3}
        sample = UR5eRTDEStateProvider(receiver, "episode_reference").read(observation, 1)
        self.assertEqual(sample.values["tcp_position"], [1, 2, 3])
        self.assertEqual(sample.values["tcp_position_frame"], "episode_reference")
        self.assertIn("tcp_read_error", sample.diagnostics)

    def test_factory_isolates_driver_private_access(self):
        receiver = Mock()
        config = SimpleNamespace(control_space="tcp_force", tcp_force_reference_frame="tcp")
        robot = SimpleNamespace(config=config, _arm={"rtde_r": receiver})
        provider = create_robot_state_provider({"provider": "ur5e_rtde"}, robot)
        self.assertIsInstance(provider, UR5eRTDEStateProvider)
        self.assertEqual(provider._fallback_frame, "episode_reference")

    def test_null_and_unknown_provider(self):
        self.assertIsInstance(create_robot_state_provider({"enabled": False}), NullRobotStateProvider)
        self.assertFalse(NullRobotStateProvider().read({}, 0).valid)
        with self.assertRaisesRegex(ValueError, "Unknown robot-state provider"):
            create_robot_state_provider({"provider": "future_robot"}, object())


if __name__ == "__main__":
    unittest.main()

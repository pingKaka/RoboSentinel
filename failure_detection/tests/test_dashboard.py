import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from failure_detection.transport.dashboard import DetectionDashboard


class DashboardTests(unittest.TestCase):
    def test_render_fault_contours_and_long_chinese_analysis(self):
        dashboard = DetectionDashboard()
        image = np.zeros((240, 424, 3), np.uint8)
        frames = {"external": image, "wrist": image}
        result = {"timestamp_sec": 1, "perception": {"tracks": [
            {"camera": "external", "target_id": "bottle", "visible": True,
             "box_xyxy": [20, 20, 100, 150], "contours": [[[20, 20], [100, 20], [100, 150]]]}]},
            "diagnosis": {"failure_analysis": "目标不可见，需要检查是否存在遮挡。" * 80,
                          "recovery_suggestions": ["检查当前场景。"]}}
        client = SimpleNamespace(status="streaming", error=None, _manual_busy=False,
                                 latest_visual=(result, frames), fault_result=result, latest_result=result,
                                 initial_result={"task_id": "bottle_test", "description": "抓取瓶子并放入试管架",
                                                 "tracking_plan": {"targets": [{"target_id": "bottle", "name": "white bottle",
                                                                                "role": "manipulated_object"}]}},
                                 latest_robot_state={"controller_state": "normal"})
        rendered = dashboard.render(frames, client, True, 2)
        self.assertEqual(rendered.shape, (940, 1600, 3))
        self.assertIn("跟踪目标", "\n".join(dashboard.info_lines))
        self.assertIn("bottle_test", "\n".join(dashboard.info_lines))
        self.assertIn("white bottle", "\n".join(dashboard.info_lines))
        self.assertIn("外部: 可见 / 腕部: 未知", dashboard.info_lines)
        self.assertIn("机器人控制器: normal", dashboard.info_lines)
        self.assertGreater(dashboard.line_count, 7)
        self.assertGreater(np.unique(rendered.reshape(-1, 3), axis=0).shape[0], 100)
        dashboard.scroll = 9999
        dashboard.render(frames, client, True, 2)
        self.assertEqual(dashboard.scroll, dashboard.line_count - 7)

    def test_no_client_or_upload_before_space(self):
        path = Path("ur5e_video_teleop/ur5e_video_teleop/run_video_teleop.py")
        spec = importlib.util.spec_from_file_location("teleop_gate_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: module}):
            spec.loader.exec_module(module)
            cfg = module.VideoTeleopConfig.from_yaml(Path("cfg.yaml"))
            cfg.preview = False
            keys = module.ToggleKeyState("space", "esc")
            keys.start = Mock()
            keys.stop = Mock()
            robot, teleop = Mock(), Mock()
            client = Mock(interval=0.01)
            client._thread.is_alive.return_value = False
            turn = 0
            def observation():
                nonlocal turn
                turn += 1
                if turn == 1:
                    factory.assert_not_called()
                if turn == 2:
                    keys.recording = True
                if turn == 3:
                    keys.recording = False
                if turn == 4:
                    keys.stop_requested = True
                return {"exterior_image": np.zeros((8, 8, 3), np.uint8), "wrist_image": np.zeros((8, 8, 3), np.uint8)}
            robot.get_observation.side_effect = observation
            with patch.object(module, "make_robot_and_teleop", return_value=(robot, teleop)), \
                 patch.object(module, "ToggleKeyState", return_value=keys), \
                 patch.object(module, "VideoSegmentRecorder"), \
                 patch("failure_detection.transport.teleop_adapter.teleop_packet", return_value=Mock()), \
                 patch("failure_detection.transport.live_client.LiveDetectionClient", return_value=Mock(start=Mock(return_value=client))) as factory, \
                 patch.object(module, "replace", side_effect=lambda packet, **kwargs: packet):
                module.run_video_teleop(cfg)
            factory.assert_called_once()
            client.submit.assert_called_once()
            client.close.assert_any_call(wait=False)

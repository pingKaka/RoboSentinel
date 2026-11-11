import unittest
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np

from failure_detection.transport.dashboard_process import DashboardProcess, _window_main


class DashboardProcessTests(unittest.TestCase):
    def test_snapshot_does_not_enqueue_client_or_mutable_camera_buffer(self):
        preview = DashboardProcess.__new__(DashboardProcess)
        preview._updates = Queue(maxsize=1)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        client = SimpleNamespace(status="streaming", socket=object())
        preview.render({"external": frame, "wrist": frame}, client, now=1)
        frame[:] = 255
        packet = preview._updates.get_nowait()
        self.assertEqual(packet["client"].status, "streaming")
        self.assertFalse(hasattr(packet["client"], "socket"))
        self.assertFalse(packet["frames"]["external"].any())

    def test_full_queue_does_not_block(self):
        preview = DashboardProcess.__new__(DashboardProcess)
        preview._updates = Queue(maxsize=1)
        preview._updates.put("pending")
        preview.render({})
        self.assertEqual(preview._updates.get_nowait(), "pending")

    def test_window_pumps_events_without_hardware_updates(self):
        stop, quit_requested = Event(), Event()
        with patch("failure_detection.transport.dashboard.DetectionDashboard") as dashboard, \
             patch.dict("sys.modules", {"cv2": MagicMock()}) as modules:
            cv = modules["cv2"]
            cv.WINDOW_NORMAL = 0
            cv.WINDOW_KEEPRATIO = 0
            cv.getWindowProperty.return_value = 1
            cv.waitKey.side_effect = [-1, -1, ord("q")]
            _window_main(Queue(), stop, quit_requested, None)
            self.assertEqual(cv.waitKey.call_count, 3)
            self.assertEqual(dashboard.return_value.render.call_count, 1)
            self.assertTrue(quit_requested.is_set())
            cv.destroyAllWindows.assert_called_once()


if __name__ == "__main__":
    unittest.main()

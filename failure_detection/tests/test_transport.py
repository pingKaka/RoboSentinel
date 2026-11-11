import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from failure_detection.transport.codec import encode_jpeg
from failure_detection.transport.server import create_app


class TransportTests(unittest.TestCase):
    def test_task_limit_stream_result_and_disconnect_cleanup(self):
        pipeline = Mock()
        pipeline.update.return_value.to_dict.return_value = {
            "timestamp_sec": 1, "perception": {"tracks": []},
            "rules": {"failure_detected": False, "findings": []},
        }
        jpeg = encode_jpeg(np.zeros((8, 8, 3), dtype=np.uint8))
        request = {"task_id": "test", "description": "move bottle", "timestamp_sec": 0,
                   "frames": {"external": jpeg, "wrist": jpeg}}
        with patch("failure_detection.transport.server.load_pipeline_factory", return_value=lambda _: pipeline):
            with TestClient(create_app({"deployment": {"pipeline_factory": "test:factory"}})) as client:
                response = client.post("/tasks", json=request)
                self.assertEqual(response.status_code, 200)
                session = response.json()["task_session_id"]
                self.assertEqual(client.post("/tasks", json=request).status_code, 409)
                with client.websocket_connect(f"/tasks/{session}/stream") as ws:
                    ws.send_json({"timestamp_sec": 1, "frames": request["frames"]})
                    result = ws.receive_json()
                    self.assertNotIn("diagnosis", result)
                    self.assertFalse(result["rules"]["failure_detected"])
                    self.assertIn("server_processing_ms", result)
                    self.assertEqual(client.delete(f"/tasks/{session}").status_code, 409)
                self.assertEqual(client.get("/health").json()["active_tasks"], 0)
                self.assertEqual(client.post("/tasks", json=request).status_code, 200)

    def test_invalid_frames_do_not_load_models(self):
        with patch("failure_detection.transport.server.load_pipeline_factory") as factory:
            with TestClient(create_app({})) as client:
                response = client.post("/tasks", json={"task_id": "test", "description": "test", "timestamp_sec": 0, "frames": {}})
                self.assertEqual(response.status_code, 503)
                factory.assert_not_called()

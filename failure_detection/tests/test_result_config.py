import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from project_config import DEFAULT_CONFIG, client_settings, load_project_config
from failure_detection.pipeline.contracts import TrackingPlan
from failure_detection.rules import create_rules
from failure_detection.rule_plugins.tcp_speed import TCPSpeedRule
from failure_detection.transport.codec import encode_jpeg
from failure_detection.transport.server import create_app


class ResultConfigTests(unittest.TestCase):
    def test_config_paths_and_unique_runs(self):
        config, root = load_project_config(DEFAULT_CONFIG)
        first = client_settings(config, root, "replay")
        second = client_settings(config, root, "replay")
        self.assertNotEqual(first["output_dir"], second["output_dir"])
        self.assertTrue(Path(first["output_dir"]).is_absolute())
        self.assertEqual(first["description"], config["task"]["description"])

    def test_persist_before_response_and_disk_failure(self):
        pipeline = Mock()
        pipeline.plan = TrackingPlan("complete", (), "test")
        pipeline.update.return_value.to_dict.return_value = {"diagnosis": None}
        jpeg = encode_jpeg(np.zeros((8, 8, 3), dtype=np.uint8))
        request = {"task_id": "test", "description": "test", "timestamp_sec": 0,
                   "frames": {"external": jpeg, "wrist": jpeg}}
        with tempfile.TemporaryDirectory() as directory:
            config = {"outputs": {"server_dir": directory, "save_annotated_frames": True}}
            with patch("failure_detection.transport.server.load_pipeline_factory", return_value=lambda _: pipeline):
                with TestClient(create_app(config)) as client:
                    with patch("failure_detection.transport.server.ResultStore.write", side_effect=OSError("disk full")):
                        self.assertEqual(client.post("/tasks", json=request).status_code, 503)
                    self.assertEqual(client.get("/health").json()["active_tasks"], 0)
                    initial = client.post("/tasks", json=request).json()["initial_result"]
                    path = Path(initial["server_result_file"])
                    self.assertEqual(json.loads(path.read_text()), initial)
                    self.assertEqual(initial["tracking_plan"]["planner"], "test")
                    self.assertTrue(all(Path(p).exists() for p in initial["annotated_frames"].values()))
                    with client.websocket_connect(f'/tasks/{initial["task_session_id"]}/stream') as ws:
                        ws.send_json(request)
                        result = ws.receive_json()
                        self.assertEqual(json.loads(path.read_text().splitlines()[-1]), result)
                        with patch("failure_detection.transport.server.ResultStore.write", side_effect=OSError("disk full")):
                            ws.send_json(request)
                            self.assertEqual(ws.receive_json(), {"error": "disk full"})
                    self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_speed_plugin_state_and_registration(self):
        rule = create_rules({"plugins": [{"name": "failure_detection.rule_plugins.tcp_speed:TCPSpeedRule",
                                          "options": {"seconds": 0.5}}]})[0]
        def update(t, speed):
            return rule.update(SimpleNamespace(observation=SimpleNamespace(
                timestamp_sec=t, robot_state={"tcp_speed": speed}))).status
        self.assertEqual(update(0, [0.4, 0, 0]), "unknown")
        self.assertEqual(update(0.5, [0.4, 0, 0]), "confirmed")
        self.assertEqual(update(0.6, None), "unknown")
        self.assertEqual(update(0.7, [0, 0, 0]), "clear")
        self.assertEqual(update(1, [0.4, 0, 0]), "unknown")
        self.assertEqual(update(4, [0.4, 0, 0]), "unknown")
        with self.assertRaises(ValueError):
            create_rules({"plugins": ["controller_fault", "controller_fault"]})
        with self.assertRaises(ValueError):
            TCPSpeedRule(limit_mps=float("nan"))

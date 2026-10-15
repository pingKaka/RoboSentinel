import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from failure_detection.pipeline.contracts import PerceptionUpdate, TrackingPlan
from failure_detection.pipeline.orchestrator import StreamingRoboSentinel
from failure_detection.stages.diagnosis import EventTriggeredVLMDiagnoser
from failure_detection.transport.codec import encode_jpeg
from failure_detection.transport.live_client import LiveDetectionClient, ObservationPacket
from failure_detection.transport.server import create_app
from failure_detection.pipeline.contracts import RuleFinding, RuleReport


class ManualFailureTests(unittest.TestCase):
    def test_manual_websocket_bypasses_rules_and_repeats(self):
        planner, tracker, rules = Mock(), Mock(), Mock()
        planner.plan.return_value = TrackingPlan("complete", (), "test")
        tracker.update.side_effect = lambda o: PerceptionUpdate(o.timestamp_sec, "complete", (), "test")
        infer = Mock(return_value={"failure_analysis": "manual", "recovery_suggestions": []})
        pipeline = StreamingRoboSentinel(planner, tracker, rules, EventTriggeredVLMDiagnoser(infer))
        jpeg = encode_jpeg(np.zeros((8, 8, 3), dtype=np.uint8))
        request = {"task_id": "test", "description": "move", "timestamp_sec": 0,
                   "frames": {"external": jpeg, "wrist": jpeg}}
        with patch("failure_detection.transport.server.load_pipeline_factory", return_value=lambda _: pipeline):
            with TestClient(create_app({})) as client:
                session = client.post("/tasks", json=request).json()["task_session_id"]
                with client.websocket_connect(f"/tasks/{session}/stream") as ws:
                    for stamp in (1, 2):
                        ws.send_json({**request, "timestamp_sec": stamp, "manual_failure": True})
                        result = ws.receive_json()
                        self.assertTrue(result["manual_failure"])
                        self.assertEqual(result["diagnosis"]["failure_analysis"], "manual")
                        self.assertEqual(result["rules"]["findings"][0]["evidence"]["source"], "operator")
        rules.update.assert_not_called()
        self.assertEqual(infer.call_count, 2)
        self.assertEqual(pipeline._diagnosed_events, set())

    def test_recovery_rearms_automatic_diagnosis(self):
        from failure_detection.pipeline.contracts import SceneSnapshot, StreamObservation, Task
        planner, tracker, rules, diagnoser = Mock(), Mock(), Mock(), Mock()
        planner.plan.return_value = TrackingPlan("complete", (), "test")
        tracker.update.return_value = PerceptionUpdate(0, "complete", (), "test")
        rules.update.return_value = RuleReport(0, "complete", (RuleFinding("fault", "confirmed"),), True)
        pipeline = StreamingRoboSentinel(planner, tracker, rules, diagnoser)
        pipeline.initialize(Task("test", "test"), SceneSnapshot(0, {}))
        pipeline.update(StreamObservation(1, {}))
        pipeline.update(StreamObservation(2, {}))
        self.assertEqual(diagnoser.diagnose.call_count, 1)
        reset = pipeline.update(StreamObservation(3, {}, reset_diagnosis=True))
        self.assertIsNone(reset.diagnosis)
        self.assertEqual(reset.rules.status, "reset")
        pipeline.update(StreamObservation(4, {}))
        self.assertEqual(diagnoser.diagnose.call_count, 2)

    def test_manual_snapshot_survives_latest_frames(self):
        client = LiveDetectionClient("http://localhost", "test")
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        def packet(t, manual=False):
            return ObservationPacket(t, {"external": image, "wrist": image}, manual_failure=manual)
        self.assertFalse(client.submit(packet(1, True)))
        client.status = "streaming"
        self.assertTrue(client.submit(packet(1)))
        self.assertTrue(client.submit(packet(2, True)))
        image[:] = 255
        self.assertTrue(client.submit(packet(3)))
        self.assertFalse(client.submit(packet(4, True)))
        manual = asyncio.run(client._next())
        self.assertEqual(manual.timestamp_sec, 2)
        self.assertEqual(int(manual.frames["external"].max()), 0)
        self.assertTrue(client._encode(manual)["manual_failure"])
        self.assertEqual(asyncio.run(client._next()).timestamp_sec, 3)

    def test_keyboard_hold_only_triggers_once(self):
        path = Path("ur5e_video_teleop/ur5e_video_teleop/run_video_teleop.py")
        spec = importlib.util.spec_from_file_location("teleop_manual_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: module}):
            spec.loader.exec_module(module)
            with patch.object(module, "normalize_key", side_effect=lambda key: key):
                keys = module.ToggleKeyState("space", "esc", "f")
                keys._on_press("f")
                self.assertTrue(keys.failure_requested.is_set())
                keys.failure_requested.clear()
                keys._on_press("f")
                self.assertFalse(keys.failure_requested.is_set())
                keys._on_release("f")
                keys._on_press("f")
                self.assertTrue(keys.failure_requested.is_set())
                self.assertFalse(keys.recording)
                self.assertFalse(keys.stop_requested)

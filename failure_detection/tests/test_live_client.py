import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import httpx
import uvicorn
from fastapi.testclient import TestClient

from failure_detection.transport.live_client import LiveDetectionClient, ObservationPacket
from failure_detection.transport.teleop_adapter import teleop_packet
from failure_detection.transport.server import create_app
from failure_detection.backends import load_qwen_once, _qwen


def wait_until(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Timed out")
        time.sleep(0.01)


class LiveClientTests(unittest.TestCase):
    def test_model_cache_reuses_instance(self):
        _qwen.cache_clear()
        try:
            with patch("failure_detection.backends.QwenBackend") as constructor:
                self.assertIs(load_qwen_once("test"), load_qwen_once("test"))
            constructor.assert_called_once_with("test", True, 768, 768)
        finally:
            _qwen.cache_clear()

    def test_preload_happens_once_per_server_lifetime(self):
        preload = Mock(return_value={"resident_models": ["vlm", "grounding_dino", "sam"]})
        with patch("failure_detection.transport.server.load_pipeline_factory", return_value=preload):
            with TestClient(create_app({"deployment": {"preload_factory": "test:preload"}})) as client:
                for _ in range(2):
                    health = client.get("/health").json()
                    self.assertTrue(health["models_ready"])
                    self.assertEqual(len(health["resident_models"]), 3)
            preload.assert_called_once()

    def test_adapter_preserves_state_semantics(self):
        receiver = Mock()
        receiver.getActualTCPPose.return_value = [1, 2, 3, 4, 5, 6]
        receiver.getSafetyMode.return_value = 3
        robot = SimpleNamespace(config=SimpleNamespace(control_space="tcp_force"), _arm={"rtde_r": receiver})
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        obs = {"wrist_image": image, "exterior_image": image, "gripper_raw_position": 0.5,
               **{f"tcp_pose.{axis}": 0.1 for axis in ("x", "y", "z", "rx", "ry", "rz")}}
        with patch("failure_detection.transport.teleop_adapter.time.monotonic", return_value=12.1):
            packet = teleop_packet(obs, robot, 10, 11.9, 12)
        self.assertEqual(packet.timestamp_sec, 2)
        self.assertEqual(packet.robot_state["controller_state"], "protective_stop")
        self.assertEqual(packet.robot_state["tcp_pose"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(packet.robot_state["teleop_ee_pose"], [0.1] * 6)
        self.assertNotIn("gripper_width", packet.robot_state)

    def test_latest_slot_owns_images_and_validates_time(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        client = LiveDetectionClient("http://localhost", "test")
        for stamp in range(10):
            client.submit(ObservationPacket(stamp, {"external": image, "wrist": image}))
        image[:] = 255
        self.assertEqual(client.dropped, 9)
        self.assertEqual(client._latest.timestamp_sec, 9)
        self.assertFalse(client._latest.frames["external"].any())
        with self.assertRaises(ValueError):
            client.submit(ObservationPacket(9, {"external": image, "wrist": image}))

    def test_real_http_websocket_record_and_replay(self):
        pipeline = Mock()

        def update(observation):
            time.sleep(0.03)
            fault = (observation.robot_state or {}).get("controller_state") == "fault"
            return SimpleNamespace(to_dict=lambda: {
                "timestamp_sec": observation.timestamp_sec,
                "perception": {"tracks": []},
                "rules": {"failure_detected": fault, "findings": [{"rule_id": "controller_fault",
                           "status": "confirmed" if fault else "clear"}]},
                "diagnosis": {"failure_analysis": "test"} if fault else None})

        pipeline.update.side_effect = update
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(create_app({"deployment": {"pipeline_factory": "test:factory"}}), log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory, patch(
                "failure_detection.transport.server.load_pipeline_factory", return_value=lambda _: pipeline):
            thread.start()
            client = LiveDetectionClient(f"http://127.0.0.1:{port}", "test", fps=20,
                                         output_dir=directory, record_observations=True).start()
            try:
                wait_until(lambda: server.started)
                client.submit(ObservationPacket(0, {"external": image, "wrist": image}))
                wait_until(lambda: client.status == "streaming")
                for i in range(1, 101):
                    client.submit(ObservationPacket(i / 100, {"external": image, "wrist": image}, {"controller_state": "normal"}))
                wait_until(lambda: client.latest_result is not None and client.latest_result["timestamp_sec"] == 1)
                client.close()
                self.assertIsNone(client.error)
                self.assertFalse(client._thread.is_alive())
                records = [json.loads(line) for line in (Path(directory) / "observations.jsonl").read_text().splitlines()]
                self.assertEqual(records[0]["kind"], "initial")
                self.assertEqual(records[-1]["robot_state"]["controller_state"], "normal")
                self.assertGreater(client.dropped, 0)
                process = subprocess.run([sys.executable, "scripts/replay_detection.py", "--server", f"http://127.0.0.1:{port}",
                                          "--observations", str(Path(directory) / "observations.jsonl"),
                                          "--output", str(Path(directory) / "replay")], capture_output=True, text=True, timeout=20)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn("PASS", process.stdout)
                fault_replay = subprocess.run([sys.executable, "scripts/replay_detection.py", "--server", f"http://127.0.0.1:{port}",
                                               "--observations", str(Path(directory) / "observations.jsonl"),
                                               "--simulate-fault-at", "0.5", "--output", str(Path(directory) / "fault")],
                                              capture_output=True, text=True, timeout=20)
                self.assertEqual(fault_replay.returncode, 0, fault_replay.stderr)
                self.assertIn("DIAGNOSIS", fault_replay.stderr)
                import yaml
                from project_config import DEFAULT_CONFIG, load_project_config
                config, _ = load_project_config(DEFAULT_CONFIG)
                config["client"].update(server=f"http://127.0.0.1:{port}", output_dir="configured_results")
                config["replay"].update(observations="observations.jsonl", simulate_fault_at=0.5)
                config_path = Path(directory) / "test.yaml"
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                configured = subprocess.run([sys.executable, "scripts/replay_detection.py", "--config", str(config_path)],
                                            capture_output=True, text=True, timeout=20)
                self.assertEqual(configured.returncode, 0, configured.stderr)
                self.assertIn("DIAGNOSIS", configured.stderr)
                self.assertEqual(len(list((Path(directory) / "configured_results").glob("replay/*/results.jsonl"))), 1)
                entered, completed = threading.Event(), threading.Event()

                def slow_initialize(*args):
                    entered.set()
                    time.sleep(0.15)
                    completed.set()

                pipeline.initialize.side_effect = slow_initialize
                cancelled = LiveDetectionClient(f"http://127.0.0.1:{port}", "test", output_dir=Path(directory) / "cancel").start()
                try:
                    cancelled.submit(ObservationPacket(0, {"external": image, "wrist": image}))
                    wait_until(entered.is_set)
                    cancelled.close()
                    wait_until(completed.is_set)
                    with httpx.Client(trust_env=False) as http:
                        wait_until(lambda: http.get(f"http://127.0.0.1:{port}/health").json()["active_tasks"] == 0)
                finally:
                    cancelled.close()
            finally:
                client.close()
                server.should_exit = True
                thread.join(timeout=10)
                sock.close()

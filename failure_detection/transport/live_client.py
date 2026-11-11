"""Latest-observation transport shared by live and replay detection clients."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from threading import Event, Lock, Thread
import time
from urllib.parse import urlsplit, urlunsplit

import httpx
import numpy as np
import websockets

from .codec import encode_jpeg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservationPacket:
    timestamp_sec: float
    frames: dict[str, np.ndarray]
    robot_state: dict | None = None
    synchronization: dict | None = None
    manual_failure: bool = False


def create_detection_client(settings):
    return LiveDetectionClient(
        settings["server"], settings["description"],
        task_id=settings.get("task_id", "live"), fps=float(settings.get("fps", 10)),
        output_dir=Path(settings.get("output_dir", "result/client")),
        record_observations=settings.get("record_observations", True),
        timeout=float(settings.get("timeout_sec", 600)),
        jpeg_quality=int(settings.get("jpeg_quality", 85)),
        ws_max_size=int(settings.get("ws_max_size", 2**24)),
        ping_timeout=float(settings.get("ping_timeout_sec", 120)),
    )


class LiveDetectionClient:
    """submit() copies RGB images only; encoding, files and network run off-thread.

    A producer must use increasing seconds from one monotonic origin. This client
    never returns robot commands, opens cameras or calls a robot control API.
    """

    def __init__(self, server, description, task_id="live", fps=5, output_dir="result/live",
                 record_observations=False, timeout=600, jpeg_quality=85,
                 ws_max_size=2**24, ping_timeout=120):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.server = server.rstrip("/")
        self.description, self.task_id = description, task_id
        self.interval, self.timeout = 1 / fps, timeout
        self.jpeg_quality = int(jpeg_quality)
        self.ws_max_size, self.ping_timeout = int(ws_max_size), float(ping_timeout)
        if timeout <= 0 or not 1 <= self.jpeg_quality <= 100 or min(self.ws_max_size, self.ping_timeout) <= 0:
            raise ValueError("Invalid transport quality, size or timeout")
        self.output_dir = Path(output_dir)
        self.record_observations = record_observations
        self._lock, self._stop = Lock(), Event()
        self._latest = None
        self._manual = None
        self._manual_busy = False
        self._thread = None
        self._loop = None
        self._task = None
        self._last_submitted = -float("inf")
        self.status = "idle"
        self.error = None
        self.latest_result = None
        self.latest_visual = None
        self.initial_result = None
        self.latest_robot_state = None
        self.fault_result = None
        self.dropped = 0
        self.responses = 0

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Client has already started")
        self._thread = Thread(target=self._run, name="robosentinel-client", daemon=True)
        self._thread.start()
        return self

    def submit(self, packet: ObservationPacket):
        if self._stop.is_set():
            return False
        if packet.manual_failure and (self.status != "streaming" or self._manual_busy):
            return False
        if set(packet.frames) != {"external", "wrist"}:
            raise ValueError("Both external and wrist RGB frames are required")
        if not np.isfinite(packet.timestamp_sec) or packet.timestamp_sec <= self._last_submitted:
            raise ValueError("Observation timestamps must be finite and strictly increasing")
        for image in packet.frames.values():
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("Frames must be uint8 RGB HWC arrays")
        copy = ObservationPacket(packet.timestamp_sec, {k: v.copy() for k, v in packet.frames.items()},
                                 json.loads(json.dumps(packet.robot_state, allow_nan=False)),
                                 dict(packet.synchronization or {}), packet.manual_failure)
        with self._lock:
            self._last_submitted = packet.timestamp_sec
            self.dropped += self._latest is not None
            if packet.manual_failure:
                self._manual, self._latest, self._manual_busy = copy, None, True
            else:
                self._latest = copy
        return True

    def close(self, wait=True):
        self._stop.set()
        if self._loop is not None and self._task is not None:
            try:
                self._loop.call_soon_threadsafe(self._task.cancel)
            except RuntimeError:
                pass
        if wait and self._thread is not None:
            self._thread.join(timeout=12)

    def _run(self):
        try:
            asyncio.run(self._main())
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.error, self.status = str(exc), "error"
            logger.error("[Sentinel] %s (teleoperation continues)", exc)
        finally:
            self._stop.set()
            if self.status != "error":
                self.status = "closed"

    async def _next(self):
        while not self._stop.is_set():
            with self._lock:
                if self._manual is not None:
                    packet, self._manual = self._manual, None
                else:
                    packet, self._latest = self._latest, None
            if packet is not None:
                return packet
            await asyncio.sleep(0.01)
        raise asyncio.CancelledError

    def _encode(self, packet):
        return {"timestamp_sec": packet.timestamp_sec,
                "frames": {k: encode_jpeg(np.ascontiguousarray(v[:, :, ::-1]), self.jpeg_quality) for k, v in packet.frames.items()},
                "robot_state": packet.robot_state, "synchronization": packet.synchronization or {},
                "manual_failure": packet.manual_failure}

    async def _main(self):
        self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        session_id = None
        self.status = "waiting_for_observation"
        with (self.output_dir / "results.jsonl").open("a", encoding="utf-8") as results:
            observations = ((self.output_dir / "observations.jsonl").open("a", encoding="utf-8")
                            if self.record_observations else None)
            try:
                async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as http:
                    source = await self._next()
                    packet = self._encode(source)
                    self.status = "initializing"
                    response = await http.post(self.server + "/tasks", json={**packet, "task_id": self.task_id,
                                                                            "description": self.description})
                    response.raise_for_status()
                    session_id = response.json()["task_session_id"]
                    initial = response.json().get("initial_result")
                    self.initial_result = initial
                    if initial is not None:
                        self.latest_visual = (initial, source.frames)
                        results.write(json.dumps(initial, ensure_ascii=False) + "\n")
                        results.flush()
                    if observations:
                        observations.write(json.dumps({**packet, "kind": "initial", "task_session_id": session_id}) + "\n")
                        observations.flush()
                    parsed = urlsplit(self.server)
                    url = urlunsplit(("wss" if parsed.scheme == "https" else "ws", parsed.netloc,
                                     parsed.path + f"/tasks/{session_id}/stream", "", ""))
                    async with websockets.connect(url, max_size=self.ws_max_size, ping_timeout=self.ping_timeout, proxy=None) as ws:
                        self.status = "streaming"
                        logger.warning("[Sentinel] STREAMING session=%s; logs=%s", session_id, self.output_dir)
                        while not self._stop.is_set():
                            source = await self._next()
                            packet = self._encode(source)
                            started = time.monotonic()
                            await ws.send(json.dumps(packet, allow_nan=False))
                            result = json.loads(await asyncio.wait_for(ws.recv(), self.timeout))
                            if "error" in result:
                                raise RuntimeError(result["error"])
                            self.responses += 1
                            result["client_roundtrip_ms"] = round((time.monotonic() - started) * 1000, 1)
                            result["task_session_id"] = session_id
                            result["dropped_observations"] = self.dropped
                            results.write(json.dumps(result, ensure_ascii=False) + "\n")
                            results.flush()
                            if observations:
                                observations.write(json.dumps({**packet, "kind": "update", "task_session_id": session_id}) + "\n")
                                observations.flush()
                            tracks = result.get("perception", {}).get("tracks", [])
                            logger.info("[Sentinel] t=%.2f visible=%d/%d failure=%s server=%sms dropped=%d",
                                        packet["timestamp_sec"], sum(t["visible"] for t in tracks), len(tracks),
                                        result.get("rules", {}).get("failure_detected"), result.get("server_processing_ms"), self.dropped)
                            self.latest_result = result
                            self.latest_visual = (result, source.frames)
                            self.latest_robot_state = source.robot_state
                            if result.get("rules", {}).get("failure_detected"):
                                self.fault_result = result
                                logger.warning("[Sentinel] FAILURE %s", json.dumps(
                                    result.get("rules", {}).get("findings", []), ensure_ascii=False))
                            if packet["manual_failure"]:
                                self._manual_busy = False
                            await asyncio.sleep(max(0, self.interval - (time.monotonic() - started)))
            finally:
                if observations:
                    observations.close()
                if session_id:
                    try:
                        async with httpx.AsyncClient(timeout=5, trust_env=False) as cleanup:
                            await cleanup.delete(self.server + f"/tasks/{session_id}")
                    except httpx.HTTPError:
                        pass

#!/usr/bin/env python3
"""Send two live camera streams to a RoboSentinel GPU server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import httpx
import websockets

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failure_detection.transport.codec import encode_jpeg


def source(value: str):
    return int(value) if value.isdigit() else value


def read_pair(external: cv2.VideoCapture, wrist: cv2.VideoCapture):
    ok_external, external_frame = external.read()
    ok_wrist, wrist_frame = wrist.read()
    if not ok_external or not ok_wrist:
        raise RuntimeError("Failed to read both camera streams")
    return {
        "external": encode_jpeg(external_frame),
        "wrist": encode_jpeg(wrist_frame),
    }


async def run(args: argparse.Namespace) -> None:
    external = cv2.VideoCapture(source(args.external))
    wrist = cv2.VideoCapture(source(args.wrist))
    if not external.isOpened() or not wrist.isOpened():
        raise RuntimeError("Cannot open camera streams")
    started = time.monotonic()
    try:
        initial = {
            "task_id": args.task_id,
            "description": args.description,
            "timestamp_sec": 0.0,
            "frames": read_pair(external, wrist),
        }
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            response = await client.post(f"{args.server}/tasks", json=initial)
            response.raise_for_status()
            session_id = response.json()["task_session_id"]
        ws_base = args.server.replace("http://", "ws://").replace("https://", "wss://")
        async with websockets.connect(f"{ws_base}/tasks/{session_id}/stream", max_size=2**24) as websocket:
            interval = 1.0 / args.fps
            while True:
                timestamp = time.monotonic() - started
                await websocket.send(json.dumps({
                    "timestamp_sec": timestamp,
                    "frames": read_pair(external, wrist),
                    "robot_state": None,
                }))
                print(await websocket.recv(), flush=True)
                delay = interval - (time.monotonic() - started - timestamp)
                if delay > 0:
                    await asyncio.sleep(delay)
    finally:
        external.release()
        wrist.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--external", required=True, help="Camera index or stream URL")
    parser.add_argument("--wrist", required=True, help="Camera index or stream URL")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

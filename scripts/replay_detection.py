#!/usr/bin/env python3
"""Exercise the real server from paired videos or recorded observations, without robot hardware."""
import argparse
import json
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from failure_detection.transport.codec import decode_jpeg
from failure_detection.transport.live_client import LiveDetectionClient, ObservationPacket
from project_config import DEFAULT_CONFIG, load_project_config, client_settings, resolve_path


def video_packets(external, wrist):
    captures = [cv2.VideoCapture(str(path)) for path in (external, wrist)]
    try:
        if not all(cap.isOpened() for cap in captures):
            raise ValueError("Cannot open both videos")
        rates = [cap.get(cv2.CAP_PROP_FPS) for cap in captures]
        if min(rates) <= 0 or abs(rates[0] - rates[1]) > 0.1:
            raise ValueError("Replay requires paired videos with matching known FPS")
        index = 0
        while True:
            pair = [cap.read() for cap in captures]
            if not all(ok for ok, _ in pair):
                return
            yield ObservationPacket(index / rates[0], {
                name: cv2.cvtColor(pair[i][1], cv2.COLOR_BGR2RGB)
                for i, name in enumerate(("external", "wrist"))}, None,
                {"video_frame_index": index})
            index += 1
    finally:
        for cap in captures:
            cap.release()


def recorded_packets(path):
    session = None
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            if session is None:
                session = item.get("task_session_id")
            elif item.get("task_session_id") != session:
                raise ValueError("Use a recording containing a single session")
            yield ObservationPacket(item["timestamp_sec"], {
                k: cv2.cvtColor(decode_jpeg(v), cv2.COLOR_BGR2RGB) for k, v in item["frames"].items()
            }, item.get("robot_state"), item.get("synchronization"))


def wait_for(client, predicate, timeout=600):
    deadline = time.monotonic() + timeout
    while not predicate():
        if client.error:
            raise RuntimeError(client.error)
        if time.monotonic() > deadline:
            raise TimeoutError("No server progress before timeout")
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--server")
    parser.add_argument("--external", type=Path)
    parser.add_argument("--wrist", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--description")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--simulate-fault-at", type=float, help="Replay-only synthetic controller fault at this second")
    args = parser.parse_args()
    config, root = load_project_config(args.config)
    client = client_settings(config, root, "replay")
    replay = config.get("replay", {})
    explicit_source = any((args.observations, args.external, args.wrist))
    if not explicit_source:
        for key in ("observations", "external", "wrist"):
            value = replay.get(key)
            setattr(args, key, resolve_path(root, value) if value else None)
        if args.observations:
            args.external = args.wrist = None
    for key, value in {"server": client.get("server", "http://127.0.0.1:8000"),
                       "description": client["description"], "fps": client.get("fps", 5),
                       "seconds": replay.get("seconds", 30), "output": client["output_dir"],
                       "simulate_fault_at": replay.get("simulate_fault_at")}.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.seconds <= 0 or args.fps <= 0:
        parser.error("seconds and fps must be positive")
    if args.observations:
        if args.external or args.wrist:
            parser.error("Choose observations or a video pair")
        packets = recorded_packets(args.observations)
    elif args.external and args.wrist:
        packets = video_packets(args.external, args.wrist)
    else:
        parser.error("Provide --external and --wrist, or --observations")
    logging.basicConfig(level=client.get("log_level", "INFO"), format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    timeout = float(client.get("timeout_sec", 600))
    client = LiveDetectionClient(args.server, args.description, client["task_id"], args.fps, args.output,
                                 client.get("record_observations", True), timeout,
                                 client.get("jpeg_quality", 85), client.get("ws_max_size", 2**24),
                                 client.get("ping_timeout_sec", 120)).start()
    try:
        first = next(packets)
        first_state = dict(first.robot_state) if first.robot_state else None
        if first_state is not None:
            first_state["original_timestamp_sec"] = first_state.get("timestamp_sec")
            first_state["timestamp_sec"] = 0
        client.submit(ObservationPacket(0, first.frames, first_state, first.synchronization))
        wait_for(client, lambda: client.status == "streaming", timeout)
        started, previous, final = time.monotonic(), 0.0, None
        for packet in packets:
            stamp = packet.timestamp_sec - first.timestamp_sec
            if stamp > args.seconds:
                break
            if stamp - previous < 1 / args.fps:
                continue
            time.sleep(max(0, started + stamp - time.monotonic()))
            state = dict(packet.robot_state) if packet.robot_state else None
            if state is not None:
                state["original_timestamp_sec"] = state.get("timestamp_sec")
                state["timestamp_sec"] = stamp
            if args.simulate_fault_at is not None and stamp >= args.simulate_fault_at:
                state = {**(state or {}), "timestamp_sec": stamp, "controller_state": "fault", "synthetic": True}
            client.submit(ObservationPacket(stamp, packet.frames, state, packet.synchronization))
            previous, final = stamp, stamp
            if client.error:
                raise RuntimeError(client.error)
        if final is None:
            raise ValueError("Recording too short to send an update")
        wait_for(client, lambda: client.latest_result is not None and client.latest_result["timestamp_sec"] >= final, timeout)
        if args.simulate_fault_at is not None and "controller_fault" not in client.diagnosed_rules:
            raise RuntimeError("No controller fault diagnosis received; check clip duration and enabled rules")
        print(f"PASS: {client.responses} responses; output={args.output}")
        if args.simulate_fault_at is not None:
            print("Synthetic controller state was used only in replay; inspect results.jsonl for diagnosis.")
    finally:
        packets.close()
        client.close()


if __name__ == "__main__":
    main()

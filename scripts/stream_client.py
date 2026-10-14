#!/usr/bin/env python3
"""Standalone dual-camera client. Do not run alongside teleoperation cameras."""
import argparse
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from project_config import DEFAULT_CONFIG, load_project_config, client_settings
from failure_detection.transport.live_client import LiveDetectionClient, ObservationPacket


def source(value):
    value = str(value)
    return int(value) if value.isdigit() else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for key in ("server", "external", "wrist", "task-id", "description"):
        parser.add_argument("--" + key)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    config, root = load_project_config(args.config)
    settings = client_settings(config, root, "cameras")
    settings.update(config["camera_client"])
    for key, value in vars(args).items():
        if value is not None and key != "config":
            settings["timeout_sec" if key == "timeout" else key] = value
    logging.basicConfig(level=settings.get("log_level", "INFO"), format="%(message)s")
    cameras = {name: cv2.VideoCapture(source(settings[name])) for name in ("external", "wrist")}
    client = LiveDetectionClient(
        settings["server"], settings["description"], settings["task_id"], settings["fps"],
        settings["output_dir"], settings["record_observations"], settings["timeout_sec"],
        settings["jpeg_quality"], settings["ws_max_size"], settings["ping_timeout_sec"],
    )
    try:
        if not all(camera.isOpened() for camera in cameras.values()):
            raise RuntimeError("Cannot open both cameras")
        client.start()
        origin = time.monotonic()
        while True:
            started = time.monotonic()
            frames = {}
            for name, camera in cameras.items():
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Cannot read " + name)
                frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            client.submit(ObservationPacket(time.monotonic() - origin, frames))
            if client.error:
                raise RuntimeError(client.error)
            time.sleep(max(0, client.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        for camera in cameras.values():
            camera.release()
        client.close()


if __name__ == "__main__":
    main()

"""Persist each server result before it is acknowledged to the client."""
import json
import logging
import os
from pathlib import Path

import cv2

from project_config import resolve_path


class ResultStore:
    def __init__(self, config, session_id):
        self.options = config.get("outputs", {})
        directory = self.options.get("server_dir")
        self.directory = (resolve_path(Path(config.get("_config_root", ".")), directory) / session_id
                          if directory else None)
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=False)

    def write(self, result, frames=None):
        if self.directory:
            if self.options.get("save_annotated_frames", False) and frames:
                tracks = (result.get("perception") or {}).get("tracks", [])
                images = {}
                for camera in ("external", "wrist"):
                    image = frames[camera].image.copy()
                    for track in tracks:
                        box = track.get("box_xyxy")
                        if track["camera"] == camera and track["visible"] and box:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
                            cv2.putText(image, track["target_id"], (max(0, x1), max(16, y1)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
                    path = self.directory / f'{result["frame_id"]:08d}_{camera}.jpg'
                    if not cv2.imwrite(str(path), image):
                        raise OSError(f"Cannot save annotated image: {path}")
                    images[camera] = str(path)
                result["annotated_frames"] = images
            result["server_result_file"] = str(self.directory / "results.jsonl")
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
            with (self.directory / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                if self.options.get("fsync", False):
                    os.fsync(handle.fileno())
        if self.options.get("console", False):
            logging.getLogger("uvicorn.error").info("RESULT %s", json.dumps(result, ensure_ascii=False))
        return result

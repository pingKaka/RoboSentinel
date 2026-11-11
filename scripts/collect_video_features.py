#!/usr/bin/env python3
"""Collect standard visual features from paired videos without robot state."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from failure_detection.backends import create_online_tracker
from failure_detection.features import FEATURE_SCHEMA_VERSION, StandardFeatureExtractor, feature_catalog
from failure_detection.pipeline import Frame, SceneSnapshot, StreamObservation, TrackingPlan, TrackingTarget
from project_config import DEFAULT_CONFIG, load_project_config, resolve_path


DEFAULT_TARGETS = (
    TrackingTarget("bottle", "white-capped reagent bottle", "manipulated_object"),
    TrackingTarget("cup", "black cup", "destination"),
    TrackingTarget("gripper", "robot gripper", "gripper"),
)


def video_pairs(directory: Path):
    for external in sorted(directory.glob("exterior_*.mp4")):
        wrist = directory / external.name.replace("exterior_", "wrist_", 1)
        if not wrist.is_file():
            raise FileNotFoundError(f"Missing paired video: {wrist}")
        yield external.stem.removeprefix("exterior_"), external, wrist


def parse_target(value: str) -> TrackingTarget:
    parts = value.split(":", 2)
    if len(parts) != 3 or parts[2] not in {"manipulated_object", "destination", "gripper", "other"}:
        raise argparse.ArgumentTypeError("target must be ID:VISUAL_NAME:ROLE")
    return TrackingTarget(*parts)


def read_pair(captures, index, fps):
    pair = [capture.read() for capture in captures]
    if not all(ok for ok, _ in pair):
        return None
    timestamp = index / fps
    return {name: Frame(name, timestamp, index, pair[position][1])
            for position, name in enumerate(("external", "wrist"))}


def validate_captures(captures, paths):
    if not all(capture.isOpened() for capture in captures):
        raise ValueError(f"Cannot open video pair: {paths}")
    rates = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
    if min(rates) <= 0 or abs(rates[0] - rates[1]) > 0.1:
        raise ValueError(f"Video FPS mismatch: {paths}")
    counts = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures]
    if counts[0] != counts[1]:
        raise ValueError(f"Video frame-count mismatch: {paths}")
    return rates[0]


def update_counts(snapshot, counts):
    counts["snapshots"] += 1
    counts["vision_valid"] += int(snapshot.quality.vision_valid)
    counts["robot_valid"] += int(snapshot.quality.robot_state_valid)
    counts["relations"] += len(snapshot.relations)
    counts["near_duplicate_relations"] += sum(relation.mask_iou >= 0.95 for relation in snapshot.relations)
    for entity in snapshot.entities:
        for camera in entity.cameras:
            counts["entity_camera_slots"] += 1
            counts["visibility_known"] += int(camera.visibility != "unknown")
            counts["visible"] += int(camera.visibility == "visible")
            counts["geometry"] += int(camera.mask_centroid_xy is not None)
            counts["tracking_quality"] += int(camera.tracking_quality is not None)


def rates(counts):
    snapshots = max(1, counts["snapshots"])
    slots = max(1, counts["entity_camera_slots"])
    return {
        **counts,
        "vision_valid_rate": counts["vision_valid"] / snapshots,
        "robot_valid_rate": counts["robot_valid"] / snapshots,
        "visibility_known_rate": counts["visibility_known"] / slots,
        "visible_rate": counts["visible"] / slots,
        "geometry_rate": counts["geometry"] / slots,
        "tracking_quality_rate": counts["tracking_quality"] / slots,
    }


def empty_counts():
    return {key: 0 for key in (
        "snapshots", "vision_valid", "robot_valid", "relations", "entity_camera_slots",
        "near_duplicate_relations", "visibility_known", "visible", "geometry", "tracking_quality",
    )}


def write_flat_features(entity_writer, relation_writer, video, frame_index, snapshot):
    for entity in snapshot.entities:
        for camera in entity.cameras:
            center = camera.mask_centroid_xy or (None, None)
            entity_writer.writerow({
                "video": video,
                "frame_index": frame_index,
                "timestamp_sec": snapshot.timestamp_sec,
                "camera": camera.camera,
                "target_id": entity.target_id,
                "target_name": entity.name,
                "role": entity.role,
                "visibility": camera.visibility,
                "mask_centroid_x": center[0],
                "mask_centroid_y": center[1],
                "mask_area_ratio": camera.mask_area_ratio,
                "mask_perimeter_normalized": camera.mask_perimeter_normalized,
                "mask_component_count": camera.mask_component_count,
                "mask_compactness": camera.mask_compactness,
                "tracking_quality": camera.tracking_quality,
                "visible_camera_count": entity.visible_camera_count,
                "visible_any": entity.visible_any,
                "missing_all_cameras": entity.missing_all_cameras,
            })
    for relation in snapshot.relations:
        relation_writer.writerow({
            "video": video,
            "frame_index": frame_index,
            "timestamp_sec": snapshot.timestamp_sec,
            "camera": relation.camera,
            "entity_a": relation.entity_a,
            "entity_b": relation.entity_b,
            "delta_x": relation.delta_xy[0],
            "delta_y": relation.delta_xy[1],
            "centroid_distance": relation.centroid_distance,
            "mask_iou": relation.mask_iou,
            "mask_intersection_ratio": relation.mask_intersection_ratio,
            "mask_coverage_a": relation.mask_coverage_a,
            "mask_coverage_b": relation.mask_coverage_b,
        })


ENTITY_FIELDS = (
    "video", "frame_index", "timestamp_sec", "camera", "target_id", "target_name", "role",
    "visibility", "mask_centroid_x", "mask_centroid_y", "mask_area_ratio",
    "mask_perimeter_normalized", "mask_component_count", "mask_compactness", "tracking_quality",
    "visible_camera_count", "visible_any", "missing_all_cameras",
)
RELATION_FIELDS = (
    "video", "frame_index", "timestamp_sec", "camera", "entity_a", "entity_b", "delta_x",
    "delta_y", "centroid_distance", "mask_iou", "mask_intersection_ratio", "mask_coverage_a",
    "mask_coverage_b",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video-dir", type=Path, default=Path("video"))
    parser.add_argument("--output", type=Path, default=Path("result/feature_collection/video_features.jsonl"))
    parser.add_argument("--entity-csv", type=Path, help="Flat per-frame entity feature CSV.")
    parser.add_argument("--relation-csv", type=Path, help="Flat per-frame spatial relation CSV.")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Run the tracker on every Nth source frame; default validates every frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum collected frames per pair; 0 means all.")
    parser.add_argument("--target", action="append", type=parse_target,
                        help="ID:VISUAL_NAME:ROLE; repeat for multiple targets.")
    args = parser.parse_args()
    if args.frame_step < 1 or args.max_frames < 0:
        parser.error("frame-step must be positive and max-frames cannot be negative")
    pairs = tuple(video_pairs(args.video_dir))
    if not pairs:
        parser.error(f"No exterior_*.mp4 videos found in {args.video_dir}")

    config, root = load_project_config(args.config)
    paths = {name: str(resolve_path(root, value)) for name, value in config["models"].items()}
    backend = create_online_tracker(paths["grounding_dino"], paths["sam"], config.get("backend", {}))
    plan = TrackingPlan("complete", tuple(args.target or DEFAULT_TARGETS), "fixed_video_test_plan")
    extractor = StandardFeatureExtractor()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    entity_csv = args.entity_csv or args.output.with_name("video_entity_features.csv")
    relation_csv = args.relation_csv or args.output.with_name("video_relations.csv")
    entity_csv.parent.mkdir(parents=True, exist_ok=True)
    relation_csv.parent.mkdir(parents=True, exist_ok=True)
    total = empty_counts()
    pair_reports = {}

    with (args.output.open("w", encoding="utf-8") as output,
          entity_csv.open("w", encoding="utf-8", newline="") as entity_output,
          relation_csv.open("w", encoding="utf-8", newline="") as relation_output):
        entity_writer = csv.DictWriter(entity_output, fieldnames=ENTITY_FIELDS)
        relation_writer = csv.DictWriter(relation_output, fieldnames=RELATION_FIELDS)
        entity_writer.writeheader()
        relation_writer.writeheader()
        for label, external, wrist in pairs:
            captures = [cv2.VideoCapture(str(path)) for path in (external, wrist)]
            pair_counts = empty_counts()
            try:
                fps = validate_captures(captures, (external, wrist))
                frames = read_pair(captures, 0, fps)
                if frames is None:
                    raise ValueError(f"Empty video pair: {external}, {wrist}")
                backend.initialize(plan, SceneSnapshot(0.0, frames))
                extractor.reset()
                snapshot = extractor.extract(plan, StreamObservation(0.0, frames), backend.initial_perception)
                output.write(json.dumps({"video": label, "frame_index": 0, "features": asdict(snapshot)},
                                        ensure_ascii=False, allow_nan=False) + "\n")
                write_flat_features(entity_writer, relation_writer, label, 0, snapshot)
                update_counts(snapshot, pair_counts)
                update_counts(snapshot, total)
                collected, index = 1, 0
                while args.max_frames == 0 or collected < args.max_frames:
                    index += 1
                    frames = read_pair(captures, index, fps)
                    if frames is None:
                        break
                    if index % args.frame_step:
                        continue
                    observation = StreamObservation(index / fps, frames)
                    snapshot = extractor.extract(plan, observation, backend.update(observation))
                    output.write(json.dumps({"video": label, "frame_index": index, "features": asdict(snapshot)},
                                            ensure_ascii=False, allow_nan=False) + "\n")
                    write_flat_features(entity_writer, relation_writer, label, index, snapshot)
                    update_counts(snapshot, pair_counts)
                    update_counts(snapshot, total)
                    collected += 1
            finally:
                for capture in captures:
                    capture.release()
            pair_reports[label] = rates(pair_counts)
            print(f"{label}: {pair_reports[label]}")

    report = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "validation_scope": "standard_feature_collection_only",
        "fault_judgment_performed": False,
        "source_frame_step": args.frame_step,
        "video_directory": str(args.video_dir.resolve()),
        "output": str(args.output.resolve()),
        "entity_csv": str(entity_csv.resolve()),
        "relation_csv": str(relation_csv.resolve()),
        "targets": [asdict(target) for target in plan.targets],
        "total": rates(total),
        "videos": pair_reports,
    }
    report_path = args.output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_name("feature_catalog.json").write_text(
        json.dumps(feature_catalog(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["total"], ensure_ascii=False, indent=2))
    print(f"features={args.output}\nentities={entity_csv}\nrelations={relation_csv}\nsummary={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

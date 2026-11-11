from dataclasses import asdict
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from failure_detection.features import FEATURE_CATALOG, FeatureContext, StandardFeatureExtractor
from failure_detection.pipeline import Frame, PerceptionUpdate, StreamObservation, TrackState, TrackingPlan, TrackingTarget


class StandardFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = (Path("video/exterior_success.mp4"), Path("video/wrist_success.mp4"))
        captures = [cv2.VideoCapture(str(path)) for path in paths]
        try:
            if not all(capture.isOpened() for capture in captures):
                raise unittest.SkipTest("video/success pair is unavailable")
            pair = [capture.read() for capture in captures]
            if not all(ok for ok, _ in pair):
                raise unittest.SkipTest("video/success pair has no frames")
            cls.images = {name: pair[index][1] for index, name in enumerate(("external", "wrist"))}
        finally:
            for capture in captures:
                capture.release()

    def setUp(self):
        self.plan = TrackingPlan("complete", (
            TrackingTarget("object", "white bottle", "manipulated_object"),
            TrackingTarget("goal", "black cup", "destination"),
        ), "test")
        self.frames = {name: Frame(name, 1.0, 15, image) for name, image in self.images.items()}

    def test_collects_mask_visual_and_relation_features_from_video_frames(self):
        height, width = self.images["external"].shape[:2]
        object_mask = np.zeros((height, width), dtype=np.uint8)
        object_mask[:height // 2, :width // 2] = 1
        goal_mask = np.zeros((height, width), dtype=np.uint8)
        goal_mask[height // 4:height * 3 // 4, width // 4:width * 3 // 4] = 1
        tracks = (
            TrackState("object", "external", True, mask=object_mask, quality=0.8),
            TrackState("goal", "external", True, mask=goal_mask),
            TrackState("object", "wrist", False),
            TrackState("goal", "wrist", False),
        )
        snapshot = StandardFeatureExtractor().extract(
            self.plan, StreamObservation(1.0, self.frames),
            PerceptionUpdate(1.0, "complete", tracks, "test"),
        )
        external = snapshot.entity("object").cameras[0]
        self.assertEqual(external.visibility, "visible")
        self.assertEqual(external.mask_centroid_xy, (0.25, 0.25))
        self.assertEqual(external.mask_area_ratio, 0.25)
        self.assertEqual(external.mask_component_count, 1)
        self.assertGreater(external.mask_perimeter_normalized, 0)
        self.assertGreater(external.mask_compactness, 0)
        self.assertEqual(len(snapshot.relations), 1)
        self.assertAlmostEqual(snapshot.relations[0].centroid_distance, 2 ** -1.5)
        self.assertAlmostEqual(snapshot.relations[0].mask_iou, 1 / 7)
        self.assertAlmostEqual(snapshot.relations[0].mask_coverage_a, 0.25)
        self.assertAlmostEqual(snapshot.relations[0].mask_coverage_b, 0.25)
        self.assertTrue(snapshot.quality.vision_valid)
        self.assertFalse(snapshot.quality.robot_state_valid)
        json.dumps(asdict(snapshot), allow_nan=False)

    def test_robot_schema_keeps_only_tcp_position_and_gripper(self):
        state = {"tcp_pose": [1, 2, 3, 4, 5, 6], "tcp_pose_frame": "base",
                 "gripper_position_normalized": 0.4, "controller_state": "fault",
                 "tcp_speed": [9, 9, 9, 9, 9, 9]}
        snapshot = StandardFeatureExtractor().extract(
            self.plan, StreamObservation(1.0, self.frames, state),
            PerceptionUpdate(1.0, "complete", (), "test"),
        )
        self.assertEqual(snapshot.robot.tcp_position_m, (1.0, 2.0, 3.0))
        self.assertEqual(snapshot.robot.gripper_position_normalized, 0.4)
        self.assertEqual(set(asdict(snapshot.robot)), {
            "tcp_position_m", "tcp_frame", "gripper_position_normalized",
        })

    def test_unknown_visibility_and_catalog_boundaries(self):
        snapshot = StandardFeatureExtractor().extract(
            self.plan, StreamObservation(1.0, self.frames),
            PerceptionUpdate(1.0, "incomplete", (), "test"),
        )
        self.assertIsNone(snapshot.entity("object").visible_any)
        context = FeatureContext(snapshot, (snapshot,))
        self.assertEqual(context.snapshot.schema_version, "2.0.0")
        paths = {item.path for item in FEATURE_CATALOG}
        self.assertIn("robot.tcp_position_m", paths)
        self.assertIn("robot.gripper_position_normalized", paths)
        self.assertFalse(any("speed" in path or "controller" in path or "joint" in path for path in paths))
        self.assertFalse(any("bbox" in path or "size_wh" in path for path in paths))
        self.assertIn("relations.*.mask_iou", paths)

    def test_ambiguous_tracker_entity_is_unknown(self):
        tracks = (
            TrackState("object", "external", False, observation_valid=False),
            TrackState("goal", "external", False, observation_valid=False),
        )
        snapshot = StandardFeatureExtractor().extract(
            self.plan, StreamObservation(1.0, self.frames),
            PerceptionUpdate(1.0, "complete", tracks, "test"),
        )
        self.assertEqual(snapshot.entity("object").cameras[0].visibility, "unknown")
        self.assertIsNone(snapshot.entity("object").visible_any)


if __name__ == "__main__":
    unittest.main()

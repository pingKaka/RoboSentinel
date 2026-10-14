from __future__ import annotations

import argparse
import logging
import time
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VideoTeleopConfig:
    fps: int
    debug: bool
    record_key: str
    exit_key: str
    output_dir: Path
    preview: bool

    port: str
    use_gripper: bool
    hardware_offsets: list[float]
    joint_ids: list[int]
    joint_offsets: list[float]
    joint_signs: list[int]
    gripper_config: tuple[int, float, float]
    control_mode: str

    robot_ip: str
    gripper_port: str
    close_threshold: float
    gripper_reverse: bool
    gripper_bin_threshold: float
    gripper_force: int
    gripper_speed: int
    control_space: str
    tcp_force_reference_frame: str
    tcp_position_reference_frame: str
    tcp_position_speed: float
    tcp_position_acceleration: float
    tcp_position_servo_time: float
    tcp_position_lookahead_time: float
    tcp_position_gain: int
    robot_urdf_path: str
    kp: int
    kd: int
    kp_rot: int
    kd_rot: int
    rtde_freq: int
    select_vector: list[int]
    force_limit: list[int]
    pos_delta: float
    vel_delta: float
    gain_scale: float
    look_ahead_time: float
    dt: float
    gain: int

    wrist_cam_serial: str
    exterior_cam_serial: str
    width: int
    height: int
    sentinel: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "VideoTeleopConfig":
        path = path.expanduser().resolve()
        with open(path, "r") as f:
            root = yaml.safe_load(f)
        if "project_config" in root:
            return cls.from_yaml(path.parent / root["project_config"])
        if "client" in root:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from project_config import client_settings, resolve_path
            settings = client_settings(root, path.parent, "teleop")
            settings.update(root["video_teleop"].get("sentinel", {}))
            root["video_teleop"]["sentinel"] = settings
            root["video_teleop"]["output_dir"] = str(resolve_path(path.parent, root["video_teleop"]["output_dir"]))

        cfg = root["video_teleop"]
        teleop = cfg["teleop"]
        dxl = teleop["dynamixel_config"]
        robot = cfg["robot"]
        force = robot["force_mode"]
        joint = robot["joint_mode"]
        cams = cfg["cameras"]

        tcp_position = robot.get("tcp_position", {})
        return cls(
            fps=int(cfg.get("fps", 15)),
            debug=bool(cfg.get("debug", True)),
            record_key=str(cfg.get("record_key", "space")).lower(),
            exit_key=str(cfg.get("exit_key", "esc")).lower(),
            output_dir=Path(cfg.get("output_dir", "videos")),
            preview=bool(cfg.get("preview", True)),
            port=dxl["port"],
            use_gripper=bool(dxl["use_gripper"]),
            hardware_offsets=list(dxl["hardware_offsets"]),
            joint_ids=list(dxl["joint_ids"]),
            joint_offsets=list(dxl["joint_offsets"]),
            joint_signs=list(dxl["joint_signs"]),
            gripper_config=tuple(dxl["gripper_config"]),
            control_mode=teleop.get("control_mode", "isoteleop"),
            robot_ip=robot["ip"],
            gripper_port=robot["gripper_port"],
            close_threshold=float(robot["close_threshold"]),
            gripper_reverse=bool(robot["gripper_reverse"]),
            gripper_bin_threshold=float(robot["gripper_bin_threshold"]),
            gripper_force=int(robot.get("gripper_force", 70)),
            gripper_speed=int(robot.get("gripper_speed", 60)),
            control_space=robot["control_space"],
            tcp_force_reference_frame=robot.get("tcp_force", {}).get("reference_frame", "base"),
            tcp_position_reference_frame=tcp_position.get("reference_frame", "base"),
            tcp_position_speed=float(tcp_position.get("speed", 0.5)),
            tcp_position_acceleration=float(tcp_position.get("acceleration", 0.5)),
            tcp_position_servo_time=float(tcp_position.get("servo_time", 0.1)),
            tcp_position_lookahead_time=float(tcp_position.get("lookahead_time", 0.1)),
            tcp_position_gain=int(tcp_position.get("gain", 300)),
            robot_urdf_path=robot["robot_urdf_path"],
            kp=int(force["kp"]),
            kd=int(force["kd"]),
            kp_rot=int(force["kp_rot"]),
            kd_rot=int(force["kd_rot"]),
            rtde_freq=int(force["rtde_freq"]),
            select_vector=list(force["select_vector"]),
            force_limit=list(force["force_limit"]),
            pos_delta=float(force["pos_delta"]),
            vel_delta=float(force["vel_delta"]),
            gain_scale=float(force["gain_scale"]),
            look_ahead_time=float(joint["look_ahead_time"]),
            dt=float(joint["dt"]),
            gain=int(joint["gain"]),
            wrist_cam_serial=cams["wrist_cam_serial"],
            exterior_cam_serial=cams["exterior_cam_serial"],
            width=int(cams["width"]),
            height=int(cams["height"]),
            sentinel=dict(cfg.get("sentinel", {})),
        )


class ToggleKeyState:
    def __init__(self, record_key: str, exit_key: str):
        self.record_key = record_key
        self.exit_key = exit_key
        self.recording = False
        self.record_key_down = False
        self.stop_requested = False
        self.listener: keyboard.Listener | None = None

    def start(self) -> None:
        global keyboard
        from pynput import keyboard
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> bool | None:
        name = normalize_key(key)
        if name == self.exit_key:
            self.stop_requested = True
            return False
        if name == self.record_key and not self.record_key_down:
            self.record_key_down = True
            self.recording = not self.recording
            logger.info(
                "====== [VIDEO] Recording %s ======",
                "started" if self.recording else "stopped",
            )
        return None

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if normalize_key(key) == self.record_key:
            self.record_key_down = False


class VideoSegmentRecorder:
    def __init__(self, output_dir: Path, camera_name: str, fps: int):
        self.output_dir = output_dir
        self.camera_name = camera_name
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.path: Path | None = None
        self.size: tuple[int, int] | None = None
        self.segment_index = 0
        self.frame_count = 0

    @property
    def is_open(self) -> bool:
        return self.writer is not None

    def update(self, should_record: bool, frame_rgb: np.ndarray) -> None:
        if should_record and self.writer is None:
            self._open(frame_rgb)
        elif not should_record and self.writer is not None:
            self.close()

        if self.writer is not None:
            frame_bgr = prepare_bgr_frame(frame_rgb, self.size)
            self.writer.write(frame_bgr)
            self.frame_count += 1

    def _open(self, frame_rgb: np.ndarray) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frame_bgr = prepare_bgr_frame(frame_rgb)
        height, width = frame_bgr.shape[:2]
        self.size = (width, height)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.segment_index += 1
        self.frame_count = 0
        self.path = self.output_dir / f"{self.camera_name}_{timestamp}_{self.segment_index:03d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self.size)
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(f"Failed to open video writer: {self.path}")
        logger.info("====== [VIDEO] Recording started: %s ======", self.path)

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.release()
        logger.info(
            "====== [VIDEO] Recording stopped: %s (%d frames) ======",
            self.path,
            self.frame_count,
        )
        self.writer = None
        self.path = None
        self.size = None
        self.frame_count = 0


def normalize_key(key: keyboard.Key | keyboard.KeyCode) -> str:
    if isinstance(key, keyboard.KeyCode):
        return str(key.char).lower() if key.char is not None else ""
    return str(key).replace("Key.", "").lower()


def prepare_bgr_frame(frame: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB frame with 3 channels, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    if size is not None and (bgr.shape[1], bgr.shape[0]) != size:
        bgr = cv2.resize(bgr, size)
    return bgr


def make_robot_and_teleop(cfg: VideoTeleopConfig) -> tuple[Any, Any]:
    from lerobot.cameras.configs import ColorMode, Cv2Rotation
    from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
    from lerobot_robot_ur5e import UR5e, UR5eConfig
    from lerobot_teleoperator_ur5e import UR5eTeleop, UR5eTeleopConfig

    wrist_image_cfg = RealSenseCameraConfig(
        serial_number_or_name=cfg.wrist_cam_serial,
        fps=cfg.fps,
        width=cfg.width,
        height=cfg.height,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.NO_ROTATION,
    )
    exterior_image_cfg = RealSenseCameraConfig(
        serial_number_or_name=cfg.exterior_cam_serial,
        fps=cfg.fps,
        width=cfg.width,
        height=cfg.height,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.NO_ROTATION,
    )
    camera_config = {
        "wrist_image": wrist_image_cfg,
        "exterior_image": exterior_image_cfg,
    }

    teleop_config = UR5eTeleopConfig(
        port=cfg.port,
        use_gripper=cfg.use_gripper,
        hardware_offsets=cfg.hardware_offsets,
        joint_ids=cfg.joint_ids,
        joint_offsets=cfg.joint_offsets,
        joint_signs=cfg.joint_signs,
        gripper_config=cfg.gripper_config,
        control_mode=cfg.control_mode,
        control_space=cfg.control_space,
        tcp_force_reference_frame=cfg.tcp_force_reference_frame,
        tcp_position_reference_frame=cfg.tcp_position_reference_frame,
        robot_urdf_path=cfg.robot_urdf_path,
    )
    robot_config = UR5eConfig(
        robot_ip=cfg.robot_ip,
        gripper_port=cfg.gripper_port,
        cameras=camera_config,
        debug=cfg.debug,
        close_threshold=cfg.close_threshold,
        use_gripper=cfg.use_gripper,
        gripper_reverse=cfg.gripper_reverse,
        gripper_bin_threshold=cfg.gripper_bin_threshold,
        gripper_force=cfg.gripper_force,
        gripper_speed=cfg.gripper_speed,
        control_space=cfg.control_space,
        tcp_force_reference_frame=cfg.tcp_force_reference_frame,
        tcp_position_reference_frame=cfg.tcp_position_reference_frame,
        robot_urdf_path=cfg.robot_urdf_path,
        tcp_position_speed=cfg.tcp_position_speed,
        tcp_position_acceleration=cfg.tcp_position_acceleration,
        tcp_position_servo_time=cfg.tcp_position_servo_time,
        tcp_position_lookahead_time=cfg.tcp_position_lookahead_time,
        tcp_position_gain=cfg.tcp_position_gain,
        kp=cfg.kp,
        kd=cfg.kd,
        kp_rot=cfg.kp_rot,
        kd_rot=cfg.kd_rot,
        rtde_freq=cfg.rtde_freq,
        select_vector=cfg.select_vector,
        force_limit=cfg.force_limit,
        look_ahead_time=cfg.look_ahead_time,
        dt=cfg.dt,
        gain=cfg.gain,
        pos_delta=cfg.pos_delta,
        vel_delta=cfg.vel_delta,
        gain_scale=cfg.gain_scale,
    )

    robot = UR5e(robot_config)
    teleop = UR5eTeleop(teleop_config)
    teleop.set_robot(robot)
    return robot, teleop


def run_video_teleop(cfg: VideoTeleopConfig) -> None:
    if cfg.record_key == cfg.exit_key:
        raise ValueError("record_key and exit_key must be different.")

    output_dir = cfg.output_dir
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    robot = None
    teleop = None
    sentinel = None
    sentinel_adapter = None
    sentinel_origin = time.monotonic()
    last_sentinel_submit = -float("inf")
    key_state = ToggleKeyState(cfg.record_key, cfg.exit_key)
    recorders = {
        "wrist_image": VideoSegmentRecorder(output_dir, "wrist_image", cfg.fps),
        "exterior_image": VideoSegmentRecorder(output_dir, "exterior_image", cfg.fps),
    }

    try:
        if cfg.sentinel.get("enabled", False):
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from failure_detection.transport.live_client import LiveDetectionClient
            from failure_detection.transport.teleop_adapter import teleop_packet
            sentinel_adapter = teleop_packet
            sentinel = LiveDetectionClient(
                cfg.sentinel["server"], cfg.sentinel["description"],
                task_id=cfg.sentinel.get("task_id", "teleop"),
                fps=float(cfg.sentinel.get("fps", 5)),
                output_dir=cfg.sentinel.get("output_dir", "result/teleop"),
                record_observations=cfg.sentinel.get("record_observations", True),
                timeout=float(cfg.sentinel.get("timeout_sec", 600)),
                jpeg_quality=int(cfg.sentinel.get("jpeg_quality", 85)),
                ws_max_size=int(cfg.sentinel.get("ws_max_size", 2**24)),
                ping_timeout=float(cfg.sentinel.get("ping_timeout_sec", 120)),
            ).start()
        robot, teleop = make_robot_and_teleop(cfg)
        robot.connect()
        teleop.connect()
        robot.set_episode_reference_pose()
        key_state.start()

        logger.info(
            "====== [READY] Press '%s' once to start recording both cameras, "
            "press it again to stop. Press '%s' to exit. ======",
            cfg.record_key,
            cfg.exit_key,
        )

        while not key_state.stop_requested:
            t0 = time.perf_counter()
            action = teleop.get_action()
            robot.send_action(action)
            capture_started = time.monotonic()
            observation = robot.get_observation()
            capture_finished = time.monotonic()
            if sentinel is not None and capture_finished - last_sentinel_submit >= sentinel.interval:
                last_sentinel_submit = capture_finished
                try:
                    sentinel.submit(sentinel_adapter(observation, robot, sentinel_origin,
                                                     capture_started, capture_finished, action))
                except Exception:
                    logger.exception("[Sentinel] Observation rejected; teleoperation continues")

            camera_names = ("wrist_image", "exterior_image")
            for camera_name in camera_names:
                if camera_name not in observation:
                    raise KeyError(f"Camera '{camera_name}' was not found in observation.")
                recorders[camera_name].update(key_state.recording, observation[camera_name])

            if cfg.preview:
                previews = []
                for camera_name in camera_names:
                    preview = prepare_bgr_frame(observation[camera_name])
                    cv2.putText(
                        preview,
                        f"{camera_name}: {'REC' if key_state.recording else 'STANDBY'}"
                        + (f" | Sentinel: {sentinel.status}" if sentinel is not None else ""),
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255) if key_state.recording else (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    previews.append(preview)
                preview = np.hstack(previews)
                cv2.imshow("ur5e-video-teleop", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            sleep_s = 1.0 / cfg.fps - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        logger.info("\n====== [INFO] Ctrl+C detected ======")
    finally:
        for recorder in recorders.values():
            recorder.close()
        key_state.stop()
        cv2.destroyAllWindows()
        try:
            if robot is not None:
                robot.disconnect()
        finally:
            try:
                if teleop is not None:
                    teleop.disconnect()
            finally:
                if sentinel is not None:
                    sentinel.close()


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().parents[2] / "cfg.yaml"
    parser = argparse.ArgumentParser(
        description="UR5e teleoperation with toggle-to-record video from both cameras."
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--record-key", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--sentinel-server", help="Enable live detection at this HTTP URL")
    parser.add_argument("--task-description", help="Task text sent to the scene planner")
    parser.add_argument("--sentinel-output", type=Path, help="Detection result and observation directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VideoTeleopConfig.from_yaml(args.config)
    logging.getLogger().setLevel(cfg.sentinel.get("log_level", "INFO"))
    if args.record_key is not None:
        cfg.record_key = args.record_key.lower()
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.fps is not None:
        cfg.fps = args.fps
    if args.no_preview:
        cfg.preview = False
    if args.sentinel_server:
        cfg.sentinel.update(enabled=True, server=args.sentinel_server)
    if args.task_description:
        cfg.sentinel["description"] = args.task_description
    if args.sentinel_output:
        cfg.sentinel["output_dir"] = str(args.sentinel_output)
    run_video_teleop(cfg)


if __name__ == "__main__":
    main()

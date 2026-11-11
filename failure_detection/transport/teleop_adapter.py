"""Adapt the existing LeRobot UR observation without opening another device."""
import time

from .live_client import ObservationPacket


def teleop_packet(observation, robot, origin, capture_started, capture_finished, action=None):
    state = {
        "source": "ur5e_teleop",
        "gripper_position_normalized": (float(observation["gripper_raw_position"])
                                        if observation.get("gripper_raw_position") is not None else None),
    }
    receiver = getattr(robot, "_arm", {}).get("rtde_r")
    if receiver is not None:
        try:
            state["tcp_position"] = [float(v) for v in receiver.getActualTCPPose()[:3]]
            state["tcp_position_frame"] = "base"
        except Exception:
            state["tcp_position"] = None
    if state.get("tcp_position") is None:
        position = [observation.get(f"tcp_pose.{axis}") for axis in ("x", "y", "z")]
        if all(value is not None for value in position):
            state["tcp_position"] = [float(value) for value in position]
            reference = getattr(robot.config, robot.config.control_space + "_reference_frame", "base")
            state["tcp_position_frame"] = "episode_reference" if reference == "tcp" else reference
    sampled = time.monotonic()
    state["timestamp_sec"] = sampled - origin
    return ObservationPacket(
        capture_finished - origin,
        {"external": observation["exterior_image"], "wrist": observation["wrist_image"]},
        state,
        {"capture_span_sec": capture_finished - capture_started,
         "state_read_delay_sec": sampled - capture_finished},
    )

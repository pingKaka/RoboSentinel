"""Adapt the existing LeRobot UR observation without opening another device."""
import time

from .live_client import ObservationPacket


def teleop_packet(observation, robot, origin, capture_started, capture_finished, action=None):
    axes = ("x", "y", "z", "rx", "ry", "rz")

    def vector(keys):
        values = [observation.get(key) for key in keys]
        return [float(value) for value in values] if all(value is not None for value in values) else None

    state = {
        "source": "ur5e_teleop",
        "joint_position": vector([f"joint_{i}.pos" for i in range(1, 7)]),
        "joint_velocity": vector([f"joint_{i}.vel" for i in range(1, 7)]),
        "tcp_speed": vector([f"tcp_speed.{axis}" for axis in axes]),
        "tcp_force": vector([f"tcp_force.{axis}" for axis in axes]),
        "gripper_position_normalized": (float(observation["gripper_raw_position"])
                                        if observation.get("gripper_raw_position") is not None else None),
        "gripper_action_bin": (float(observation["gripper_action_bin"])
                               if observation.get("gripper_action_bin") is not None else None),
        # The existing observation may be relative EE Euler pose, not base TCP rotvec.
        "teleop_ee_pose": vector([f"tcp_pose.{axis}" for axis in axes]),
        "control_space": robot.config.control_space,
        "commanded_action": {k: float(v) for k, v in (action or {}).items()},
    }
    receiver = getattr(robot, "_arm", {}).get("rtde_r")
    if robot.config.control_space in {"tcp_force", "tcp_position"}:
        reference = getattr(robot.config, robot.config.control_space + "_reference_frame", "unknown")
        state["teleop_ee_pose_representation"] = "xyz_m_euler_rad"
        state["teleop_ee_pose_frame"] = "episode_reference" if reference == "tcp" else reference
    else:
        state["teleop_ee_pose_representation"] = "xyz_m_rotation_vector_rad"
        state["teleop_ee_pose_frame"] = "base"
    if receiver is not None:
        try:
            state["tcp_pose"] = [float(v) for v in receiver.getActualTCPPose()]
            state["tcp_pose_frame"] = "base_xyz_m_rotation_vector_rad"
            mode = int(receiver.getSafetyMode())
            state["safety_mode"] = mode
            state["controller_state"] = {
                1: "normal", 2: "normal", 3: "protective_stop", 5: "stopped",
                6: "emergency_stop", 7: "emergency_stop", 8: "fault", 9: "fault",
            }.get(mode, "unknown")
        except Exception as exc:
            state["controller_state"] = "unknown"
            state["state_read_error"] = str(exc)
    sampled = time.monotonic()
    state["timestamp_sec"] = sampled - origin
    return ObservationPacket(
        capture_finished - origin,
        {"external": observation["exterior_image"], "wrist": observation["wrist_image"]},
        state,
        {"capture_span_sec": capture_finished - capture_started,
         "state_read_delay_sec": sampled - capture_finished},
    )

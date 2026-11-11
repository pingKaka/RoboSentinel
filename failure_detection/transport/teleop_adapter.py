"""Adapt teleoperation observations to the robot-independent transport."""
import time

from .live_client import ObservationPacket


def teleop_packet(observation, state_provider, origin, capture_started, capture_finished, action=None):
    sampled = time.monotonic()
    state = state_provider.read(observation, sampled - origin).to_wire()
    return ObservationPacket(
        capture_finished - origin,
        {"external": observation["exterior_image"], "wrist": observation["wrist_image"]},
        state,
        {"capture_span_sec": capture_finished - capture_started,
         "state_read_delay_sec": sampled - capture_finished},
    )

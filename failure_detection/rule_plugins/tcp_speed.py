"""Example only: monitoring is not a robot safety controller."""
import math

from failure_detection.pipeline.contracts import RuleFinding


class TCPSpeedRule:
    rule_id = "tcp_speed"

    def __init__(self, limit_mps=0.25, seconds=0.5, max_gap_seconds=2.0):
        self.limit = float(limit_mps)
        self.seconds = float(seconds)
        self.max_gap = float(max_gap_seconds)
        if any(not math.isfinite(v) or v <= 0 for v in (self.limit, self.seconds, self.max_gap)):
            raise ValueError("Speed and time limits must be finite and positive")
        self.since = self.previous = None

    def update(self, context):
        now = context.observation.timestamp_sec
        velocity = (context.observation.robot_state or {}).get("tcp_speed")
        try:
            if len(velocity) < 3:
                raise ValueError("Missing linear velocity")
            speed = math.hypot(*(float(v) for v in velocity[:3]))
            if not math.isfinite(speed) or not math.isfinite(now):
                raise ValueError("Nonfinite observation")
        except (TypeError, ValueError, OverflowError):
            self.since = self.previous = None
            return RuleFinding(self.rule_id, "unknown")
        if self.previous is not None and (now <= self.previous or now - self.previous > self.max_gap):
            self.since = None
        self.previous = now
        evidence = {"speed_mps": speed, "limit_mps": self.limit}
        if speed <= self.limit:
            self.since = None
            return RuleFinding(self.rule_id, "clear", evidence=evidence)
        if self.since is None:
            self.since = now
        confirmed = now - self.since >= self.seconds
        return RuleFinding(self.rule_id, "confirmed" if confirmed else "unknown",
                           "tcp_overspeed" if confirmed else None, evidence=evidence)

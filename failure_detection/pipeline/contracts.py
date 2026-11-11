from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..features.contracts import FeatureSnapshot
    from ..rule_dsl.contracts import RulePackage


@dataclass(frozen=True)
class Task:
    task_id: str
    description: str


@dataclass(frozen=True)
class Frame:
    camera: str
    timestamp_sec: float
    frame_id: int
    image: Any


@dataclass(frozen=True)
class SceneSnapshot:
    timestamp_sec: float
    frames: dict[str, Frame]


@dataclass(frozen=True)
class TrackingTarget:
    target_id: str
    name: str
    role: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackingPlan:
    status: str
    targets: tuple[TrackingTarget, ...]
    planner: str | None


@dataclass(frozen=True)
class MonitoringPlan:
    tracking: TrackingPlan
    rule_package: RulePackage

    @property
    def status(self) -> str:
        return self.tracking.status

    @property
    def targets(self) -> tuple[TrackingTarget, ...]:
        return self.tracking.targets

    @property
    def planner(self) -> str | None:
        return self.tracking.planner


@dataclass(frozen=True)
class StreamObservation:
    timestamp_sec: float
    frames: dict[str, Frame]
    robot_state: dict[str, Any] | None = None
    synchronization: dict[str, float] = field(default_factory=dict)
    manual_failure: bool = False


@dataclass(frozen=True)
class TrackState:
    target_id: str
    camera: str
    visible: bool
    box_xyxy: tuple[float, float, float, float] | None = None
    mask: Any = None
    quality: float | None = None
    contours: list = field(default_factory=list)
    observation_valid: bool = True


@dataclass(frozen=True)
class PerceptionUpdate:
    timestamp_sec: float
    status: str
    tracks: tuple[TrackState, ...]
    backend: str | None


@dataclass(frozen=True)
class RuleFinding:
    rule_id: str
    status: str
    failure_type: str | None = None
    confidence: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleReport:
    timestamp_sec: float
    status: str
    findings: tuple[RuleFinding, ...]
    failure_detected: bool | None


@dataclass(frozen=True)
class PipelineUpdate:
    timestamp_sec: float
    perception: PerceptionUpdate
    features: FeatureSnapshot
    rules: RuleReport

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

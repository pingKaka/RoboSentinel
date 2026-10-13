from __future__ import annotations

from collections.abc import Iterable

from .contracts import RuleContext, RuleReport
from .interfaces import StreamingFailureRule


class StreamingRuleEngine:
    def __init__(self, rules: Iterable[StreamingFailureRule]) -> None:
        self.rules = tuple(rules)

    def update(self, context: RuleContext) -> RuleReport:
        if not self.rules:
            return RuleReport(context.observation.timestamp_sec, "unavailable", (), None)
        findings = tuple(rule.update(context) for rule in self.rules)
        confirmed = any(item.status == "confirmed" for item in findings)
        unknown = any(item.status == "unknown" for item in findings)
        return RuleReport(
            context.observation.timestamp_sec,
            "complete",
            findings,
            True if confirmed else None if unknown else False,
        )

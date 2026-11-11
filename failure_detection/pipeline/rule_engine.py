from __future__ import annotations

from collections.abc import Iterable

from ..features.contracts import FeatureContext
from ..rule_dsl import compile_rules
from .contracts import RuleReport
from .interfaces import StreamingFailureRule


class StreamingRuleEngine:
    def __init__(self, rules: Iterable[StreamingFailureRule] = ()) -> None:
        self.rules = tuple(rules)

    def initialize(self, task, plan, rule_package) -> None:
        if rule_package is not None and rule_package.rules:
            self.rules = compile_rules(rule_package.rules)

    def update(self, context: FeatureContext) -> RuleReport:
        if not self.rules:
            return RuleReport(context.snapshot.timestamp_sec, "unavailable", (), None)
        findings = tuple(rule.update(context) for rule in self.rules)
        confirmed = any(item.status == "confirmed" for item in findings)
        unknown = any(item.status == "unknown" for item in findings)
        return RuleReport(
            context.snapshot.timestamp_sec,
            "complete",
            findings,
            True if confirmed else None if unknown else False,
        )

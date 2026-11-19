"""
Assignment 11 — Monitoring & Alerts (implemented).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    # Bộ đếm chi tiết theo lớp chặn (input_guardrail, rate_limiter, …)
    blocked_by_layer: dict = field(default_factory=dict)
    redacted_responses: int = 0
    errors: int = 0

    def record(
        self,
        *,
        blocked: bool,
        layer: str | None = None,
        redacted: bool = False,
        error: bool = False,
    ) -> None:
        """Cập nhật counters sau mỗi request (gọi từ pipeline)."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
            key = layer or "unknown"
            self.blocked_by_layer[key] = self.blocked_by_layer.get(key, 0) + 1
        if layer == "rate_limiter":
            self.rate_limit_hits += 1
        if redacted:
            self.redacted_responses += 1
        if error:
            self.errors += 1

    def record_judge(self, *, passed: bool) -> None:
        self.judge_checks += 1
        if not passed:
            self.judge_fails += 1

    def check_metrics(self, *, verbose: bool = True) -> list[Alert]:
        """Compute rates; (re)build the alert list from the current counters."""
        alerts: list[Alert] = []
        if self.total_requests:
            block_rate = self.blocked_requests / self.total_requests
            if block_rate > self.block_rate_threshold:
                alerts.append(
                    Alert(
                        metric="block_rate",
                        value=round(block_rate, 4),
                        threshold=self.block_rate_threshold,
                        message=(
                            f"Block rate {block_rate:.0%} > {self.block_rate_threshold:.0%}: "
                            "possible attack campaign or over-blocking false positives."
                        ),
                    )
                )
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            alerts.append(
                Alert(
                    metric="rate_limit_hits",
                    value=float(self.rate_limit_hits),
                    threshold=float(self.rate_limit_hit_threshold),
                    message=(
                        f"{self.rate_limit_hits} rate-limit hits (≥ {self.rate_limit_hit_threshold}): "
                        "a client may be flooding the assistant."
                    ),
                )
            )
        if self.judge_checks:
            judge_fail_rate = self.judge_fails / self.judge_checks
            if judge_fail_rate > self.judge_fail_rate_threshold:
                alerts.append(
                    Alert(
                        metric="judge_fail_rate",
                        value=round(judge_fail_rate, 4),
                        threshold=self.judge_fail_rate_threshold,
                        message=(
                            f"Judge fail rate {judge_fail_rate:.0%} > "
                            f"{self.judge_fail_rate_threshold:.0%}: model output quality/safety degraded."
                        ),
                    )
                )
        self.alerts = alerts
        if verbose:
            for a in alerts:
                print(f"[ALERT] {a.metric}: {a.message}")
        return alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        self.check_metrics(verbose=False)  # làm mới alerts, không in lặp
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **self.snapshot(),
            "blocked_by_layer": dict(self.blocked_by_layer),
            "redacted_responses": self.redacted_responses,
            "errors": self.errors,
            "thresholds": {
                "block_rate": self.block_rate_threshold,
                "rate_limit_hits": self.rate_limit_hit_threshold,
                "judge_fail_rate": self.judge_fail_rate_threshold,
            },
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }

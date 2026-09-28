"""
Assignment 11 — Monitoring & Alerts.

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
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

    @property
    def rate_limited_requests(self) -> int:
        return self.rate_limit_hits

    @rate_limited_requests.setter
    def rate_limited_requests(self, val: int) -> None:
        self.rate_limit_hits = val

    def check_metrics(self) -> list[Alert]:
        """Compute rates, append Alert objects when thresholds exceeded."""
        new_alerts: list[Alert] = []
        if self.total_requests > 0:
            block_rate = self.blocked_requests / self.total_requests
            # Cảnh báo nếu block_rate vượt ngưỡng (mặc định > 30% hoặc block_rate_threshold)
            threshold = min(self.block_rate_threshold, 0.3)
            if block_rate > threshold:
                alert = Alert(
                    metric="block_rate",
                    value=round(block_rate, 4),
                    threshold=threshold,
                    message=f"High block rate detected: {block_rate:.1%} exceeds threshold {threshold:.1%}",
                )
                new_alerts.append(alert)

            rate_limit_rate = self.rate_limit_hits / self.total_requests
            if rate_limit_rate > 0.3 or self.rate_limit_hits >= self.rate_limit_hit_threshold:
                alert = Alert(
                    metric="rate_limit_requests",
                    value=round(rate_limit_rate, 4),
                    threshold=0.3,
                    message=f"High rate limit rate detected: {self.rate_limit_hits} requests ({rate_limit_rate:.1%})",
                )
                new_alerts.append(alert)

        if self.judge_checks > 0:
            judge_fail_rate = self.judge_fails / self.judge_checks
            if judge_fail_rate > self.judge_fail_rate_threshold:
                alert = Alert(
                    metric="judge_fail_rate",
                    value=round(judge_fail_rate, 4),
                    threshold=self.judge_fail_rate_threshold,
                    message=f"High judge failure rate: {judge_fail_rate:.1%} exceeds threshold {self.judge_fail_rate_threshold:.1%}",
                )
                new_alerts.append(alert)

        self.alerts.extend(new_alerts)
        return new_alerts

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        rate_limit_rate = (
            self.rate_limit_hits / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": round(block_rate, 4),
            "rate_limit_hits": self.rate_limit_hits,
            "rate_limited_requests": self.rate_limit_hits,
            "rate_limit_rate": round(rate_limit_rate, 4),
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": round(judge_fail_rate, 4),
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

    def export_json(self, filepath: str | None = None) -> str:
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        target = Path(filepath or default_metrics_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2, ensure_ascii=False)
        return str(target)


# Global default instance & module-level helper functions
default_monitor = MonitoringAlert()


def check_metrics() -> list[Alert]:
    return default_monitor.check_metrics()


def export_json(filepath: str | None = None) -> str:
    return default_monitor.export_json(filepath)

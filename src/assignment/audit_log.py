"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending_inputs: dict[str, dict] = {}

    def record_input(
        self,
        user_id: str = "anonymous",
        query: str | None = None,
        timestamp: float | str | None = None,
        *,
        text: str | None = None,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Store input + start timestamp keyed by request_id/user_id."""
        actual_text = text if text is not None else (query or kwargs.get("user_input") or "")
        req_id = request_id or kwargs.get("req_id") or f"{user_id}_{time.time()}_{len(self.logs)}"
        start_ts = timestamp if isinstance(timestamp, (int, float)) else time.time()
        
        self._open[req_id] = start_ts
        self._pending_inputs[req_id] = {
            "user_id": user_id,
            "query": actual_text,
            "timestamp": utc_now_iso() if not isinstance(timestamp, str) else timestamp,
            "start_time": start_ts,
        }
        return req_id

    def record_output(
        self,
        user_id: str = "anonymous",
        response: str | None = None,
        blocked: bool = False,
        layer: str | None = None,
        latency: float | None = None,
        *,
        text: str | None = None,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Store output, layer decision, latency; append to self.logs."""
        actual_response = text if text is not None else (response or kwargs.get("output") or "")
        req_id = request_id or kwargs.get("req_id")

        start_ts = self._open.pop(req_id, None) if req_id else None
        if latency is None:
            if start_ts is not None:
                latency = time.time() - start_ts
            else:
                latency = 0.0

        pending = self._pending_inputs.pop(req_id, {}) if req_id else {}
        query_text = pending.get("query", kwargs.get("input") or "")
        query_ts = pending.get("timestamp", utc_now_iso())

        entry = {
            "request_id": req_id or f"{user_id}_{len(self.logs)}",
            "user_id": user_id,
            "query": query_text,
            "response": actual_response,
            "blocked": bool(blocked),
            "layer": layer,
            "latency": round(latency, 4) if latency is not None else None,
            "latency_ms": round(latency * 1000, 2) if latency is not None else None,
            "timestamp": query_ts,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        target = Path(filepath or default_audit_log_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
        return str(target)


# Global default instance & module-level helper functions
default_audit_logger = AuditLogPlugin()


def record_input(*args, **kwargs) -> str:
    return default_audit_logger.record_input(*args, **kwargs)


def record_output(*args, **kwargs) -> dict:
    return default_audit_logger.record_output(*args, **kwargs)


def export_json(filepath: str | None = None) -> str:
    return default_audit_logger.export_json(filepath)

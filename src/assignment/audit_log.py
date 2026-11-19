"""
Assignment 11 — Audit Log (implemented).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

        self._pending: dict[str, dict] = {}
        self._seq = 0

    def _key(self, user_id: str, request_id: str | None) -> str:
        return request_id or f"user:{user_id}"

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id (fallback: user_id).

        Returns the request_id actually used so the caller can pair the output.
        """
        self._seq += 1
        rid = request_id or f"req-{self._seq:05d}"
        key = self._key(user_id, rid)
        self._open[key] = time.perf_counter()
        self._pending[key] = {
            "request_id": rid,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": text,
        }
        return rid

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        **extra,
    ):
        """Store output, layer decision, latency; append the full entry to self.logs."""
        key = self._key(user_id, request_id)
        started = self._open.pop(key, None)
        entry = self._pending.pop(key, None) or {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": None,
        }
        entry.update(
            {
                "output": text,
                "blocked": bool(blocked),
                "layer": layer,
                "latency_ms": (
                    round((time.perf_counter() - started) * 1000, 1)
                    if started is not None
                    else None
                ),
                "completed_at": utc_now_iso(),
            }
        )
        # Thông tin phụ (lý do chặn, issues redaction, lỗi provider…) nếu có
        entry.update({k: v for k, v in extra.items() if v is not None})
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

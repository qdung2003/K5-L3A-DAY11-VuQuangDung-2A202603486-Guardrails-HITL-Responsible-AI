"""
Assignment 11 — Rate Limiter (implemented).

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0
        self.last_blocked = False

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        window = self.user_windows[user_id]

        # 1. Bỏ các timestamp đã ra khỏi cửa sổ trượt
        cutoff = now - self.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()

        # 2. Đã đủ quota trong cửa sổ → chặn, KHÔNG ghi timestamp (không gọi LLM)
        if len(window) >= self.max_requests:
            wait = max(0.0, self.window_seconds - (now - window[0]))
            self.blocked_count += 1
            self.last_blocked = True
            return self._block_response(
                f"Rate limit exceeded: max {self.max_requests} requests per "
                f"{self.window_seconds}s. Try again in {wait:.0f}s."
            )

        # 3. Còn quota → ghi nhận và cho qua
        window.append(now)
        self.last_blocked = False
        return None

    def reset(self, user_id: str | None = None) -> None:
        """Xoá cửa sổ của một user (hoặc tất cả) — dùng giữa các nhóm test."""
        if user_id is None:
            self.user_windows.clear()
        else:
            self.user_windows.pop(user_id, None)

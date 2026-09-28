"""
Assignment 11 — Rate Limiter.

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque[float]] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    def is_rate_limited(self, user_id: str = "anonymous", now: float | None = None) -> tuple[bool, float]:
        """Check if user_id is rate limited. Returns (is_blocked, wait_seconds)."""
        if now is None:
            now = time.time()
        window = self.user_windows[user_id]
        while window and window[0] <= now - self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            return True, max(wait, 0.0)

        window.append(now)
        return False, 0.0

    def check_rate_limit(self, user_id: str = "anonymous") -> str | None:
        """Helper to check rate limit directly. Returns block message or None."""
        blocked, wait = self.is_rate_limited(user_id)
        if blocked:
            self.blocked_count += 1
            return f"Rate limit exceeded. Please try again later. Try again in {wait:.0f}s."
        return None

    def process(self, user_id: str = "anonymous") -> str | None:
        """Process direct user_id check."""
        self.total_count += 1
        return self.check_rate_limit(user_id)

    def __call__(self, user_id: str = "anonymous") -> str | None:
        return self.process(user_id)

    async def on_user_message_callback(
        self,
        *,
        invocation_context: Any = None,
        user_message: Any = None,
    ) -> types.Content | None:
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        window = self.user_windows[user_id]

        while window and window[0] <= now - self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            self.blocked_count += 1
            return self._block_response(
                f"Rate limit exceeded. Please try again later. Try again in {wait:.0f}s."
            )

        window.append(now)
        return None

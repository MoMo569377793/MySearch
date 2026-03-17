from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import TextIO


def _clean_detail(detail: str, limit: int = 72) -> str:
    text = " ".join(str(detail).split()).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


@dataclass(slots=True)
class ProgressBar:
    label: str
    total: int
    enabled: bool = True
    width: int = 28
    stream: TextIO | None = None
    current: int = 0
    detail: str = ""
    _last_render_width: int = 0
    _spinner_index: int = 0
    _pulse_stop: threading.Event | None = None
    _pulse_thread: threading.Thread | None = None
    _lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        self.stream = self.stream or sys.stderr
        self.total = max(int(self.total), 1)
        self.current = 0
        self.detail = ""
        self._last_render_width = 0
        self._spinner_index = 0
        self._pulse_stop = None
        self._pulse_thread = None
        self._lock = threading.Lock()
        self.enabled = bool(self.enabled and getattr(self.stream, "isatty", lambda: False)())
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()
            self._render()

    def set_detail(self, detail: str) -> None:
        if not self.enabled:
            return
        self.detail = _clean_detail(detail)
        self._render()

    def advance(self, step: int = 1, detail: str | None = None) -> None:
        if not self.enabled:
            return
        self.current = min(self.total, self.current + max(step, 0))
        if detail is not None:
            self.detail = _clean_detail(detail)
        self._render()

    def close(self, detail: str | None = None) -> None:
        if not self.enabled:
            return
        self.stop_pulse()
        self.current = self.total
        if detail is not None:
            self.detail = _clean_detail(detail)
        self._render()
        self.stream.write("\n")
        self.stream.flush()

    def start_pulse(self, detail: str | None = None) -> None:
        if not self.enabled:
            return
        self.stop_pulse()
        if detail is not None:
            self.detail = _clean_detail(detail)
        stop_event = threading.Event()
        self._pulse_stop = stop_event

        def _runner() -> None:
            while not stop_event.wait(0.2):
                self._render(spin_only=True)

        self._pulse_thread = threading.Thread(target=_runner, daemon=True)
        self._pulse_thread.start()
        self._render(spin_only=True)

    def stop_pulse(self, detail: str | None = None) -> None:
        if not self.enabled:
            return
        if detail is not None:
            self.detail = _clean_detail(detail)
        stop_event = self._pulse_stop
        thread = self._pulse_thread
        self._pulse_stop = None
        self._pulse_thread = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)
        self._render()

    def _render(self, *, spin_only: bool = False) -> None:
        lock = self._lock
        if lock is None:
            return
        with lock:
            self._spinner_index = (self._spinner_index + 1) % 4
            spinner = "|/-\\"[self._spinner_index]
            ratio = self.current / self.total if self.total else 1.0
            filled = int(self.width * ratio)
            bar = "#" * filled + "-" * (self.width - filled)
            prefix = f"{self.label} [{bar}] {self.current}/{self.total}"
            detail = f"{spinner} {self.detail}".rstrip() if self.detail else spinner
            text = f"{prefix} {detail}".rstrip()
            padded = text.ljust(self._last_render_width)
            self.stream.write("\r" + padded)
            self.stream.flush()
            if not spin_only:
                self._last_render_width = max(self._last_render_width, len(text))

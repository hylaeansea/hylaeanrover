"""Subprocess registry: spawn, stream, stop.

Each run wraps one Popen. A daemon thread drains merged stdout/stderr
into an in-memory line buffer; readers (WebSocket or polling HTTP)
consume from a cursor, so there is no cross-thread queue wiring.
Metric extraction is pluggable: the project descriptor supplies a
line parser (e.g. the SB3 `ep_rew_mean` table regex) and the engine
just accumulates whatever it yields.
"""

from __future__ import annotations

import itertools
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

MAX_LINES = 20_000  # cap the per-run buffer; oldest lines are dropped

LineParser = Callable[[str], Optional[dict[str, Any]]]


@dataclass
class Run:
    id: str
    node_id: Optional[str]
    kind: str  # e.g. "training", "eval", "promote", "sim"
    argv: list[str]
    cwd: str
    proc: subprocess.Popen
    started_at: float
    lines: list[str] = field(default_factory=list)
    dropped: int = 0  # lines evicted from the front of the buffer
    metrics: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"  # running | done | failed | stopped
    returncode: Optional[int] = None
    _stop_requested: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "kind": self.kind,
            "argv": self.argv,
            "cwd": self.cwd,
            "status": self.status,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "n_lines": self.dropped + len(self.lines),
            "metrics": self.metrics[-500:],
        }


class Executor:
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def spawn(
        self,
        argv: list[str],
        cwd: str,
        kind: str,
        node_id: Optional[str] = None,
        line_parser: Optional[LineParser] = None,
        env: Optional[dict[str, str]] = None,
    ) -> Run:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, **(env or {})},
            # New process group so stop() can signal the whole tree
            # (train.py forks SubprocVecEnv workers).
            start_new_session=True,
        )
        run = Run(
            id=f"run{next(self._ids)}",
            node_id=node_id,
            kind=kind,
            argv=argv,
            cwd=cwd,
            proc=proc,
            started_at=time.time(),
        )
        with self._lock:
            self._runs[run.id] = run
        t = threading.Thread(
            target=self._reader, args=(run, line_parser), daemon=True
        )
        t.start()
        return run

    def _reader(self, run: Run, line_parser: Optional[LineParser]) -> None:
        assert run.proc.stdout is not None
        for raw in run.proc.stdout:
            line = raw.rstrip("\n")
            run.lines.append(line)
            if len(run.lines) > MAX_LINES:
                del run.lines[: MAX_LINES // 10]
                run.dropped += MAX_LINES // 10
            if line_parser:
                try:
                    m = line_parser(line)
                except Exception:
                    m = None
                if m:
                    run.metrics.append(m)
        run.returncode = run.proc.wait()
        if run._stop_requested:
            run.status = "stopped"
        else:
            run.status = "done" if run.returncode == 0 else "failed"

    def get(self, run_id: str) -> Optional[Run]:
        return self._runs.get(run_id)

    def list(self) -> list[dict[str, Any]]:
        return [r.snapshot() for r in self._runs.values()]

    def stop(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if run is None or run.status != "running":
            return False
        run._stop_requested = True
        try:
            os.killpg(os.getpgid(run.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        return True

    def lines_since(self, run_id: str, cursor: int) -> tuple[list[str], int, str]:
        """Return (new_lines, next_cursor, status). Cursor is an absolute
        line index so a slow reader that falls behind the eviction window
        just skips the dropped lines instead of erroring."""
        run = self._runs[run_id]
        start = max(cursor - run.dropped, 0)
        new = run.lines[start:]
        return new, run.dropped + len(run.lines), run.status

    def node_states(self) -> dict[str, dict[str, Any]]:
        """Latest run per node id (later spawns win)."""
        out: dict[str, dict[str, Any]] = {}
        for run in self._runs.values():
            if run.node_id is None:
                continue
            prev = out.get(run.node_id)
            if prev is None or run.started_at >= prev["started_at"]:
                last = run.metrics[-1] if run.metrics else None
                out[run.node_id] = {
                    "status": run.status,
                    "run_id": run.id,
                    "started_at": run.started_at,
                    "returncode": run.returncode,
                    "last_metric": last,
                }
        return out

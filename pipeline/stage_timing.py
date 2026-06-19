"""Per-stage timing for the top-level pipeline orchestrator."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

T = TypeVar("T")

_current_progress_ticker: ContextVar[Optional["StageProgressTicker"]] = ContextVar(
    "stage_progress_ticker",
    default=None,
)

DEFAULT_TIMING_LOG_NAME = "pipeline_stage_timings.log"
TIMING_LOGGER_NAME = "video2smpl.pipeline.stage_timing"


def format_duration(seconds: float) -> str:
    """Human-readable duration (e.g. 1h 02m 03.4s)."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


@dataclass
class StageTimingRecord:
    stage: str
    started_at: str
    ended_at: str
    duration_sec: float
    status: str  # completed | failed

    @property
    def duration_display(self) -> str:
        return format_duration(self.duration_sec)


@dataclass
class PipelineRunTiming:
    root_dir: str
    dataset: str
    stages_planned: List[str]
    started_at: str = field(default_factory=lambda: _now_iso())
    ended_at: str = ""
    total_duration_sec: float = 0.0
    records: List[StageTimingRecord] = field(default_factory=list)

    @property
    def total_display(self) -> str:
        return format_duration(self.total_duration_sec)


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def resolve_timing_log_path(root_dir: Path, explicit: Optional[str] = None) -> Path:
    if explicit and str(explicit).strip():
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = (root_dir / p).resolve()
        return p
    return (root_dir / "logs" / DEFAULT_TIMING_LOG_NAME).resolve()


def _configure_timing_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(TIMING_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class StageProgressTicker:
    """
    Background ticker while a stage is running.

    Prints / logs elapsed time and optional sample progress every ``interval_sec``.
    """

    def __init__(
        self,
        *,
        stage: str,
        dataset: str,
        logger: Optional[logging.Logger],
        interval_sec: float = 30.0,
    ) -> None:
        self.stage = stage
        self.dataset = dataset
        self._logger = logger
        self.interval_sec = max(0.0, float(interval_sec))
        self._t0 = time.perf_counter()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._done = 0
        self._total: Optional[int] = None
        self._item = ""
        self._note = ""

    def _snapshot(self) -> tuple[int, Optional[int], str, str, float]:
        with self._lock:
            return self._done, self._total, self._item, self._note, time.perf_counter() - self._t0

    def _progress_message(self) -> str:
        done, total, item, note, elapsed = self._snapshot()
        parts = [
            f"[pipeline-timing] STAGE TICK stage={self.stage} dataset={self.dataset}",
            f"elapsed={format_duration(elapsed)} ({elapsed:.1f}s)",
        ]
        if total is not None and total > 0:
            parts.append(f"progress={done}/{total}")
            if done > 0:
                rate = elapsed / done
                remaining = max(total - done, 0) * rate
                parts.append(f"eta={format_duration(remaining)}")
        if item:
            parts.append(f"item={item}")
        if note:
            parts.append(f"note={note}")
        return " ".join(parts)

    def _emit(self, message: str) -> None:
        if self._logger is not None:
            self._logger.info(message)
        if os.environ.get("VIDEO2SMPL_SUPPRESS_STAGE_TICKS", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            return
        print(message, flush=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._emit(self._progress_message())

    def start(self) -> None:
        if self.interval_sec <= 0:
            return
        self._thread = threading.Thread(
            target=self._loop,
            name=f"stage-tick-{self.stage}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def set_total(self, total: int) -> None:
        with self._lock:
            self._total = max(0, int(total))

    def update(
        self,
        *,
        done: Optional[int] = None,
        delta: int = 0,
        total: Optional[int] = None,
        item: str = "",
        note: str = "",
    ) -> None:
        with self._lock:
            if total is not None:
                self._total = max(0, int(total))
            if done is not None:
                self._done = max(0, int(done))
            elif delta:
                self._done = max(0, self._done + int(delta))
            if item:
                self._item = item
            if note:
                self._note = note

    def __enter__(self) -> "StageProgressTicker":
        _current_progress_ticker.set(self)
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
        _current_progress_ticker.set(None)


def get_stage_progress_ticker() -> Optional[StageProgressTicker]:
    return _current_progress_ticker.get()


def stage_progress_set_total(total: int) -> None:
    ticker = get_stage_progress_ticker()
    if ticker is not None:
        ticker.set_total(total)


def stage_progress_update(
    *,
    done: Optional[int] = None,
    delta: int = 0,
    total: Optional[int] = None,
    item: str = "",
    note: str = "",
) -> None:
    ticker = get_stage_progress_ticker()
    if ticker is not None:
        ticker.update(done=done, delta=delta, total=total, item=item, note=note)


class PipelineStageTimer:
    """Time each pipeline stage; echo to terminal and append to a log file."""

    def __init__(
        self,
        *,
        root_dir: Path,
        dataset: str,
        stages_planned: List[str],
        log_path: Optional[Path] = None,
        enabled: bool = True,
        tick_interval_sec: float = 30.0,
    ) -> None:
        self.root_dir = root_dir.resolve()
        self.dataset = dataset or "single-root"
        self.stages_planned = list(stages_planned)
        self.log_path = log_path or resolve_timing_log_path(self.root_dir)
        self.enabled = enabled
        self.tick_interval_sec = max(0.0, float(tick_interval_sec))
        self.run = PipelineRunTiming(
            root_dir=str(self.root_dir),
            dataset=self.dataset,
            stages_planned=self.stages_planned,
        )
        self._logger: Optional[logging.Logger] = None
        if self.enabled:
            self._logger = _configure_timing_logger(self.log_path)

    def _emit(self, message: str, *, also_print: bool = True) -> None:
        if self._logger is not None:
            self._logger.info(message)
        if also_print:
            print(message, flush=True)

    def begin_run(self) -> None:
        if not self.enabled:
            return
        msg = (
            f"[pipeline-timing] RUN START dataset={self.dataset} "
            f"root={self.root_dir} stages={','.join(self.stages_planned)} "
            f"log={self.log_path}"
        )
        self._emit(msg)

    def run_stage(self, stage_name: str, fn: Callable[[], T]) -> T:
        if not self.enabled:
            return fn()

        started = _now_iso()
        t0 = time.perf_counter()
        self._emit(
            f"[pipeline-timing] STAGE START stage={stage_name} dataset={self.dataset} "
            f"started_at={started}"
        )
        status = "completed"
        try:
            with StageProgressTicker(
                stage=stage_name,
                dataset=self.dataset,
                logger=self._logger,
                interval_sec=self.tick_interval_sec,
            ):
                return fn()
        except Exception:
            status = "failed"
            raise
        finally:
            duration = time.perf_counter() - t0
            ended = _now_iso()
            rec = StageTimingRecord(
                stage=stage_name,
                started_at=started,
                ended_at=ended,
                duration_sec=duration,
                status=status,
            )
            self.run.records.append(rec)
            self._emit(
                f"[pipeline-timing] STAGE END stage={stage_name} dataset={self.dataset} "
                f"status={status} duration={rec.duration_display} ({duration:.3f}s) "
                f"ended_at={ended}"
            )

    def end_run(self) -> None:
        if not self.enabled:
            return
        self.run.ended_at = _now_iso()
        if self.run.records:
            self.run.total_duration_sec = sum(r.duration_sec for r in self.run.records)
        self._print_summary()
        self._emit(
            f"[pipeline-timing] RUN END dataset={self.dataset} "
            f"total={self.run.total_display} ({self.run.total_duration_sec:.3f}s) "
            f"ended_at={self.run.ended_at}"
        )

    def _print_summary(self) -> None:
        lines = [
            "",
            f"=== Pipeline stage timing summary ({self.dataset}) ===",
            f"root: {self.root_dir}",
            f"log:  {self.log_path}",
        ]
        if not self.run.records:
            lines.append("(no stages timed)")
        else:
            name_w = max(len(r.stage) for r in self.run.records)
            for rec in self.run.records:
                lines.append(
                    f"  {rec.stage:<{name_w}}  {rec.duration_display:>12}  "
                    f"({rec.duration_sec:8.3f}s)  [{rec.status}]"
                )
            lines.append(f"  {'TOTAL':<{name_w}}  {self.run.total_display:>12}  "
                         f"({self.run.total_duration_sec:8.3f}s)")
        lines.append("=" * (len(lines[-1]) if lines else 40))
        text = "\n".join(lines)
        if self._logger is not None:
            for line in lines:
                if line.strip():
                    self._logger.info(line)
        print(text, flush=True)

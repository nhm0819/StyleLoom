"""Console rendering of core events.

The core emits `Event` objects and knows nothing about terminals. This is the
transport-side half of that seam: it satisfies `EventSink` structurally, without
the core importing anything from here.

An API would put an SSE writer in this slot; a worker would put a queue status
update. That is the whole reason the core does not print.
"""

from __future__ import annotations

import sys
import time

import typer
from styleloom_core.events import Event, EventKind

_STAGE_LABEL = {
    "ingest": "입력 정규화",
    "outline": "본문 비트 구성",
    "hook": "훅 생성",
    "storyboard": "샷 프롬프트 전개",
    "render": "샷 렌더링",
    "assemble": "자막·결합",
    "qc": "스타일 정합성 검사",
}


class ConsoleSink:
    """Prints one line per stage, with elapsed time."""

    def __init__(self, quiet: bool = False, show_run_id: bool = False) -> None:
        self.quiet = quiet
        self.show_run_id = show_run_id
        self._stage_started: float | None = None

    def _prefix(self, event: Event) -> str:
        return f"[{event.run_id}] " if self.show_run_id else ""

    def emit(self, event: Event) -> None:
        if self.quiet:
            return
        if event.kind is EventKind.RUN_STARTED:
            typer.secho(f"{self._prefix(event)}run {event.run_id}", fg="cyan")

        elif event.kind is EventKind.STAGE_STARTED:
            self._stage_started = time.monotonic()
            label = _STAGE_LABEL.get(event.stage, event.stage)
            typer.echo(f"  {event.stage:<11s} {label} ... ", nl=False)
            sys.stdout.flush()

        elif event.kind is EventKind.STAGE_FINISHED:
            took = time.monotonic() - (self._stage_started or time.monotonic())
            typer.secho(f"{took:.1f}s", fg="green")

        elif event.kind is EventKind.WARNING:
            typer.secho(f"\n  ! {event.stage}: {event.message}", fg="yellow")

        elif event.kind is EventKind.RUN_FAILED:
            typer.secho(f"\n  x failed at {event.stage}: {event.message}", fg="red")

        elif event.kind is EventKind.RUN_FINISHED:
            score = event.data.get("qc_score")
            score_text = f"  qc={score}" if score is not None else ""
            typer.secho(f'  -> "{event.message}"{score_text}', fg="bright_white", bold=True)
            video = event.data.get("final_video")
            if video:
                typer.echo(f"     {video}")

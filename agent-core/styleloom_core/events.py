"""Progress events.

The runner emits; a transport consumes. This is the seam that lets the same
core drive a CLI progress display, an API SSE stream, or a worker's queue
status updates without the core knowing which one is attached.

`EventSink` is a Protocol rather than a base class so a transport can satisfy it
without importing this module at all.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .schema import utcnow


class EventKind(str, Enum):
    RUN_STARTED = "run_started"
    STAGE_STARTED = "stage_started"
    STAGE_FINISHED = "stage_finished"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"
    WARNING = "warning"


class Event(BaseModel):
    kind: EventKind
    run_id: str
    stage: str = ""
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    at: str = Field(default_factory=utcnow)


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


class NullSink:
    """Default. Discards everything, so the core never requires a transport."""

    def emit(self, event: Event) -> None:  # noqa: D102
        return None


class ListSink:
    """Collects events in memory. Used by tests to assert on stage ordering."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [e.kind.value for e in self.events]

    def stages(self) -> list[str]:
        return [e.stage for e in self.events if e.kind is EventKind.STAGE_FINISHED]

"""Execution context: everything a tool needs that is not run-specific.

Settings, providers and stores are assembled once by the caller and passed down.
This is what replaced the module-level settings singleton: the core has no
global state, so a CLI process, a test, and later an API request handler can each
hold their own context without interfering.

Tools take `(ctx, session)`. Nothing in the core reaches for a global.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .config import Settings
from .events import Event, EventKind, EventSink, NullSink
from .memory import ChoiceHistory, RunStore, StyleStore
from .providers import BaseLLM, BaseVideoProvider, build_llm, build_video_provider


@dataclass
class Context:
    settings: Settings
    llm: BaseLLM
    video: BaseVideoProvider
    styles: StyleStore
    runs: RunStore
    history: ChoiceHistory
    events: EventSink = field(default_factory=NullSink)

    def emit(
        self,
        kind: EventKind,
        run_id: str,
        stage: str = "",
        message: str = "",
        **data: object,
    ) -> None:
        self.events.emit(
            Event(kind=kind, run_id=run_id, stage=stage, message=message, data=dict(data))
        )


def build_context(
    settings: Settings | None = None,
    events: EventSink | None = None,
    llm: BaseLLM | None = None,
    video: BaseVideoProvider | None = None,
) -> Context:
    """Assemble a context from settings.

    `llm` and `video` are injectable so a caller can substitute a provider
    without going through environment variables -- which is how the tests avoid
    both the network and the env-ordering problem the old singleton had.
    """
    settings = settings or Settings()
    settings.validate_runtime()
    settings.ensure_dirs()
    return Context(
        settings=settings,
        llm=llm or build_llm(settings),
        video=video or build_video_provider(settings),
        styles=StyleStore(settings),
        runs=RunStore(settings),
        history=ChoiceHistory(settings),
        events=events or NullSink(),
    )


def new_run_id(prefix: str = "run") -> str:
    """Time-ordered and collision-resistant.

    Time-ordered because RunStore lists runs by sorting directory names, and
    random-suffixed because a batch can start several runs inside one second.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_{stamp}_{secrets.token_hex(3)}"

"""StyleLoom core.

Decompose a reference short-form video into a reusable style schema, then
generate new videos in that style from arbitrary inputs.

A framework-independent library: no HTTP framework, no CLI framework, no global
state. Callers build a `Context`, then call `run_once`. Everything a transport
needs is re-exported here.
"""

from .config import Settings
from .context import Context, build_context, new_run_id
from .errors import (
    ConfigError,
    LLMError,
    MediaError,
    NotFoundError,
    PlanError,
    ProviderError,
    StyleLoomError,
    ToolError,
    VideoProviderError,
)
from .events import Event, EventKind, EventSink, ListSink, NullSink
from .planner import Plan, build_plan
from .runner import (
    default_plan,
    execute,
    prepare_session,
    resume_run,
    resume_session,
    run_batch,
    run_once,
)
from .schema import (
    Brief,
    Casting,
    HookResult,
    Keyframes,
    Outline,
    QCReport,
    RunInputs,
    RunRecord,
    RunStatus,
    Storyboard,
    StyleSchema,
)
from .session import RunSession
from .tools import extract_style

__version__ = "0.2.0"

__all__ = [
    "Brief",
    "Casting",
    "ConfigError",
    "Context",
    "Event",
    "EventKind",
    "EventSink",
    "HookResult",
    "Keyframes",
    "LLMError",
    "ListSink",
    "MediaError",
    "NotFoundError",
    "NullSink",
    "Outline",
    "Plan",
    "PlanError",
    "ProviderError",
    "QCReport",
    "RunInputs",
    "RunRecord",
    "RunSession",
    "RunStatus",
    "Settings",
    "Storyboard",
    "StyleLoomError",
    "StyleSchema",
    "ToolError",
    "VideoProviderError",
    "__version__",
    "build_context",
    "build_plan",
    "default_plan",
    "execute",
    "extract_style",
    "new_run_id",
    "prepare_session",
    "resume_run",
    "resume_session",
    "run_batch",
    "run_once",
]

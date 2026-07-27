"""Tool registry.

A tool is a function `(Context, RunSession) -> BaseModel | None` that declares
which artifact keys it reads and which one it writes. The declaration is the
point: because dependencies are data rather than import statements, the planner
can check a plan before anything expensive runs, and a reordering mistake fails
at plan time with a readable message instead of at render time with a KeyError.

Tools do not call other tools. Shared operations live in `styleloom_core.media`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..errors import PlanError

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from ..context import Context
    from ..session import RunSession

ToolFn = Callable[["Context", "RunSession"], BaseModel | None]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    reads: tuple[str, ...]
    writes: str | None
    fn: ToolFn
    description: str = ""

    def __call__(self, ctx: Context, session: RunSession) -> BaseModel | None:
        return self.fn(ctx, session)


REGISTRY: dict[str, ToolSpec] = {}


def tool(
    name: str,
    reads: tuple[str, ...] = (),
    writes: str | None = None,
) -> Callable[[ToolFn], ToolFn]:
    """Register a tool under `name`.

    Returns the undecorated function so tools stay directly callable and
    unit-testable without going through the registry.
    """

    def decorate(fn: ToolFn) -> ToolFn:
        if name in REGISTRY:
            raise PlanError(f"tool {name!r} is already registered")
        REGISTRY[name] = ToolSpec(
            name=name,
            reads=reads,
            writes=writes,
            fn=fn,
            description=(fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "",
        )
        return fn

    return decorate


def get(name: str) -> ToolSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise PlanError(
            f"unknown tool {name!r}. Registered: {sorted(REGISTRY)}"
        ) from None


def names() -> list[str]:
    return sorted(REGISTRY)

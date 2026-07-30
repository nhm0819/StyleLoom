"""Tools: one per pipeline stage.

Importing this package registers every pipeline tool, which is what makes the
registry populated by the time a planner asks for it. `analyze` is imported for
its `extract_style` entry point but is deliberately not registered -- it produces
the reusable style asset rather than a run artifact.
"""

from . import (
    analyze,
    assemble,
    casting,
    hook,
    ingest,
    keyframe,
    outline,
    qc,
    render,
    storyboard,
)
from .analyze import extract_style
from .registry import REGISTRY, ToolSpec, get, names, tool

__all__ = [
    "REGISTRY",
    "ToolSpec",
    "analyze",
    "assemble",
    "casting",
    "extract_style",
    "get",
    "hook",
    "ingest",
    "keyframe",
    "names",
    "outline",
    "qc",
    "render",
    "storyboard",
    "tool",
]

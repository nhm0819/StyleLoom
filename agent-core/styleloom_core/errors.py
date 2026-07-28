"""Error taxonomy.

One base class so transports can map the whole library to a status code without
knowing every failure mode, and specific subclasses so a caller that *does* care
can branch. Nothing here inherits from anything framework-specific.
"""

from __future__ import annotations


class StyleLoomError(RuntimeError):
    """Base for every error this library raises deliberately."""


class ConfigError(StyleLoomError):
    """Settings are internally inconsistent or a required key is missing."""


class NotFoundError(StyleLoomError):
    """A style, run or input file was requested that does not exist on disk."""


class ProviderError(StyleLoomError):
    """An LLM or video provider call failed."""


class LLMError(ProviderError):
    pass


class VideoProviderError(ProviderError):
    pass


class MediaError(StyleLoomError):
    """ffmpeg / OpenCV operation failed."""


class PlanError(StyleLoomError):
    """A plan is malformed: unknown tool, or a step reads an artifact that no
    earlier step writes."""


class ToolError(StyleLoomError):
    """A tool could not produce its artifact from valid inputs."""

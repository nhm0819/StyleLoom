"""Provider adapters: the only part of the core that talks to the outside world.

Tools depend on providers; providers know nothing about tools, sessions or
plans. That direction is why this is a sibling of `tools/` rather than a
subpackage of it.
"""

from .base import BaseLLM, BaseVideoProvider, MotionShot
from .kling import KlingVideoProvider, load_kling_specs
from .llm import AnthropicLLM, MockLLM, build_llm, extract_json
from .video import MockVideoProvider, build_video_provider

__all__ = [
    "BaseLLM",
    "BaseVideoProvider",
    "MotionShot",
    "AnthropicLLM",
    "MockLLM",
    "build_llm",
    "extract_json",
    "KlingVideoProvider",
    "MockVideoProvider",
    "build_video_provider",
    "load_kling_specs",
]

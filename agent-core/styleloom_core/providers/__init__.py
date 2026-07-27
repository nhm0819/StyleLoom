"""Provider adapters: the only part of the core that talks to the outside world.

Tools depend on providers; providers know nothing about tools, sessions or
plans. That direction is why this is a sibling of `tools/` rather than a
subpackage of it.
"""

from .base import BaseLLM, BaseVideoProvider, MotionShot
from .llm import AnthropicLLM, MockLLM, build_llm, extract_json
from .video import FalVideoProvider, MockVideoProvider, build_video_provider, load_fal_specs

__all__ = [
    "BaseLLM",
    "BaseVideoProvider",
    "MotionShot",
    "AnthropicLLM",
    "MockLLM",
    "build_llm",
    "extract_json",
    "FalVideoProvider",
    "MockVideoProvider",
    "build_video_provider",
    "load_fal_specs",
]

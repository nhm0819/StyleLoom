"""Memory: what persists between runs.

Three kinds, separated because they have different lifetimes:
  * StyleStore     - long-term. The reusable style asset.
  * RunStore       - per-run artifacts. The inspectable deliverable.
  * ChoiceHistory  - episodic. Recent pool selections, read by hook and casting.
"""

from .history import ChoiceHistory
from .run_store import RunStore
from .style_store import StyleStore

__all__ = ["ChoiceHistory", "RunStore", "StyleStore"]

"""Plans: which tools run, in what order.

The plan is a static DAG, chosen over an LLM planner deliberately. The claim this
system makes is that the *same* pipeline turns different inputs into videos that
read as one channel. If the model reorders or drops stages between runs, output
differences stop being attributable to the input, and the reproducibility claim
is gone. Non-determinism is confined to where it is a requirement -- hook content
-- rather than sprayed across control flow.

What makes this more than a hardcoded call chain is that the order is data and
gets checked. Each tool declares the artifacts it reads and writes, so a plan can
be validated before any expensive work starts: a misordered step fails with a
readable message instead of a KeyError halfway through rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import PlanError
from ..tools import registry

# Artifacts present before any tool runs. `style` is loaded from the store by the
# runner rather than produced by a tool, so plan validation has to know about it.
PRELOADED: frozenset[str] = frozenset({"style"})


@dataclass(frozen=True)
class Plan:
    name: str
    steps: tuple[str, ...]

    def validate(self, available: frozenset[str] = PRELOADED) -> None:
        """Raise PlanError unless every step's inputs exist when it runs."""
        if not self.steps:
            raise PlanError("plan has no steps")

        have = set(available)
        seen: set[str] = set()
        for position, name in enumerate(self.steps):
            if name in seen:
                raise PlanError(f"plan {self.name!r}: tool {name!r} appears twice")
            seen.add(name)

            spec = registry.get(name)  # raises PlanError on an unknown tool
            missing = [key for key in spec.reads if key not in have]
            if missing:
                raise PlanError(
                    f"plan {self.name!r}: step {position} ({name!r}) reads "
                    f"{missing} which nothing earlier writes. "
                    f"Available at that point: {sorted(have)}"
                )
            if spec.writes:
                have.add(spec.writes)

    def describe(self) -> list[str]:
        lines = []
        for i, name in enumerate(self.steps, start=1):
            spec = registry.get(name)
            reads = ", ".join(spec.reads) or "-"
            lines.append(f"{i}. {name}  ({reads} -> {spec.writes or '-'})")
        return lines


# The pipeline. Ordering notes worth keeping:
#   * outline before hook, so the hook can promise the actual payoff instead of
#     being a generic attention grabber.
#   * casting before storyboard, since every shot prompt carries the presenter and
#     the location; and after ingest, so a future casting rule can read the brief.
#   * qc last and separable, because it re-probes the finished file.
STANDARD_STEPS = (
    "ingest",
    "casting",
    "outline",
    "hook",
    "storyboard",
    "render",
    "assemble",
    "qc",
)


def build_plan(include_qc: bool = True) -> Plan:
    steps = STANDARD_STEPS if include_qc else tuple(
        s for s in STANDARD_STEPS if s != "qc"
    )
    plan = Plan(name="standard" if include_qc else "standard_no_qc", steps=steps)
    plan.validate()
    return plan

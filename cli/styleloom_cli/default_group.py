"""A command group whose first argument may be data rather than a subcommand.

`styleloom run <style_id>` and `styleloom run resume <run_id>` cannot both work
under a plain Typer group. A group parses its own positional arguments first and
only then looks for a subcommand, so declaring `style_id` on the group consumes the
word `resume` as a style id and the subcommand becomes unreachable -- Click reports
"No such command" for whatever followed it.

This resolves the first token itself: a registered subcommand name dispatches to
that subcommand, anything else falls through to `default_cmd` with the token left
in place as its argument.

The cost is one ambiguity, and it is worth stating plainly: a style whose id is
literally `resume` cannot be run as `styleloom run resume`. Every other style id is
unaffected, and that one is still reachable through the hidden command name.

Membership is checked up front rather than by catching the error Click raises for
an unknown command. Typer vendors its own copy of Click, so the exception raised
here is `typer._click.exceptions.UsageError` and not `click.UsageError` -- catching
the wrong one silently disables the fallback, which is exactly the failure this
comment exists to prevent someone rediscovering.
"""

from __future__ import annotations

from typing import Any

from typer.core import TyperGroup


class DefaultCommandGroup(TyperGroup):
    """Group that treats an unrecognised first argument as the default command's."""

    #: Subcommand to fall through to. Set on the subclass or assigned after init.
    default_cmd: str = ""

    def resolve_command(self, ctx: Any, args: list[str]) -> Any:
        if (
            args
            and self.default_cmd
            # An option like `--help` belongs to the group itself, so it must reach
            # Click's own handling rather than being pushed onto the default command.
            and not args[0].startswith("-")
            and self.get_command(ctx, args[0]) is None
        ):
            fallback = self.get_command(ctx, self.default_cmd)
            if fallback is not None:
                # `args` unchanged, not `args[1:]`: the token was never a command
                # name, so it is still the default command's first argument.
                return fallback.name, fallback, args
        return super().resolve_command(ctx, args)

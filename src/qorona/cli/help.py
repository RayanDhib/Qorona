"""Two-level ``--help`` for the ``qorona`` CLI.

Every command gets two help views, both computed from the live option set so neither can go
stale: ``-h`` / ``--help`` lists only the common options, flat, ending in a one-line pointer
that counts the remaining options and names their sections; ``--help-all`` lists every option,
grouped under titled sections in pipeline order. Each option carries its ``section`` and its
tier (common or advanced, :class:`QoronaOption`); a command with no advanced options prints no
pointer and behaves as stock click.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click

#: Section display order in the ``--help-all`` view: the pipeline order, the same vocabulary as
#: the config dataclasses and the end-of-run summary panel.
SECTION_ORDER = (
    "Input",
    "Field grid",
    "Volume",
    "Cache",
    "Camera",
    "Render",
    "Q-map",
    "Field lines",
    "Export",
    "Brightness",
    "Output",
    "Execution",
)

#: Context key set by the ``--help-all`` callback so ``format_options`` renders the full view.
_HELP_ALL_KEY = "qorona_help_all"


class QoronaOption(click.Option):
    """A click option that knows its help section and tier (common or advanced)."""

    def __init__(
        self, *args: Any, section: str | None = None, advanced: bool = False, **kwargs: Any
    ) -> None:
        self.section = section
        self.advanced = advanced
        super().__init__(*args, **kwargs)


def option(
    *param_decls: str,
    section: str | None = None,
    advanced: bool = False,
    **attrs: Any,
) -> Callable[[Callable], Callable]:
    """``click.option`` carrying the section/tier metadata (``cls=QoronaOption``)."""
    return click.option(*param_decls, cls=QoronaOption, section=section, advanced=advanced, **attrs)


def _show_help_all(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Eager ``--help-all`` callback: flag the context and print the full sectioned help."""
    if value and not ctx.resilient_parsing:
        ctx.meta[_HELP_ALL_KEY] = True
        click.echo(ctx.get_help(), color=ctx.color)
        ctx.exit()


def _section_key(section: str) -> int:
    """Sort key placing sections in pipeline order (unknown names last, defensively)."""
    try:
        return SECTION_ORDER.index(section)
    except ValueError:
        return len(SECTION_ORDER)


class QoronaCommand(click.Command):
    """A command with two help levels: common options by default, everything on ``--help-all``."""

    def get_params(self, ctx: click.Context) -> list[click.Parameter]:
        """Inject the eager ``--help-all`` option just before click's own help option."""
        params = super().get_params(ctx)
        if not any(param.name == "help_all" for param in params):
            help_all = click.Option(
                ["--help-all"],
                is_flag=True,
                expose_value=False,
                is_eager=True,
                callback=_show_help_all,
                help="Show every option, grouped.",
            )
            params.insert(len(params) - 1 if params else 0, help_all)
        return params

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        options = [param for param in self.get_params(ctx) if isinstance(param, click.Option)]
        if ctx.meta.get(_HELP_ALL_KEY):
            self._format_all(ctx, formatter, options)
        else:
            self._format_curated(ctx, formatter, options)

    def _format_curated(
        self, ctx: click.Context, formatter: click.HelpFormatter, options: list[click.Option]
    ) -> None:
        """The default view: the common options flat, then the pointer to ``--help-all``."""
        records = []
        advanced_sections: list[str] = []
        n_advanced = 0
        for opt in options:
            record = opt.get_help_record(ctx)
            if record is None:
                continue
            if getattr(opt, "advanced", False):
                n_advanced += 1
                section = getattr(opt, "section", None)
                if section is not None and section not in advanced_sections:
                    advanced_sections.append(section)
                continue
            records.append(record)
        if records:
            with formatter.section("Options"):
                formatter.write_dl(records)
        if n_advanced:
            names = ", ".join(sorted(advanced_sections, key=_section_key))
            formatter.write_paragraph()
            with formatter.indentation():
                formatter.write_text(
                    f"{n_advanced} more options in '{ctx.command_path} --help-all': {names}."
                )

    def _format_all(
        self, ctx: click.Context, formatter: click.HelpFormatter, options: list[click.Option]
    ) -> None:
        """The full view: a leading untitled block, one titled block per section, then Help."""
        lead = []
        by_section: dict[str, list[tuple[str, str]]] = {}
        help_block = []
        for opt in options:
            record = opt.get_help_record(ctx)
            if record is None:
                continue
            section = getattr(opt, "section", None)
            if opt.name in ("help", "help_all"):
                help_block.append(record)
            elif section is None:
                lead.append(record)
            else:
                by_section.setdefault(section, []).append(record)
        if lead:
            with formatter.section("Options"):
                formatter.write_dl(lead)
        for section in sorted(by_section, key=_section_key):
            with formatter.section(section):
                formatter.write_dl(by_section[section])
        if help_block:
            with formatter.section("Help"):
                formatter.write_dl(help_block)


class QoronaGroup(click.Group):
    """The command group whose subcommands all render the tiered help."""

    command_class = QoronaCommand

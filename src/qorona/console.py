"""Shared Rich console for friendly, consistent CLI output and progress.

All user-facing status, progress, and messages route through the single
``console`` instance defined here so styling stays uniform across the tool.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

console = Console()


@contextmanager
def status(message: str, *, enabled: bool = True) -> Iterator[None]:
    """Show an animated spinner while a slow operation runs.

    Parameters
    ----------
    message
        Text shown next to the spinner.
    enabled
        When ``False`` the spinner is suppressed (e.g. for quiet/library use)
        while the wrapped code still runs.
    """
    if not enabled:
        yield
        return
    with console.status(f"[bold cyan]{message}", spinner="dots"):
        yield


@contextmanager
def progress_bar(
    description: str, total: int, *, enabled: bool = True
) -> Iterator[Callable[[int], None]]:
    """Show a determinate progress bar, yielding a callable to set the completed count.

    For the long-running stages (tracing, volume painting) where the total amount of work is known
    up front. The yielded function takes the absolute number of units completed so far.

    Parameters
    ----------
    description
        Label shown beside the bar.
    total
        Total number of work units.
    enabled
        When ``False`` the bar is suppressed (quiet/library use) and the yielded callable is a
        no-op, while the wrapped code still runs.
    """
    if not enabled:
        yield lambda _completed: None
        return
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=total)

        def update(completed: int) -> None:
            progress.update(task, completed=completed)

        yield update


def print_step(message: str) -> None:
    """Print a progress step."""
    console.print(f"[bold blue]→[/bold blue] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]![/yellow] {message}")

"""Rich progress bars for long-running loops."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@contextmanager
def progress_bar(
    description: str,
    total: int,
    *,
    show: bool,
) -> Iterator[Callable[..., None]]:
    """Yield a callable advancing a progress bar over a loop.

    The columns match the ones
    [`joblib_progress`](https://github.com/jonghwanhyeon/joblib-progress) renders, so
    loops driven by hand look the same as the joblib-backed ones. Reach for
    `joblib_progress` itself when the loop *is* a `joblib.Parallel` call; use this
    when it is not, since `joblib_progress` advances its bar by patching
    `joblib.parallel.Parallel.print_progress` process-wide and never ticks without a
    `Parallel` to patch.

    Parameters
    ----------
    description : str
        Text shown next to the bar, e.g. `"Fitting runs..."`.
    total : int
        Number of steps the bar counts up to.
    show : bool
        Whether to render the bar. When `False`, the yielded callable does nothing,
        so callers need no second code path.

    Yields
    ------
    Callable
        Advances the bar by `amount` steps (default `1`), or does nothing when
        `show` is `False`.

    Examples
    --------
    >>> with progress_bar("Fitting runs...", total=2, show=False) as advance:
    ...     for _ in range(2):
    ...         advance()
    """
    if not show:
        yield lambda amount=1: None
        return

    progress = Progress(
        SpinnerColumn(),
        TaskProgressColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
    )
    task_id = progress.add_task(f"[cyan]{description}", total=total)
    with progress:
        yield lambda amount=1: progress.advance(task_id, amount)

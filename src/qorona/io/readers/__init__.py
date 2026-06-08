"""Reader registry and the ``read_solution`` entry point.

Readers register themselves here. ``read_solution`` selects one by explicit model
name or by inferring it from the file extension, then reads the file into a
:class:`~qorona.io.native.NativeSolution`.
"""

from __future__ import annotations

from pathlib import Path

from qorona.io.native import NativeSolution
from qorona.io.readers.base import SolutionReader
from qorona.io.readers.cfmesh import CFmeshReader

_READERS: list[type[SolutionReader]] = [CFmeshReader]


def register_reader(reader: type[SolutionReader]) -> type[SolutionReader]:
    """Register a reader so ``read_solution`` can dispatch to it."""
    _READERS.append(reader)
    return reader


def available_models() -> dict[str, tuple[str, ...]]:
    """Map each registered model to the file extensions it reads."""
    return {reader.model: reader.extensions for reader in _READERS}


def read_solution(
    path: str | Path,
    *,
    model: str | None = None,
    show_progress: bool = True,
    **reader_kwargs: object,
) -> NativeSolution:
    """Read a coronal MHD solution into a :class:`NativeSolution`.

    Parameters
    ----------
    path
        Path to the solution file.
    model
        Model identifier (e.g. ``"coconut"``). If omitted, it is inferred from the
        file extension.
    show_progress
        Whether to display reading progress.
    **reader_kwargs
        Forwarded to the selected reader's constructor.

    Returns
    -------
    NativeSolution
        The solution on its native mesh.
    """
    path = Path(path)

    if model is not None:
        reader_cls = next((r for r in _READERS if r.model == model.lower()), None)
        if reader_cls is None:
            raise ValueError(
                f"No reader for model {model!r}. Available: {sorted(available_models())}"
            )
    else:
        reader_cls = next((r for r in _READERS if r.handles(path)), None)
        if reader_cls is None:
            raise ValueError(
                f"Cannot infer model from extension {path.suffix!r}; pass `model=`. "
                f"Available: {sorted(available_models())}"
            )

    return reader_cls(**reader_kwargs).read(path, show_progress=show_progress)


__all__ = [
    "NativeSolution",
    "SolutionReader",
    "available_models",
    "read_solution",
    "register_reader",
]

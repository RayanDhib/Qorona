"""Reader registry and the ``read_solution`` entry point.

Readers register themselves here. ``read_solution`` selects one by resolving the file extension to
the readers that recognise it, narrowing by an explicit model when given, and — only when an
extension is shared by more than one model — by the readers' content check. A model may own several
formats (e.g. COCONUT's ``.CFmesh`` and ``.plt``); each is a separate reader sharing the model name.
"""

from __future__ import annotations

from pathlib import Path

from qorona.io.native import NativeSolution
from qorona.io.readers.base import SolutionReader
from qorona.io.readers.coconut import CFmeshReader, CoconutTecplotReader

_READERS: list[type[SolutionReader]] = [CFmeshReader, CoconutTecplotReader]


def register_reader(reader: type[SolutionReader]) -> type[SolutionReader]:
    """Register a reader so ``read_solution`` can dispatch to it."""
    _READERS.append(reader)
    return reader


def available_models() -> dict[str, tuple[str, ...]]:
    """Map each registered model to the file extensions it reads (across all its formats)."""
    models: dict[str, tuple[str, ...]] = {}
    for reader in _READERS:
        merged = models.get(reader.model, ()) + reader.extensions
        models[reader.model] = tuple(dict.fromkeys(merged))
    return models


def _disambiguate(candidates: list[type[SolutionReader]], path: Path) -> type[SolutionReader]:
    """Pick one reader from several sharing an extension, by their content check."""
    identified = [reader for reader in candidates if reader.identifies(path)]
    if len(identified) == 1:
        return identified[0]
    models = sorted({reader.model for reader in candidates})
    raise ValueError(
        f"Extension {path.suffix!r} is read by more than one model ({models}); pass `model=`."
    )


def _select_reader(path: Path, model: str | None) -> type[SolutionReader]:
    """Resolve a reader for ``path`` (optionally constrained to ``model``)."""
    if model is not None:
        model = model.lower()
        model_readers = [reader for reader in _READERS if reader.model == model]
        if not model_readers:
            raise ValueError(
                f"No reader for model {model!r}. Available: {sorted(available_models())}"
            )
        matches = [reader for reader in model_readers if reader.handles(path)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return _disambiguate(matches, path)
        if len(model_readers) == 1:
            return model_readers[0]
        formats = sorted(reader.file_format for reader in model_readers)
        raise ValueError(
            f"Model {model!r} has several formats ({formats}); extension {path.suffix!r} matches "
            f"none. Use the format's extension."
        )

    candidates = [reader for reader in _READERS if reader.handles(path)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return _disambiguate(candidates, path)
    raise ValueError(
        f"Cannot infer model from extension {path.suffix!r}; pass `model=`. "
        f"Available: {sorted(available_models())}"
    )


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
    reader_cls = _select_reader(path, model)
    return reader_cls(**reader_kwargs).read(path, show_progress=show_progress)


__all__ = [
    "NativeSolution",
    "SolutionReader",
    "available_models",
    "read_solution",
    "register_reader",
]

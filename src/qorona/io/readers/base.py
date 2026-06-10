"""Reader interface for reading MHD solutions.

A reader converts one model/format into a :class:`~qorona.io.native.NativeSolution`.
Supporting a new coronal model means adding one ``SolutionReader`` subclass; nothing
downstream changes.
"""

from __future__ import annotations

import gzip
import lzma
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TextIO

from qorona.io.native import NativeSolution

#: Compression suffixes a stored solution may carry. A large ASCII mesh shrinks markedly when
#: compressed, so a solution may be stored compressed; the payload format is recognised behind
#: the compression suffix and the file is opened through the matching standard-library decompressor,
#: so a ``.CFmesh.xz`` reads exactly like a plain ``.CFmesh`` with no extra dependency.
_COMPRESSION_SUFFIXES = (".xz", ".gz")


def payload_suffix(path: str | Path) -> str:
    """Return a path's format suffix, ignoring a trailing compression suffix.

    Both ``corona.CFmesh`` and ``corona.CFmesh.xz`` yield ``".CFmesh"``.
    """
    suffixes = Path(path).suffixes
    if suffixes and suffixes[-1] in _COMPRESSION_SUFFIXES:
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes else ""


def open_solution_text(path: str | Path) -> TextIO:
    """Open a solution file for line-oriented text reading, decompressing ``.xz`` / ``.gz``."""
    suffix = Path(path).suffix
    if suffix == ".xz":
        return lzma.open(path, "rt")
    if suffix == ".gz":
        return gzip.open(path, "rt")
    return Path(path).open()


class SolutionReader(ABC):
    """Base class for model-specific solution readers.

    Subclasses set the class attributes below and implement :meth:`read`.

    Attributes
    ----------
    model
        Short model identifier (e.g. ``"coconut"``).
    file_format
        Short format identifier (e.g. ``"cfmesh"``).
    extensions
        File extensions this reader recognises (including the leading dot).
    """

    model: str
    file_format: str
    extensions: tuple[str, ...]

    @abstractmethod
    def read(self, path: str | Path, *, show_progress: bool = True) -> NativeSolution:
        """Read a solution file into a :class:`NativeSolution`."""

    @classmethod
    def handles(cls, path: str | Path) -> bool:
        """Return whether this reader recognises ``path`` by its (payload) extension."""
        return payload_suffix(path) in cls.extensions

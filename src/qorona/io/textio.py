"""Shared text-I/O helpers for solution files.

A solution file may be stored compressed; the payload format is recognised behind the
compression suffix and the file is opened through the matching standard-library decompressor,
so a ``.CFmesh.xz`` or ``.plt.gz`` reads exactly like its plain form with no extra dependency.
These helpers live below both :mod:`qorona.io.formats` (container parsers) and
:mod:`qorona.io.readers` (model readers), so neither layer depends on the other.
"""

from __future__ import annotations

import gzip
import lzma
from pathlib import Path
from typing import TextIO

#: Compression suffixes a stored solution may carry.
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

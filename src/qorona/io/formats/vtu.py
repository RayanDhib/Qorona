"""Parser for the VTK XML unstructured-grid container (``.vtu``), appended-raw subset.

Model-agnostic container layer: returns pieces of raw points, connectivity, and cell
arrays keyed by the file's own names; a model reader in :mod:`qorona.io.readers` attaches
meaning. The supported subset is the one MPI-AMRVAC writes: little-endian, arrays with
``format="appended"``, one ``encoding="raw"`` payload with per-array byte-length
prefixes. Base64 or compressed payloads are rejected with a clear error.

Format reference: VTK File Formats, XML ``UnstructuredGrid``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qorona.console import progress_bar

__all__ = ["ParsedVtu", "VtuPiece", "parse_vtu"]

_DTYPES: dict[bytes, np.dtype[Any]] = {
    b"Float32": np.dtype("<f4"),
    b"Float64": np.dtype("<f8"),
    b"Int32": np.dtype("<i4"),
    b"Int64": np.dtype("<i8"),
    b"UInt32": np.dtype("<u4"),
    b"UInt64": np.dtype("<u8"),
}

_APPENDED = re.compile(rb'<AppendedData\s+encoding="(\w+)"\s*>')
_HEADER_TYPE = re.compile(rb'header_type="(\w+)"')
_BYTE_ORDER = re.compile(rb'byte_order="(\w+)"')
_PIECE = re.compile(
    rb'<Piece NumberOfPoints="\s*(\d+)" NumberOfCells="\s*(\d+)"\s*>(.*?)</Piece>', re.S
)
_ARRAY = re.compile(
    rb'<DataArray type="(\w+)"(?:\s+Name="([^"]*)")?(?:\s+NumberOfComponents="(\d+)")?'
    rb'\s+format="(\w+)"(?:\s+offset="(\d+)")?\s*/?>'
)
_FIELD_DATA = re.compile(rb"<FieldData>(.*?)</FieldData>", re.S)
_ASCII_SCALAR = re.compile(
    rb'<DataArray type="\w+" Name="([^"]*)" NumberOfTuples="1" format="ascii">'
    rb"\s*([-0-9.eE+]+)\s*</DataArray>"
)


@dataclass
class VtuPiece:
    """One ``<Piece>``: raw points, cell connectivity, and cell arrays by name.

    Attributes
    ----------
    points
        ``(n_points, 3)`` float64 node coordinates.
    connectivity
        ``(n_cells, nodes_per_cell)`` int64 0-based node indices.
    cell_data
        ``name -> (n_cells,)`` float64 arrays for the ``CellData`` variables.
    """

    points: np.ndarray
    connectivity: np.ndarray
    cell_data: dict[str, np.ndarray]


@dataclass
class ParsedVtu:
    """A parsed ``.vtu``: the pieces plus scalar ``FieldData`` values by name.

    Attributes
    ----------
    pieces
        The mesh blocks, each a :class:`VtuPiece`.
    field_data
        ``name -> value`` for the single-tuple ASCII ``FieldData`` scalars (e.g. ``TIME``).
    """

    pieces: list[VtuPiece]
    field_data: dict[str, float]


def _section(body: bytes, tag: bytes, path: Path) -> bytes:
    start = body.find(b"<" + tag + b">")
    end = body.find(b"</" + tag + b">")
    if start < 0 or end < 0:
        raise ValueError(f"Piece without a <{tag.decode()}> section in {path.name}.")
    return body[start:end]


def parse_vtu(
    path: object, *, cell_arrays: tuple[str, ...] | None = None, show_progress: bool = True
) -> ParsedVtu:
    """Parse an appended-raw ``.vtu`` into pieces of points, connectivity, and cell data.

    Parameters
    ----------
    path
        The ``.vtu`` file.
    cell_arrays
        When given, read only these cell arrays (skipping the rest of the payload).
    show_progress
        Whether to display per-piece reading progress.

    Returns
    -------
    ParsedVtu
        The parsed pieces and the file's scalar ``FieldData`` values.
    """
    path = Path(str(path))
    raw = path.read_bytes()
    appended = _APPENDED.search(raw)
    if appended is None:
        raise ValueError(
            f'No <AppendedData encoding="raw"> section in {path.name}; '
            f"only appended-raw VTU files are supported."
        )
    if appended.group(1) != b"raw":
        raise ValueError(
            f"Appended encoding {appended.group(1).decode()!r} in {path.name} is not "
            f'supported (only "raw"; base64 payloads are out of scope).'
        )
    header = raw[: appended.start()]
    if b"compressor=" in header:
        raise ValueError(f"Compressed VTU payloads are not supported ({path.name}).")
    order = _BYTE_ORDER.search(header)
    if order is not None and order.group(1) != b"LittleEndian":
        raise ValueError(f"Byte order {order.group(1).decode()!r} in {path.name} is not supported.")
    header_type = _HEADER_TYPE.search(header)
    prefix = _DTYPES[header_type.group(1) if header_type else b"UInt32"]
    payload = raw.index(b"_", appended.end()) + 1

    def read_array(match: re.Match[bytes]) -> np.ndarray:
        dtype = _DTYPES.get(match.group(1))
        if dtype is None or match.group(4) != b"appended" or match.group(5) is None:
            raise ValueError(
                f"Unsupported DataArray (type {match.group(1).decode()!r}, "
                f"format {match.group(4).decode()!r}) in {path.name}."
            )
        offset = payload + int(match.group(5))
        nbytes = int(np.frombuffer(raw, prefix, 1, offset)[0])
        return np.frombuffer(raw, dtype, nbytes // dtype.itemsize, offset + prefix.itemsize)

    field_data: dict[str, float] = {}
    field_block = _FIELD_DATA.search(header)
    if field_block is not None:
        for name, value in _ASCII_SCALAR.findall(field_block.group(1)):
            field_data[name.decode()] = float(value)

    piece_matches = list(_PIECE.finditer(header))
    if not piece_matches:
        raise ValueError(f"No <Piece> sections found in {path.name}.")
    pieces: list[VtuPiece] = []
    with progress_bar(f"Reading {path.name}", len(piece_matches), enabled=show_progress) as tick:
        for count, piece in enumerate(piece_matches, start=1):
            n_cells, body = int(piece.group(2)), piece.group(3)
            points = connectivity = offsets = None
            for match in _ARRAY.finditer(_section(body, b"Points", path)):
                points = read_array(match).astype(np.float64).reshape(-1, 3)
            for match in _ARRAY.finditer(_section(body, b"Cells", path)):
                name = (match.group(2) or b"").decode()
                if name == "connectivity":
                    connectivity = read_array(match).astype(np.int64)
                elif name == "offsets":
                    offsets = read_array(match).astype(np.int64)
            if points is None or connectivity is None or offsets is None:
                raise ValueError(f"Piece without points/connectivity/offsets in {path.name}.")
            sizes = np.diff(np.concatenate(([0], offsets)))
            if not np.all(sizes == sizes[0]):
                raise ValueError(f"Mixed cell sizes in {path.name} are not supported.")
            cell_data: dict[str, np.ndarray] = {}
            for match in _ARRAY.finditer(_section(body, b"CellData", path)):
                name = (match.group(2) or b"").decode()
                if cell_arrays is None or name in cell_arrays:
                    cell_data[name] = read_array(match).astype(np.float64)
            pieces.append(
                VtuPiece(
                    points=points,
                    connectivity=connectivity.reshape(n_cells, int(sizes[0])),
                    cell_data=cell_data,
                )
            )
            tick(count)
    return ParsedVtu(pieces=pieces, field_data=field_data)

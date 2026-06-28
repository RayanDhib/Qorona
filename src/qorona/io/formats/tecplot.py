"""Parser for COOLFluiD Tecplot ASCII (``.plt``) solution files (model-agnostic container).

Reads a single finite-element volume zone written by COOLFluiD's Tecplot writer into a neutral
:class:`ParsedTecplot`: the nodal and cell-centred variables keyed by the file's own ``VARIABLES``
names, plus the element connectivity. It carries no model meaning (no notion of "corona", of which
column is ``B``, or of units); a model reader in :mod:`qorona.io.readers` maps the raw variables and
builds the :class:`~qorona.io.native.NativeSolution`.

The supported layout is the one the COCONUT corona export uses: ASCII, ``DATAPACKING=BLOCK`` (each
variable a contiguous block, a ``NODAL`` variable ``N`` values long and a ``CELLCENTERED`` one ``E``
long, in ``VARIABLES`` order), a single ``ZONETYPE=FEBRICK`` zone, then ``E`` rows of eight 1-based
node indices. Prisms are stored as degenerate bricks ``[a,b,c,c,d,e,f,f]`` (columns 3 and 7 repeat
the triangle apexes), so the six unique prism nodes are columns :data:`PRISM_COLUMNS`. Any other
variant (binary, ``POINT`` packing, multiple zones, a non-``FEBRICK`` element type) raises.

The body is read by streaming whitespace-separated tokens in bounded-memory chunks, so multi-GB
files parse without loading the whole text, with progress reported through :mod:`qorona.console`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from qorona.console import progress_bar
from qorona.io.textio import open_solution_text

__all__ = ["FEBRICK_NODES", "PRISM_COLUMNS", "ParsedTecplot", "parse_tecplot"]

#: Node count of a Tecplot ``FEBRICK`` element.
FEBRICK_NODES = 8

#: 0-based columns of the six unique prism nodes in a degenerate ``FEBRICK`` ``[a,b,c,c,d,e,f,f]``
#: (columns 3 and 7 repeat the two triangle apexes). The two triangular faces are columns
#: ``(0, 1, 2)`` and ``(4, 5, 6)``.
PRISM_COLUMNS = (0, 1, 2, 4, 5, 6)

#: Text-chunk size for the streaming numeric reader (characters per read).
_CHUNK_CHARS = 1 << 22


@dataclass
class ParsedTecplot:
    """A parsed Tecplot volume zone, in the file's own variable names and raw values.

    Attributes
    ----------
    title
        The file ``TITLE`` (empty if absent).
    n_nodes, n_elements
        Zone node and element counts (``N`` / ``E``).
    nodal
        ``name -> (n_nodes,)`` arrays for the ``NODAL`` variables (the coordinates ``x0,x1,x2``).
    cell
        ``name -> (n_elements,)`` arrays for the ``CELLCENTERED`` variables.
    connectivity
        ``(n_elements, 8)`` 0-based node indices (the raw degenerate brick).
    solution_time
        The zone ``SOLUTIONTIME`` (0.0 if absent).
    """

    title: str
    n_nodes: int
    n_elements: int
    nodal: dict[str, np.ndarray]
    cell: dict[str, np.ndarray]
    connectivity: np.ndarray
    solution_time: float


class _TokenStream:
    """Sequential reader of whitespace-separated numbers from a text handle, in bounded memory.

    Reads the handle in fixed character chunks, parses each chunk to floats, and serves exactly the
    requested count per :meth:`read` (buffering any overrun for the next call), so the body's blocks
    are sliced by count without loading the whole file. A trailing partial token is carried to the
    next chunk so a number split across a chunk boundary is never truncated.
    """

    def __init__(self, handle: object, progress: object) -> None:
        self._handle = handle
        self._progress = progress
        self._text_leftover = ""
        self._pending: list[np.ndarray] = []
        self._pending_size = 0
        self._consumed = 0
        self._eof = False

    def _refill(self) -> bool:
        chunk = self._handle.read(_CHUNK_CHARS)  # type: ignore[attr-defined]
        if not chunk:
            self._eof = True
            if self._text_leftover.strip():
                values = np.fromstring(self._text_leftover, sep=" ")
                self._text_leftover = ""
                if values.size:
                    self._pending.append(values)
                    self._pending_size += values.size
            return False
        data = self._text_leftover + chunk
        cut = max(data.rfind(" "), data.rfind("\n"), data.rfind("\t"), data.rfind("\r"))
        if cut == -1:
            self._text_leftover = data
            return True
        self._text_leftover = data[cut + 1 :]
        values = np.fromstring(data[:cut], sep=" ")
        if values.size:
            self._pending.append(values)
            self._pending_size += values.size
        return True

    def read(self, count: int, dtype: type) -> np.ndarray:
        """Return the next ``count`` values as ``dtype``."""
        while self._pending_size < count and not self._eof:
            self._refill()
        if self._pending_size < count:
            raise ValueError(
                f"Truncated Tecplot body: expected {count} more values, got {self._pending_size}."
            )
        buffer = self._pending[0] if len(self._pending) == 1 else np.concatenate(self._pending)
        out: np.ndarray = buffer[:count].astype(dtype)
        remainder = buffer[count:]
        self._pending = [remainder] if remainder.size else []
        self._pending_size = remainder.size
        self._consumed += count
        self._progress(self._consumed)  # type: ignore[operator]
        return out

    def has_more_nonblank(self) -> bool:
        """Return whether any non-whitespace content remains (e.g. an unsupported second zone)."""
        if self._text_leftover.strip():
            return True
        while not self._eof:
            chunk = self._handle.read(_CHUNK_CHARS)  # type: ignore[attr-defined]
            if not chunk:
                self._eof = True
                break
            if chunk.strip():
                self._text_leftover = chunk
                return True
        return False


def _zone_int(zone: str, key: str) -> int:
    match = re.search(rf"\b{key}\s*=\s*(\d+)", zone)
    if match is None:
        raise ValueError(f"Tecplot ZONE is missing {key}=.")
    return int(match.group(1))


def _zone_word(zone: str, key: str, default: str | None = None) -> str:
    match = re.search(rf"\b{key}\s*=\s*([A-Za-z_]+)", zone)
    if match is None:
        if default is not None:
            return default
        raise ValueError(f"Tecplot ZONE is missing {key}=.")
    return match.group(1).upper()


def _parse_varlocation(zone: str, n_vars: int) -> list[str]:
    """Return per-variable location (``"NODAL"`` / ``"CELLCENTERED"``), 0-based, length ``n_vars``.

    Tecplot's default is ``NODAL`` for any variable not named in ``VARLOCATION``; ranges are
    ``[a-b]=LOC`` or ``[a]=LOC`` with 1-based, inclusive indices.
    """
    location = ["NODAL"] * n_vars
    block = re.search(r"VARLOCATION\s*=\s*\((.*?)\)", zone, re.IGNORECASE)
    if block is None:
        return location
    for lo, hi, loc in re.findall(r"\[\s*(\d+)\s*-\s*(\d+)\s*\]\s*=\s*(\w+)", block.group(1)):
        for index in range(int(lo) - 1, int(hi)):
            location[index] = loc.upper()
    for single, loc in re.findall(r"\[\s*(\d+)\s*\]\s*=\s*(\w+)", block.group(1)):
        location[int(single) - 1] = loc.upper()
    return location


def parse_tecplot(path: object, *, show_progress: bool = True) -> ParsedTecplot:
    """Parse a COOLFluiD Tecplot ASCII ``.plt`` volume zone into a :class:`ParsedTecplot`."""
    from pathlib import Path

    path = Path(str(path))
    title = ""
    var_names: list[str] | None = None
    zone: str | None = None

    with open_solution_text(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            keyword = stripped.split("=", 1)[0].strip().upper()
            if keyword == "TITLE":
                title = stripped.split("=", 1)[1].strip()
            elif keyword == "VARIABLES":
                var_names = re.findall(r'"([^"]*)"', stripped)
            elif keyword.startswith("ZONE"):
                zone = stripped
                break
            elif var_names is None and not stripped[0].isalpha() and "#!" in stripped:
                raise ValueError(f"{path.name} looks like a binary Tecplot file (unsupported).")

        if var_names is None or zone is None:
            raise ValueError(
                f"{path.name} is not an ASCII Tecplot file with VARIABLES and a ZONE header."
            )

        packing = _zone_word(zone, "DATAPACKING", default="POINT")
        if packing != "BLOCK":
            raise ValueError(f"Only DATAPACKING=BLOCK is supported, got {packing} in {path.name}.")
        zonetype = _zone_word(zone, "ZONETYPE", default="ORDERED")
        if zonetype != "FEBRICK":
            raise ValueError(f"Only ZONETYPE=FEBRICK is supported, got {zonetype} in {path.name}.")

        n_nodes = _zone_int(zone, "N")
        n_elements = _zone_int(zone, "E")
        locations = _parse_varlocation(zone, len(var_names))
        lengths = [n_nodes if loc == "NODAL" else n_elements for loc in locations]
        total = sum(lengths) + FEBRICK_NODES * n_elements

        nodal: dict[str, np.ndarray] = {}
        cell: dict[str, np.ndarray] = {}
        time_match = re.search(r"SOLUTIONTIME\s*=\s*([0-9eE.+-]+)", zone)
        solution_time = float(time_match.group(1)) if time_match else 0.0

        with progress_bar(f"Reading {path.name}", total, enabled=show_progress) as progress:
            stream = _TokenStream(handle, progress)
            for name, length, loc in zip(var_names, lengths, locations, strict=True):
                block = stream.read(length, np.float64)
                (nodal if loc == "NODAL" else cell)[name] = block
            flat = stream.read(FEBRICK_NODES * n_elements, np.int64)
            connectivity = flat.reshape(n_elements, FEBRICK_NODES) - 1
            if stream.has_more_nonblank():
                raise ValueError(
                    f"{path.name} has more than one zone; only a single volume zone is supported."
                )

    return ParsedTecplot(
        title=title,
        n_nodes=n_nodes,
        n_elements=n_elements,
        nodal=nodal,
        cell=cell,
        connectivity=connectivity,
        solution_time=solution_time,
    )

"""Reader for the run settings in an Icarus ``.par`` file (the ``&icarus_list`` group).

The group holds the magnetogram time and the relaxation and CME-insertion delays in days. It also
holds ``delta_phi``, Earth's longitude in the Icarus frame at the magnetogram time, in radians.
These values set the epoch and the frame of every snapshot of the run. The reader parses the text
only.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qorona.config import IcarusRunConfig

__all__ = ["read_icarus_par"]

_GROUP = re.compile(r"&icarus_list\b(.*?)(?:^\s*(?:/|&end)|/\s*$)", re.S | re.M | re.I)
_ENTRY = re.compile(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|[^,\s!/]+)")


def read_icarus_par(path: str | Path) -> IcarusRunConfig:
    """Return the run settings from the ``&icarus_list`` group of an Icarus ``.par`` file.

    ``cme_insertion`` and ``delta_phi`` default to 0 when the group does not set them.
    ``delta_phi`` converts from radians to degrees. Raises :class:`ValueError` when the group or
    one of ``magnetogram_time`` and ``relaxation`` is missing.
    """
    from qorona.config import IcarusRunConfig

    path = Path(path)
    text = "\n".join(line.split("!", 1)[0] for line in path.read_text().splitlines())
    group = _GROUP.search(text)
    if group is None:
        raise ValueError(f"{path.name} has no &icarus_list group")
    entries = {key.lower(): value.strip("'\"") for key, value in _ENTRY.findall(group.group(1))}
    missing = [key for key in ("magnetogram_time", "relaxation") if key not in entries]
    if missing:
        raise ValueError(f"the &icarus_list group of {path.name} does not set {', '.join(missing)}")

    def number(key: str) -> float:
        return float(entries.get(key, "0").lower().replace("d", "e"))

    return IcarusRunConfig(
        magnetogram_time=entries["magnetogram_time"],
        relaxation=number("relaxation"),
        cme_insertion=number("cme_insertion"),
        delta_phi=math.degrees(number("delta_phi")),
    )

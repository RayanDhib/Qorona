"""Shared image finishing for the rendered products.

The pieces every image-producing path (the Q⊥ render, the white-light / pB products) finishes
with: the per-channel percentile stretch, the dependency-free PNG writer, and the two occultation
primitives, the in-integral body mask and the image-level eclipse disk.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Literal

import numpy as np

__all__ = ["eclipse_alpha", "occultation_mask", "scale_intensity", "write_png"]


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write a ``(H, W, 3)`` ``uint8`` array to a truecolour 8-bit PNG (no external dependency)."""
    height, width = rgb.shape[:2]

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        crc = struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + body + crc

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit, truecolour RGB
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))  # filter byte 0 per row
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def occultation_mask(impact: np.ndarray, s: np.ndarray, r_occult: float) -> np.ndarray:
    """Return ``True`` where a sample is hidden behind the opaque body ``r < r_occult``.

    A sample's scattered light reaches the observer (toward ``+s``) only if its onward path clears
    the body. For a ray that pierces the body (``rho < r_occult``) the body spans
    ``s ∈ [-s_body, +s_body]`` with ``s_body = sqrt(r_occult² - rho²)``, so a sample behind it
    (``s < -s_body``) is occulted; rays that miss the body occult nothing.
    """
    s_body = np.sqrt(np.clip(r_occult**2 - impact[:, None] ** 2, 0.0, None))
    return (impact[:, None] < r_occult) & (s[None, :] < -s_body)


def eclipse_alpha(impact: np.ndarray, r_occult: float, softness: float) -> np.ndarray:
    """Return the eclipse occulter's per-ray opacity ``alpha`` in ``[0, 1]``.

    The image-level dark disk, the orthogonal companion to :func:`occultation_mask`, which masks
    individual line-of-sight samples *inside* the integral (the opaque body). This is a
    post-stretch radial darkening on the *finished* image, so it never touches the hot loop and is
    identical across the kernel and NumPy paths. ``alpha = 0`` over the opaque disk core (so it
    reads dark), ramping to ``1`` at the limb, leaving off-limb corona untouched.

    With ``softness <= 0`` the edge is a hard black circle (``alpha = 1`` only where the impact
    parameter is ``>= r_occult``); with ``softness > 0`` ``alpha`` follows a smoothstep across
    ``[r_occult - softness, r_occult]`` for a feathered, slightly-transparent limb. ``impact`` is
    the per-ray impact parameter (``rays.impact``).
    """
    if softness <= 0.0:
        return (impact >= r_occult).astype(np.float64)
    t = np.clip((impact - (r_occult - softness)) / softness, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def scale_intensity(
    values: np.ndarray,
    scaling: Literal["linear", "log"],
    percentiles: tuple[float, float],
    anchor: np.ndarray | None = None,
) -> np.ndarray:
    """Map each channel of a ``(n, C)`` per-pixel array to display intensity in ``[0, 1]``.

    Each channel is stretched between its **own** low/high percentiles: a pooled stretch is
    dominated by the shallow channels and crushes a steep one to ≈0, while per-channel balancing
    lets every channel use the full range. ``NaN`` pixels map to 0.

    ``anchor`` (a boolean mask over the ``n`` pixels) restricts which pixels set the percentiles,
    used to anchor a stretch on the pixels that carry the product (the disk for a low-corona
    preset, the positive-brightness corona for a white-light frame) so background pixels cannot
    drag it. The resulting stretch is still applied to every pixel; the mask falls back to all
    finite pixels where it selects none.
    """
    scaled = np.log10(np.clip(values, 1e-300, None)) if scaling == "log" else values
    out = np.zeros_like(scaled, dtype=np.float64)
    for channel in range(scaled.shape[-1]):
        column = scaled[..., channel]
        finite = np.isfinite(column)
        if not finite.any():
            continue
        sample = finite if anchor is None else finite & anchor
        if not sample.any():
            sample = finite
        low, high = np.percentile(column[sample], percentiles)
        span = high - low if high > low else 1.0
        out[..., channel] = np.nan_to_num(np.clip((column - low) / span, 0.0, 1.0))
    return out

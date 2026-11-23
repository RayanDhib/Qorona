"""Build the EUV response tables that Qorona ships.

For each instrument channel ``b`` and abundance set, the table holds

    K_b(T, nₑ) = (n_H / nₑ) (1 / 4π) Σ  ε(λ; T, nₑ) R_b(λ)

with ``ε`` the photon emission of the lines and continua per unit ``n_H nₑ`` and ``R_b`` the
channel's wavelength response. ``K_b`` is in ``cm⁵ s⁻¹ sr⁻¹`` times the channel unit (DN, or
detected photons for EUVI), so that one pixel of the instrument records ``Ω_pix ∫ nₑ² K_b ds``.
The table also holds the foreground absorption cross-section per hydrogen nucleus, ``κ_b(T)`` in
``cm²``, at the channel's nominal wavelength.

The emission comes from the CHIANTI database through fiasco: bound-bound lines from the level
populations of every ion, plus the free-free, free-bound and two-photon continua, in collisional
ionization equilibrium. In photon units the line energy cancels, so each ion folds as
``f(T)/nₑ · Σ_j N_j(T, nₑ) Σ_{lines from j} A R_b(λ)``. Level populations do not depend on the
abundance set, so every element is computed once with unit abundance and weighted per set. An
ion is computed only at the temperatures where it, or for the free-bound continuum the next ion,
is present. Each finished ion is cached under the output directory, so a stopped build resumes.
Delete the cache after a change to the physics in this script.

Run this script in its own environment (``tools/euv-tables-environment.yml``, Python 3.12 with
fiasco and aiapy). It is a maintainer tool and is not part of the installed package. The
photoionization cross-sections follow ``ismeuv.pro`` of the IDL Astronomy User's Library
(W. Landsman, BSD-2-Clause): hydrogenic H I and He II, and the He I fit for wavelengths of 46 Å
and more.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np

#: The temperature axis of every table: log10 T from 4.0 to 8.0 by 0.02.
LOG_T = np.round(np.arange(4.0, 8.0 + 1e-9, 0.02), 4)
#: The density axis of every table: log10 nₑ (cm⁻³) from 6 to 13 by 0.1.
LOG_NE = np.round(np.arange(6.0, 13.0 + 1e-9, 0.1), 4)
#: The wavelength grid on which the continua are folded (Å).
CONTINUUM_WAVELENGTH = np.arange(20.0, 1000.0 + 1e-9, 0.25)
#: Elements included, by atomic number (H to Zn).
MAX_ATOMIC_NUMBER = 30
#: An ion contributes only at temperatures where its ionization fraction exceeds this value.
MIN_IONIZATION_FRACTION = 1e-10
#: Temperatures per level-population solve. fiasco holds one dense rate matrix per temperature,
#: so chunks bound the memory of the large iron ions.
TEMPERATURE_CHUNK = 17
#: The floor applied to K before the logarithm is taken (same unit as K).
K_FLOOR = 1e-60
#: The abundance set whose proton-to-electron ratio enters the level populations.
POPULATION_ABUNDANCE = "sun_coronal_2021_chianti"
#: Abundance sets, by the name used in the table file names.
ABUNDANCE_SETS = {
    "coronal-2021": "sun_coronal_2021_chianti",
    "schmelz-2012": "sun_coronal_2012_schmelz",
}
#: Average plate scale of the EUI Full Sun Imager (arcsec), from the EUI team's parameters. The
#: published FSI responses include the pixel solid angle of this plate scale.
FSI_PLATE_SCALE_ARCSEC = 4.43873
#: EUVI plate scale (arcsec) per spacecraft, from the ``CDELT1`` of the Level 0 headers.
EUVI_PLATE_SCALE_ARCSEC = {"a": 1.58777, "b": 1.59000}

_ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn"
).split()
_HC_ERG_ANGSTROM = (1.0 * u.ph).to_value(u.erg, equivalencies=u.spectral_density(1.0 * u.AA))
_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


@dataclass(frozen=True)
class Channel:
    """One instrument channel and its wavelength response.

    Attributes
    ----------
    name
        Table name, for example ``aia-171`` or ``euvi-a-195``.
    instrument, band
        Instrument name and nominal wavelength (Å).
    unit
        What one count of the channel is: ``DN`` or ``ph`` (detected photons).
    wavelength
        Response wavelengths (Å).
    response
        Response per incident photon (cm² DN ph⁻¹ or cm²), without the pixel solid angle.
    native_pixel_arcsec
        Plate scale of the instrument's native pixel.
    source
        The response file and its SHA-256 digest.
    """

    name: str
    instrument: str
    band: int
    unit: str
    wavelength: np.ndarray
    response: np.ndarray
    native_pixel_arcsec: float
    source: dict[str, str]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aia_channels(instrument_file: Path) -> list[Channel]:
    from aiapy.response import Channel as AiaChannel

    channels = []
    for band in (94, 131, 171, 193, 211, 335):
        aia = AiaChannel(band * u.AA, instrument_file=str(instrument_file))
        response = aia.wavelength_response()
        plate_scale = aia.plate_scale.to_value(u.sr / u.pix)
        channels.append(
            Channel(
                name=f"aia-{band}",
                instrument="AIA",
                band=band,
                unit="DN",
                wavelength=aia.wavelength.to_value(u.AA),
                response=response.to_value(u.cm**2 * u.DN / u.ph),
                native_pixel_arcsec=float(np.sqrt(plate_scale) / _ARCSEC_TO_RAD),
                source={
                    "file": instrument_file.name,
                    "sha256": _digest(instrument_file),
                    "note": "aiapy wavelength_response, no degradation, crosstalk included",
                },
            )
        )
    return channels


def _euvi_channels(effective_area_file: Path, spacecraft: str) -> list[Channel]:
    from scipy.io import readsav

    record = readsav(str(effective_area_file), python_dict=True)["p0"][0]
    wavelength = np.asarray(record["LAMBDA"], dtype=np.float64)
    bands = np.asarray(record["WAVELNTH"])[:, 0]
    filters = [entry.decode().strip() for entry in np.asarray(record["FILTER"])[0]]
    open_filter = filters.index("OPEN")
    channels = []
    for band in (171, 195, 284):
        row = int(np.flatnonzero(bands == band)[0])
        channels.append(
            Channel(
                name=f"euvi-{spacecraft}-{band}",
                instrument=f"EUVI-{spacecraft.upper()}",
                band=band,
                unit="ph",
                wavelength=wavelength,
                response=np.asarray(record["AREA"][row, open_filter], dtype=np.float64),
                native_pixel_arcsec=EUVI_PLATE_SCALE_ARCSEC[spacecraft],
                source={
                    "file": effective_area_file.name,
                    "sha256": _digest(effective_area_file),
                    "note": "OPEN filter wheel; effective area includes CCD quantum efficiency",
                },
            )
        )
    return channels


def _fsi_channel(response_file: Path) -> Channel:
    table = np.loadtxt(response_file, comments="#")
    pixel_solid_angle = (FSI_PLATE_SCALE_ARCSEC * _ARCSEC_TO_RAD) ** 2
    return Channel(
        name="eui-fsi-174",
        instrument="EUI/FSI",
        band=174,
        unit="DN",
        wavelength=table[:, 0] * 10.0,
        response=table[:, 1] / pixel_solid_angle,
        native_pixel_arcsec=FSI_PLATE_SCALE_ARCSEC,
        source={
            "file": response_file.name,
            "sha256": _digest(response_file),
            "note": "file unit DN ph-1 cm2 sr divided by the FSI pixel solid angle",
        },
    )


def _hydrogenic_cross_section(wavelength: float, nuclear_charge: int) -> float:
    """Photoionization cross-section of a hydrogenic ground state (cm²), zero above the edge."""
    ratio = nuclear_charge**2 * wavelength / 911.75
    if ratio >= 1.0:
        return 0.0
    z = np.sqrt(ratio / (1.0 - ratio))
    denominator = nuclear_charge**2 * (1.0 - np.exp(-2.0 * np.pi * z))
    return float(3.44e-16 * ratio**4 * np.exp(-4.0 * z * np.arctan(1.0 / z)) / denominator)


def _helium_i_cross_section(wavelength: float) -> float:
    """He I photoionization cross-section (cm²) without the autoionization resonances."""
    if wavelength >= 503.97:
        return 0.0
    if wavelength >= 46.0:
        coefficients = (
            -2.953607e01,
            7.083061e00,
            8.678646e-01,
            -1.221932e00,
            4.052997e-02,
            1.317109e-01,
            -3.265795e-02,
            2.500933e-03,
        )
    else:
        coefficients = (
            -2.465188e01,
            4.354679e00,
            -3.553024e00,
            5.573040e00,
            -5.872938e00,
            3.720797e00,
            -1.226919e00,
            1.576657e-01,
        )
    x = np.log10(wavelength)
    return float(10.0 ** sum(c * x**i for i, c in enumerate(coefficients)))


def _fold_weights(channels: list[Channel], wavelength: np.ndarray) -> np.ndarray:
    """Each channel's response at ``wavelength`` (Å), zero outside its tabulated range."""
    return np.stack(
        [np.interp(wavelength, c.wavelength, c.response, left=0.0, right=0.0) for c in channels],
        axis=-1,
    )


# Worker state, set once per process by _init_worker.
_WORKER: dict[str, object] = {}


def _init_worker(hdf5_root: str, channels: list[Channel], population_abundance: str) -> None:
    import fiasco

    fiasco.log.setLevel(logging.ERROR)
    continuum_weights = _fold_weights(channels, CONTINUUM_WAVELENGTH)
    # Photons per erg at each continuum wavelength, times the bin width.
    photons_per_erg = CONTINUUM_WAVELENGTH / _HC_ERG_ANGSTROM
    step = np.gradient(CONTINUUM_WAVELENGTH)
    _WORKER.update(
        hdf5_root=hdf5_root,
        channels=channels,
        population_abundance=population_abundance,
        continuum_fold=continuum_weights * (photons_per_erg * step)[:, None],
    )


def _chunk_contribution(
    ion_name: str, log_t: np.ndarray, *, recombination: bool
) -> tuple[np.ndarray, list[str]]:
    """One part of an ion's contribution over the temperatures ``log_t``, ``(n_t, n_ne, n_ch)``.

    With ``recombination`` false, the part is the bound-bound lines and the free-free and
    two-photon continua, weighted by this ion's fraction. With ``recombination`` true, it is the
    free-bound continuum of the recombination onto this ion, weighted by the next ion's fraction.
    """
    import fiasco
    from fiasco.util.exceptions import MissingDatasetException

    channels: list[Channel] = _WORKER["channels"]  # type: ignore[assignment]
    continuum_fold: np.ndarray = _WORKER["continuum_fold"]  # type: ignore[assignment]
    density = 10.0**LOG_NE * u.cm**-3
    wavelength = CONTINUUM_WAVELENGTH * u.AA
    emissivity = u.erg * u.cm**3 / u.s / u.AA
    ion = fiasco.Ion(
        ion_name,
        10.0**log_t * u.K,
        abundance=_WORKER["population_abundance"],
        hdf5_dbase_root=_WORKER["hdf5_root"],
    )
    total = np.zeros((log_t.size, LOG_NE.size, len(channels)))
    notes: list[str] = []
    if recombination:
        next_fraction = np.asarray(ion.next_ion().ionization_fraction, dtype=np.float64)
        try:
            free_bound = ion.free_bound(wavelength).to_value(emissivity)
            total += (next_fraction[:, None] * (free_bound @ continuum_fold))[:, None, :]
        except MissingDatasetException:
            pass
        return total, notes

    fraction = np.asarray(ion.ionization_fraction, dtype=np.float64)
    # Bound-bound lines: f(T)/nₑ · Σ_j N_j Σ_{lines from j} A R_b(λ).
    try:
        transitions = ion.transitions
        bound = transitions.is_bound_bound
        line_wavelength = np.abs(transitions.wavelength[bound].to_value(u.AA))
        einstein_a = transitions.A[bound].to_value(1 / u.s)
        upper = np.asarray(transitions.upper_level[bound], dtype=np.int64)
        line_weights = _fold_weights(channels, line_wavelength) * einstein_a[:, None]
        if np.any(line_weights > 0.0):
            level_weights = np.zeros((ion.n_levels, len(channels)))
            np.add.at(level_weights, upper - 1, line_weights)
            populations = np.asarray(ion.level_populations(density), dtype=np.float64)
            total += (
                fraction[:, None, None]
                / density.to_value(u.cm**-3)[None, :, None]
                * (populations @ level_weights)
            )
    except MissingDatasetException as error:
        notes.append(f"lines skipped: {error}")
    # Free-free and two-photon continua.
    try:
        free_free = ion.free_free(wavelength).to_value(emissivity)
        total += (fraction[:, None] * (free_free @ continuum_fold))[:, None, :]
    except MissingDatasetException:
        pass
    try:
        two_photon = ion.two_photon(wavelength, density).to_value(emissivity)
        total += fraction[:, None, None] * (two_photon @ continuum_fold)
    except MissingDatasetException:
        pass
    return total, notes


def _ion_contribution(ion_name: str) -> tuple[str, np.ndarray, list[str]]:
    """The per-unit-abundance contribution of one ion to every channel, ``(n_t, n_ne, n_ch)``.

    The result is in channel counts ``cm⁵ s⁻¹`` per unit ``n_H nₑ``, emitted into 4π sr. It is
    zero at the temperatures where the ion (or the next ion, for the free-bound continuum) is
    absent.
    """
    import fiasco
    from fiasco.util.exceptions import MissingDatasetException

    n_channels = len(_WORKER["channels"])  # type: ignore[arg-type]
    total = np.zeros((LOG_T.size, LOG_NE.size, n_channels))
    ion = fiasco.Ion(ion_name, 10.0**LOG_T * u.K, hdf5_dbase_root=_WORKER["hdf5_root"])
    try:
        fraction = np.asarray(ion.ionization_fraction, dtype=np.float64)
    except MissingDatasetException:
        return ion_name, total, ["no ionization fraction"]
    next_fraction = np.zeros_like(fraction)
    if ion.charge_state < ion.atomic_number:
        try:
            next_fraction = np.asarray(ion.next_ion().ionization_fraction, dtype=np.float64)
        except MissingDatasetException:
            pass
    notes: list[str] = []
    for recombination, present in ((False, fraction), (True, next_fraction)):
        index = np.flatnonzero(present > MIN_IONIZATION_FRACTION)
        for start in range(0, index.size, TEMPERATURE_CHUNK):
            chunk = index[start : start + TEMPERATURE_CHUNK]
            part, part_notes = _chunk_contribution(
                ion_name, LOG_T[chunk], recombination=recombination
            )
            total[chunk] += part
            notes.extend(note for note in part_notes if note not in notes)
    return ion_name, total, notes


def _ion_names(hdf5_root: str) -> list[str]:
    import fiasco

    fiasco.log.setLevel(logging.ERROR)
    temperature = 10.0**LOG_T * u.K
    names = []
    for symbol in _ELEMENTS[:MAX_ATOMIC_NUMBER]:
        element = fiasco.Element(symbol, temperature, hdf5_dbase_root=hdf5_root)
        names.extend(ion.ion_name for ion in element)
    return names


def abundances_of(hdf5_root: str, abundance_file: str) -> dict[str, float]:
    """The abundance of each element relative to hydrogen, from a CHIANTI abundance file."""
    import fiasco
    from fiasco.util.exceptions import MissingDatasetException

    temperature = 10.0**LOG_T * u.K
    values = {}
    for symbol in _ELEMENTS[:MAX_ATOMIC_NUMBER]:
        ion = fiasco.Ion(
            f"{symbol} 1", temperature, abundance=abundance_file, hdf5_dbase_root=hdf5_root
        )
        # An element absent from the set has no abundance.
        try:
            values[symbol] = float(ion.abundance)
        except MissingDatasetException:
            values[symbol] = 0.0
    return values


def _absorption(hdf5_root: str, abundances: dict[str, float], band: int) -> np.ndarray:
    """``κ_b(T)`` per hydrogen nucleus (cm²) at the nominal wavelength ``band`` (Å)."""
    import fiasco

    temperature = 10.0**LOG_T * u.K

    def fraction(name: str) -> np.ndarray:
        ion = fiasco.Ion(name, temperature, hdf5_dbase_root=hdf5_root)
        return np.asarray(ion.ionization_fraction, dtype=np.float64)

    helium = abundances["He"] / abundances["H"]
    return fraction("H 1") * _hydrogenic_cross_section(band, 1) + helium * (
        fraction("He 1") * _helium_i_cross_section(band)
        + fraction("He 2") * _hydrogenic_cross_section(band, 2)
    )


def _electron_to_hydrogen(abundances: dict[str, float]) -> float:
    """``nₑ / n_H`` for a fully ionized plasma of these abundances."""
    return (
        sum(abundances[symbol] * (z + 1) for z, symbol in enumerate(_ELEMENTS[:MAX_ATOMIC_NUMBER]))
        / abundances["H"]
    )


def _cache_key(hdf5: Path, channels: list[Channel], population_abundance: str) -> str:
    """A key that changes when the database, a response, a grid or the population set changes."""
    record = {
        "database": hdf5.name,
        "channels": [(c.name, c.source["sha256"]) for c in channels],
        "responses": hashlib.sha256(
            b"".join(np.ascontiguousarray(c.response).tobytes() for c in channels)
        ).hexdigest(),
        "grids": [LOG_T.tolist(), LOG_NE.tolist(), CONTINUUM_WAVELENGTH.tolist()],
        "min_fraction": MIN_IONIZATION_FRACTION,
        "population_abundance": population_abundance,
    }
    return hashlib.sha256(json.dumps(record).encode()).hexdigest()[:12]


def fold_elements(
    hdf5: Path,
    channels: list[Channel],
    *,
    cache_root: Path,
    workers: int,
    ions: list[str] | None = None,
    population_abundance: str = POPULATION_ABUNDANCE,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Fold every ion of the database through ``channels`` and sum the ions of each element.

    Parameters
    ----------
    hdf5
        The fiasco CHIANTI HDF5 database.
    channels
        The channels to fold.
    cache_root
        The directory that holds the per-ion cache.
    workers
        Number of parallel processes.
    ions
        The ions to fold (default: every ion of the database).
    population_abundance
        The abundance set whose proton-to-electron ratio enters the level populations.

    Returns
    -------
    elements
        Element symbol → contribution per unit abundance, ``(n_t, n_ne, n_ch)``, in channel
        counts ``cm⁵ s⁻¹`` per unit ``n_H nₑ``, emitted into 4π sr.
    notes
        Ion name → the data that was missing for it.
    """
    hdf5_root = str(hdf5)
    ions = ions or _ion_names(hdf5_root)
    cache = cache_root / f"ions-{_cache_key(hdf5, channels, population_abundance)}"
    cache.mkdir(parents=True, exist_ok=True)

    elements: dict[str, np.ndarray] = {}
    notes: dict[str, list[str]] = {}

    def add(name: str, contribution: np.ndarray, ion_notes: list[str]) -> None:
        symbol = name.split()[0]
        elements[symbol] = elements.get(symbol, 0.0) + contribution
        if ion_notes:
            notes[name] = ion_notes

    pending = []
    for name in ions:
        stem = cache / name.replace(" ", "_")
        if stem.with_suffix(".json").exists():
            add(
                name,
                np.load(stem.with_suffix(".npy")),
                json.loads(stem.with_suffix(".json").read_text()),
            )
        else:
            pending.append(name)
    print(
        f"{len(channels)} channels, {len(ions)} ions ({len(ions) - len(pending)} cached), "
        f"{workers} workers"
    )
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(hdf5_root, channels, population_abundance),
    ) as pool:
        futures = [pool.submit(_ion_contribution, name) for name in pending]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            name, contribution, ion_notes = future.result()
            stem = cache / name.replace(" ", "_")
            np.save(stem.with_suffix(".npy"), contribution)
            stem.with_suffix(".json").write_text(json.dumps(ion_notes))
            add(name, contribution, ion_notes)
            elapsed = time.time() - started
            print(f"[{done}/{len(pending)}] {name:<8} {elapsed:7.0f} s", flush=True)
    return elements, notes


def response_tables(
    elements: dict[str, np.ndarray], abundances: dict[str, float]
) -> tuple[np.ndarray, float]:
    """``K_b`` of every channel for one abundance set, ``(n_t, n_ne, n_ch)``, and ``n_H/nₑ``.

    ``K_b`` is per unit ``nₑ²`` and per steradian.
    """
    hydrogen_per_electron = 1.0 / _electron_to_hydrogen(abundances)
    emission = sum(abundances[symbol] * values for symbol, values in elements.items())
    return hydrogen_per_electron * emission / (4.0 * np.pi), hydrogen_per_electron


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hdf5", required=True, type=Path, help="fiasco CHIANTI HDF5 database")
    parser.add_argument("--ascii", required=True, type=Path, help="CHIANTI ASCII database root")
    parser.add_argument("--aia", required=True, type=Path, help="aia_V8_all_fullinst.genx")
    parser.add_argument("--euvi-a", required=True, type=Path, help="ahead_sra_001.geny")
    parser.add_argument("--euvi-b", required=True, type=Path, help="behind_sra_001.geny")
    parser.add_argument("--fsi", required=True, type=Path, help="FSI 174 response (.txt)")
    parser.add_argument("--output", required=True, type=Path, help="output directory")
    parser.add_argument("--workers", type=int, default=8, help="parallel processes")
    parser.add_argument("--ions", nargs="*", help="restrict to these ions (for timing tests)")
    args = parser.parse_args()

    import aiapy
    import fiasco

    channels = (
        aia_channels(args.aia)
        + _euvi_channels(args.euvi_a, "a")
        + _euvi_channels(args.euvi_b, "b")
        + [_fsi_channel(args.fsi)]
    )
    hdf5_root = str(args.hdf5)
    elements, notes = fold_elements(
        args.hdf5, channels, cache_root=args.output, workers=args.workers, ions=args.ions
    )

    chianti_version = (args.ascii / "VERSION").read_text().strip()
    for set_name, abundance_file in ABUNDANCE_SETS.items():
        abundances = abundances_of(hdf5_root, abundance_file)
        k_table, hydrogen_per_electron = response_tables(elements, abundances)
        for index, channel in enumerate(channels):
            kappa = _absorption(hdf5_root, abundances, channel.band)
            meta = {
                "channel": channel.name,
                "instrument": channel.instrument,
                "band_angstrom": channel.band,
                "unit": channel.unit,
                "k_unit": f"{channel.unit} cm5 s-1 sr-1",
                "kappa_unit": "cm2 per hydrogen nucleus",
                "native_pixel_arcsec": channel.native_pixel_arcsec,
                "abundances": set_name,
                "abundance_file": abundance_file,
                "n_h_over_n_e": hydrogen_per_electron,
                "helium_to_hydrogen": abundances["He"] / abundances["H"],
                "chianti_version": chianti_version,
                "ionization_equilibrium": "chianti",
                "population_abundance_file": POPULATION_ABUNDANCE,
                "fiasco_version": fiasco.__version__,
                "aiapy_version": aiapy.__version__,
                "response": channel.source,
                "continuum": "free-free, free-bound, two-photon",
                "k_floor": K_FLOOR,
                "skipped": notes,
                "built": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            }
            np.savez_compressed(
                args.output / f"{channel.name}_{set_name}.npz",
                log_t=LOG_T,
                log_ne=LOG_NE,
                log_k=np.log10(np.maximum(k_table[:, :, index], K_FLOOR)),
                kappa=kappa,
                meta=np.array(json.dumps(meta)),
            )
    print(f"wrote {len(channels) * len(ABUNDANCE_SETS)} tables to {args.output}")


if __name__ == "__main__":
    main()

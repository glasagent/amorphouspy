"""VACF route workflow: NVE production blocks with block-averaged VDOS and VACF."""

import math
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from amorphouspy.lammps.runner import Thermostat, _run_lammps_md, thermostat_ensembles
from amorphouspy.properties.structural.qn import CutoffSpec
from amorphouspy.properties.vibrational._common import _is_real_number
from amorphouspy.properties.vibrational._trajectory import _MIN_FRAMES
from amorphouspy.properties.vibrational.classification import classify_vibrational_groups
from amorphouspy.properties.vibrational.vacf import compute_partial_vacf
from amorphouspy.properties.vibrational.vdos import VdosWindow, _validate_window, compute_partial_vdos

if TYPE_CHECKING:
    from ase import Atoms

# Spectra of silicate glasses reach about 40 THz; a lower Nyquist frequency cuts and aliases the highest modes.
_MIN_NYQUIST_THZ = 50.0
# Name of the text dump written by lammpsparser in each run directory.
_DUMP_FILE = "dump.out"


def _whole_steps(name: str, duration_ps: float, timestep: float) -> int:
    """Return ``duration_ps`` in MD steps; raise if it is not a whole number of timesteps."""
    steps = duration_ps * 1000 / timestep
    whole_steps = round(steps)
    if not math.isclose(steps, whole_steps, rel_tol=1e-9):
        msg = f"{name}={duration_ps} ps is not a whole number of {timestep} fs timesteps ({steps:g} steps)."
        raise ValueError(msg)
    return whole_steps


def _vdos_simulation_settings(
    potential: pd.DataFrame,
    temperature_sim: float,
    timestep: float,
    production_ps: float,
    block_ps: float,
    dump_every: int,
    equilibration_steps: int,
    normalization: str,
    window: object,
) -> tuple[int, int, int, float]:
    """Validate the arguments of :func:`vdos_simulation`; return ``(block_steps, block_frames, n_blocks, dt_fs)``."""
    if potential.empty:
        msg = "No matching potential found for the given configuration."
        raise ValueError(msg)
    for name, value in (
        ("temperature_sim", temperature_sim),
        ("timestep", timestep),
        ("block_ps", block_ps),
        ("production_ps", production_ps),
    ):
        if not (_is_real_number(value) and np.isfinite(value) and value > 0):
            msg = f"{name} must be positive and finite, got {value}."
            raise ValueError(msg)
    if isinstance(dump_every, bool) or not isinstance(dump_every, int | np.integer) or dump_every < 1:
        msg = f"dump_every must be a positive integer, got {dump_every!r}."
        raise ValueError(msg)
    if (
        isinstance(equilibration_steps, bool)
        or not isinstance(equilibration_steps, int | np.integer)
        or equilibration_steps < 0
    ):
        msg = f"equilibration_steps must not be negative and must be an integer, got {equilibration_steps!r}."
        raise ValueError(msg)
    if normalization not in ("unit", "3N"):
        msg = f"Unknown normalization {normalization!r}; allowed values are 'unit' and '3N'."
        raise ValueError(msg)
    _validate_window(window)
    block_steps = _whole_steps("block_ps", block_ps, timestep)
    if block_steps % dump_every:
        msg = f"The block of {block_steps} steps is not a multiple of dump_every={dump_every}."
        raise ValueError(msg)
    block_frames = block_steps // dump_every
    if block_frames < _MIN_FRAMES:
        msg = f"A block needs at least 2 frames, got {block_frames} ({block_steps} steps, dump_every={dump_every})."
        raise ValueError(msg)
    production_steps = _whole_steps("production_ps", production_ps, timestep)
    if production_steps % block_steps:
        msg = f"The production of {production_steps} steps is not a multiple of the block of {block_steps} steps."
        raise ValueError(msg)
    return block_steps, block_frames, production_steps // block_steps, timestep * dump_every


def _split_total(partial: dict[str, dict[str, NDArray[np.float64]]]) -> dict[str, Any]:
    """Move the all-atoms group of :func:`vdos_simulation` to a top-level ``"total"`` entry."""
    return {"total": partial["total"]["all"], **{name: labels for name, labels in partial.items() if name != "total"}}


def _nve_block(
    structure: "Atoms",
    nve_kwargs: dict[str, Any],
    block: int,
    n_blocks: int,
    block_frames: int,
    masses: NDArray[np.float64],
    dt_fs: float,
    spectral_groups: dict[str, dict[str, NDArray]],
    normalization: Literal["unit", "3N"],
    window: VdosWindow,
    block_dir: str | None,
    *,
    keep_trajectory: bool,
) -> tuple["Atoms", dict[str, Any], NDArray[np.float64], NDArray[np.float64], float]:
    """Run NVE block number ``block`` (1-based) in ``block_dir``.

    Return ``(final structure, spectra, temperature, energy_tot, step-0 energy_tot)``. Unless ``block_dir`` is None
    or ``keep_trajectory`` is True, the block's dump is deleted once its spectra and thermo are extracted.

    The parsed output is local to this function, so its trajectory arrays are freed on return, before the next block
    is launched. Nothing returned here may hold a view of them.
    """
    structure_final, parsed_output = _run_lammps_md(structure=structure, **nve_kwargs, tmp_working_directory=block_dir)

    # LAMMPS also dumps step 0 (the end state of the previous stage); it is dropped below.
    generic = parsed_output["generic"]
    for key in ("velocities", "temperature", "energy_tot"):
        if len(generic[key]) != block_frames + 1:
            msg = (
                f"NVE block {block} of {n_blocks} returned {len(generic[key])} frames of {key!r}, expected "
                f"{block_frames + 1} ({block_frames} plus the step-0 frame)."
            )
            if block_dir is not None:
                dumps = [str(dump) for dump in sorted(Path(block_dir).rglob(_DUMP_FILE))] or [block_dir]
                msg += f" Its dump is kept in {', '.join(dumps)}."
            raise RuntimeError(msg)
    velocities = np.asarray(generic["velocities"])[1:]

    frequencies_thz, no_window = compute_partial_vdos(
        velocities, masses, dt_fs, spectral_groups, normalization=normalization
    )
    vdos = {"no_window": _split_total(no_window)}
    if window is not None:
        _, windowed = compute_partial_vdos(
            velocities, masses, dt_fs, spectral_groups, normalization=normalization, window=window
        )
        vdos["windowed"] = _split_total(windowed)
    lag_ps, vacf = compute_partial_vacf(velocities, masses, dt_fs, spectral_groups)

    spectra = {"frequencies_thz": frequencies_thz, "lag_ps": lag_ps, "vdos": vdos, "vacf": _split_total(vacf)}
    temperature, energy_tot = np.asarray(generic["temperature"]), np.asarray(generic["energy_tot"])

    if block_dir is not None and not keep_trajectory:
        dumps = sorted(Path(block_dir).rglob(_DUMP_FILE))
        if not dumps:
            warnings.warn(
                f"No {_DUMP_FILE} found under {block_dir} for NVE block {block}; nothing was deleted.",
                UserWarning,
                stacklevel=3,
            )
        for dump in dumps:
            dump.unlink()
    return structure_final, spectra, temperature[1:], energy_tot[1:], float(energy_tot[0])


def _slope(x: ArrayLike, y: ArrayLike) -> float:
    """Return the least-squares slope of ``y`` against ``x``."""
    x_array, y_array = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x_centred, y_centred = x_array - x_array.mean(), y_array - y_array.mean()
    return float(x_centred @ y_centred / (x_centred @ x_centred))


def _nve_drift(
    temperature: NDArray[np.float64],
    energy_tot: NDArray[np.float64],
    step0_energies: list[float],
    n_atoms: int,
    dt_fs: float,
) -> dict[str, Any]:
    """Return the energy and temperature drift of the concatenated kept NVE frames; see :func:`vdos_simulation`."""
    n_blocks = len(step0_energies)
    energy_mev = energy_tot * 1e3 / n_atoms
    time_ps = (np.arange(len(energy_tot)) + 1) * dt_fs * 1e-3
    block_energy, block_time = energy_mev.reshape(n_blocks, -1), time_ps.reshape(n_blocks, -1)
    return {
        "energy_drift_mev_per_atom_ps": _slope(time_ps, energy_mev),
        "block_energy_drift_mev_per_atom_ps": np.array(
            [_slope(t, e) for t, e in zip(block_time, block_energy, strict=True)]
        ),
        "boundary_energy_jump_mev_per_atom": np.asarray(step0_energies[1:]) * 1e3 / n_atoms - block_energy[:-1, -1],
        "max_energy_deviation_mev_per_atom": float(np.max(np.abs(energy_mev - energy_mev[0]))),
        "temperature_mean_k": float(np.mean(temperature)),
        "temperature_std_k": float(np.std(temperature)),
        "temperature_drift_k_per_ps": _slope(time_ps, temperature),
    }


def _block_statistics(blocks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the mean and the standard error over blocks of every curve in nested dicts of one layout."""
    mean, sem = {}, {}
    for key, value in blocks[0].items():
        values = [block[key] for block in blocks]
        if isinstance(value, dict):
            mean[key], sem[key] = _block_statistics(values)
        else:
            mean[key] = np.mean(values, axis=0)
            # The standard error of one block is undefined, and np.std with ddof=1 would warn.
            single = len(blocks) == 1
            sem[key] = np.full_like(value, np.nan) if single else np.std(values, axis=0, ddof=1) / np.sqrt(len(blocks))
    return mean, sem


def vdos_simulation(
    structure: "Atoms",
    potential: pd.DataFrame,
    temperature_sim: float = 300.0,
    timestep: float = 1.0,
    server_kwargs: dict[str, Any] | None = None,
    *,
    thermostat: Thermostat,
    equilibration_steps: int = 100_000,
    production_ps: float = 200.0,
    block_ps: float = 10.0,
    dump_every: int = 5,
    normalization: Literal["unit", "3N"] = "unit",
    window: VdosWindow = "hann",
    cutoff: CutoffSpec | None = None,
    former_types: list[int] | None = None,
    o_type: int = 8,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
    keep_trajectory: bool = False,
) -> dict[str, Any]:
    r"""Equilibrate a structure, run consecutive NVE blocks and average their VDOS and VACF, total and per atom group.

    Three MD stages run in LAMMPS at the constant volume of the input cell:

    1. Langevin thermalisation at ``temperature_sim`` for 10,000 steps, with velocities created at ``temperature_sim``
       (as in :func:`diffusion_simulation`).
    2. NVT equilibration with ``thermostat`` for ``equilibration_steps`` steps, with velocities created again at
       ``temperature_sim``. ``equilibration_steps=0`` skips this stage.
    3. NVE production of ``production_ps``, run as ``n_blocks = production_steps // block_steps`` consecutive blocks of
       ``block_ps``, i.e. ``block_steps = block_ps * 1000 / timestep`` steps each, with velocities and thermo output
       every ``dump_every`` steps. The first block starts from the velocities of the previous stage and every later
       block from the end of the block before it, as a new LAMMPS run.

    The production is microcanonical because a thermostat perturbs the velocities and distorts the spectrum. LAMMPS
    also writes the frame of step 0 of every block, which is the end state of the previous stage or block. It is
    dropped, so the kept frames of consecutive blocks do not overlap. Each block holds ``n_frames = block_steps //
    dump_every`` frames spaced by ``dt_fs = timestep * dump_every`` fs, and the frequency resolution is exactly
    ``1 / block_ps`` THz. If the parsed output of a block does not hold ``n_frames + 1`` frames, a ``RuntimeError`` is
    raised; no frame is trimmed silently. The Nyquist frequency is ``1 / (2 dt_fs)``: 100 THz for the defaults (1 fs,
    every 5 steps).

    From the kept frames of each block (velocities in Å/fs, masses from ``structure.get_masses()`` in amu) the function
    computes:

    - The VDOS without a window and, unless ``window`` is ``None``, with ``window`` (Hann by default), both with
      :func:`compute_partial_vdos`, the given ``normalization`` and one shared frequency axis. The total is computed
      as the partial VDOS of a group of all atoms, so it is the total VDOS of :func:`compute_vdos_from_velocities`
      and integrates to 1 (``"unit"``) or ``3 * n_atoms`` (``"3N"``); within each grouping the partials add up to it.
    - The VACF with :func:`compute_partial_vacf`, unwindowed, normalised so that the total $C(0) = 1$, with lags up to
      half the block.

    The centre-of-mass velocity of the whole system is removed from every frame in both.

    Every returned curve is the mean over blocks, with the block standard error ``std(ddof=1) / sqrt(n_blocks)`` in
    ``"vdos_sem"`` and ``"vacf_sem"``; with one block the standard error is NaN. Each block's VDOS and VACF are
    normalised by that block's own kinetic energy, so the mean of the block curves integrates exactly to the
    normalisation, the total VACF has $C(0) = 1$ exactly, and within each grouping the mean partials still add up to
    the mean total. The standard errors are not additive: the standard error of a total is not the sum of those of
    its partials.

    The standard error is the scatter between the blocks of this run, and it assumes that they are independent. The
    blocks come from one NVE trajectory of one glass, so it contains no glass-to-glass variation at all, and nothing
    that stays constant over the run: neither the energy of the run nor a nearly harmonic mode that keeps its energy
    between blocks. A calibration compared the means of two independent thermal histories of one 300-atom
    $\mathrm{SiO_2}$ glass (PMMCS, 300 K, 10 blocks of 10 ps each), where exact standard errors put 94 % of the bins
    within two combined standard errors. Between 5 and 34 THz this held for 93 % (no window) and 95 % (Hann) of the
    bins. Below 5 THz it held for 96 % and 98 %, so no low-frequency underestimate was seen, although 50 bins would not
    show a small one. Between 34 and 40 THz, above the top of the spectrum, it held for only 75 % and 78 %: the weight
    there differed by 4 to 5 %, as did the temperature of the two runs. The lag-1 correlation of the block residuals
    was close to the -0.1 of independent blocks, but a part that is constant over the run cannot show up in it. One
    pair of runs says nothing about larger boxes, lower temperatures or shorter blocks.

    The atom groups (element, oxygen class, $\mathrm{Q}^n$) come from :func:`classify_vibrational_groups` on the input
    ``structure``, before any MD, so thermal motion of bonds near the cutoff cannot change them. They stay fixed for
    all blocks.

    Memory: the velocities of one block take ``n_frames * n_atoms * 3 * 8`` bytes as float64, e.g. 0.29 GB for 2000
    frames (10 ps at 5 fs) of 6000 atoms. The parsed LAMMPS output holds three more arrays of that size (positions,
    unwrapped positions and forces), and parsing it needs more still: a 10 ps block of 5400 atoms peaked 4.5 GB above
    the baseline, 17 times its velocities (measured). The peak scales with one block, not with the whole production: a
    block's parsed output is freed before the next block starts, and only the spectra and the thermo output accumulate.
    Not all freed memory goes back to the system, though: over 8 blocks of 2 ps the resident size levelled off at 1.7
    times the peak of the first block (measured on macOS).

    Disk: with ``tmp_working_directory=None`` every run removes its files. In a caller-supplied directory every run
    keeps a sub-directory, and each NVE block runs in its own ``vdos_nve_block_<block>_<random>`` directory. Its dump
    (``dump.out``, 88 bytes per atom per frame, measured: 0.47 MB per frame and 0.95 GB per 10 ps block at 5400 atoms)
    is deleted once the block's spectra and thermo output are extracted, before the next block starts, so at most one
    dump is on disk at a time. ``log.lammps``, the input script and the data file stay, as do all files of the Langevin
    and NVT stages. ``keep_trajectory=True`` keeps every dump (20 for the defaults). A block that fails keeps its dump,
    and the error names it. If no dump is found to delete, a warning is raised and the result is still returned.

    Drift: ``"nve_drift"`` reports the energy and temperature of the kept NVE frames, frame ``k`` (0-based over all
    blocks) at ``t = (k + 1) * dt_fs`` fs. Each block restarts from the positions and velocities of the previous
    block's dump, written with 6 significant digits, so the total energy can jump at a block boundary. The jump is
    measured between the last kept frame of a block and the step-0 frame of the next, which hold the same state, so it
    is the restart loss alone. The thermo output has 8 significant digits, so energies are resolved to about
    ``|E_tot| / N * 1e-5`` meV/atom, one unit in the last digit (measured for $\mathrm{SiO_2}$ with PMMCS: 3.3e-4
    meV/atom at 300 atoms, 1.9e-4 at 5400). In those runs (300 K, blocks of 2 and 10 ps) every jump was within two
    such units, at most 6.7e-4 meV/atom, and smaller than the drift within any block over the block length.

    Args:
        structure: Input structure, e.g. a quenched glass. Its cell sets the volume of every stage, and the atom
            groups and masses are taken from it.
        potential: LAMMPS potential as a pandas DataFrame.
        temperature_sim: Temperature of the Langevin and NVT stages in K.
        timestep: MD timestep in fs.
        server_kwargs: Additional arguments for the LAMMPS server (e.g. ``{"cores": 4}``).
        thermostat: Thermostat of the NVT stage, ``"nose_hoover"`` or ``"langevin"``.
        equilibration_steps: Length of the NVT stage in MD steps; 0 skips it. Use a small value for quick demos and
            the default for production-quality runs.
        production_ps: Total length of the NVE production in ps, a whole number of timesteps and a multiple of
            ``block_ps``; nothing is truncated.
        block_ps: Length of one NVE block in ps, a whole number of timesteps. It sets the resolution 1 / ``block_ps``
            THz.
        dump_every: Interval between dumped frames in MD steps, a positive integer that divides the block.
        normalization: ``"unit"`` (the total VDOS integrates to 1, in 1/THz) or ``"3N"`` (to ``3 * n_atoms``). The
            VACF is always normalised to $C(0) = 1$.
        window: Window of the second VDOS, as in :func:`compute_partial_vdos`: ``"hann"`` (default), ``"hamming"``,
            ``"blackmanharris"``, ``("gaussian", std)`` with ``std > 0`` in frames (``dt_fs`` apart, so 250 frames
            are 1.25 ps at the defaults), or ``None`` to compute only the unwindowed VDOS.
        cutoff: Former-O cutoff in Å for the groups, as in :func:`classify_vibrational_groups`. Derived from the RDF
            if ``None``.
        former_types: Atomic numbers of the network formers. Derived from the composition if ``None``.
        o_type: Atomic number of oxygen.
        seed: Random seed for the velocities created in the Langevin and NVT stages.
        tmp_working_directory: Directory for temporary simulation files. If None, every run removes its files.
        keep_trajectory: Keep the dump of every NVE block in ``tmp_working_directory`` instead of deleting it once its
            spectra are computed. Requires ``tmp_working_directory``.

    Returns:
        A dict with:
            ``"frequencies_thz"``: frequency axis in THz, length ``n_frames // 2 + 1``, shared by both VDOS.
            ``"vdos"``: mean over blocks, ``{"no_window": ..., "windowed": ...}`` (no ``"windowed"`` entry for
            ``window=None``), each ``{"total": array, "element": {...}, "oxygen": {...}, "qn": {...}}`` with the labels
            of ``"groups"``, in 1/THz.
            ``"vdos_sem"``: block standard error of ``"vdos"``, same layout and units; NaN for one block.
            ``"lag_ps"``: VACF lags in ps, length ``n_frames // 2 + 1``.
            ``"vacf"``: mean over blocks, ``{"total": array, "element": {...}, "oxygen": {...}, "qn": {...}}``,
            dimensionless.
            ``"vacf_sem"``: block standard error of ``"vacf"``, same layout; NaN for one block.
            ``"groups"``: the output of :func:`classify_vibrational_groups` on the input structure.
            ``"structure"``: the final structure of the last NVE block.
            ``"temperature"``: thermo temperature in K of each kept frame of all blocks, as parsed, length
            ``n_blocks * n_frames``.
            ``"energy_tot"``: total energy in eV of each kept frame of all blocks, as parsed, length
            ``n_blocks * n_frames``.
            ``"nve_drift"``: a dict over all kept NVE frames, energies per atom (``N`` atoms):
            ``"energy_drift_mev_per_atom_ps"``, the least-squares slope of ``E_tot / N`` in meV/atom/ps over all
            blocks, boundary jumps included; ``"block_energy_drift_mev_per_atom_ps"``, the same slope within each
            block, array of length ``n_blocks``; ``"boundary_energy_jump_mev_per_atom"``, ``E_tot / N`` of the step-0
            frame of block ``b + 1`` minus that of the last kept frame of block ``b`` in meV/atom, array of length
            ``n_blocks - 1``; ``"max_energy_deviation_mev_per_atom"``, the largest ``|E_tot - E_tot[0]| / N`` in
            meV/atom from the first kept frame; ``"temperature_mean_k"`` and ``"temperature_std_k"`` (population
            standard deviation, ddof=0) in K; and ``"temperature_drift_k_per_ps"``, the least-squares slope in K/ps.
            ``"dt_fs"``: time between frames in fs.
            ``"n_blocks"``: number of NVE blocks.
            ``"n_frames"``: number of kept frames per block.
            ``"resolution_thz"``: 1 / ``block_ps``.
            ``"nyquist_thz"``: 1 / (2 ``dt_fs``) in THz.
            ``"normalization"``: the VDOS normalisation.
            ``"window"``: the window of ``vdos["windowed"]``, as passed.

    Raises:
        ValueError: Before any MD, if the potential is empty, ``temperature_sim``, ``timestep``, ``block_ps`` or
            ``production_ps`` is not positive and finite, ``dump_every`` is not a positive integer,
            ``equilibration_steps`` is negative or not an integer, ``normalization`` is not ``"unit"`` or ``"3N"``,
            ``window`` is unknown, ``block_ps`` or ``production_ps`` is not a whole number of timesteps, the block is
            not a multiple of ``dump_every`` or holds fewer than 2 frames, the production is not a multiple of the
            block, ``thermostat`` is unknown, or ``keep_trajectory`` is True without a ``tmp_working_directory``.
        RuntimeError: If the output of an NVE block does not hold ``n_frames + 1`` frames of velocities, temperature
            and total energy. The message names the block and, with a ``tmp_working_directory``, its kept dump; no
            later block is run.

    Warns:
        UserWarning: If the Nyquist frequency is below 50 THz, or if no dump of a block is found to delete.

    Example:
        ```pycon
        >>> out = vdos_simulation(structure, potential, thermostat="nose_hoover")  # doctest: +SKIP
        >>> out["n_blocks"], out["n_frames"], out["resolution_thz"], out["nyquist_thz"]  # doctest: +SKIP
        (20, 2000, 0.1, 100.0)
        >>> out["vdos"]["windowed"]["element"]["Si"]  # doctest: +SKIP
        >>> out["vdos_sem"]["windowed"]["element"]["Si"]  # doctest: +SKIP
        >>> out["nve_drift"]["energy_drift_mev_per_atom_ps"]  # doctest: +SKIP

        ```
    """
    if keep_trajectory and tmp_working_directory is None:
        msg = "keep_trajectory=True needs a tmp_working_directory; without one every run removes its files."
        raise ValueError(msg)
    block_steps, block_frames, n_blocks, dt_fs = _vdos_simulation_settings(
        potential,
        temperature_sim,
        timestep,
        production_ps,
        block_ps,
        dump_every,
        equilibration_steps,
        normalization,
        window,
    )
    nyquist_thz = 500.0 / dt_fs
    if nyquist_thz < _MIN_NYQUIST_THZ:
        warnings.warn(
            f"The Nyquist frequency is {nyquist_thz:g} THz (timestep {timestep} fs, dump_every {dump_every}), below "
            f"{_MIN_NYQUIST_THZ:g} THz; silicate spectra reach about 40 THz, so the highest modes are cut and aliased.",
            UserWarning,
            stacklevel=2,
        )
    nvt_ensemble, _ = thermostat_ensembles(thermostat)
    groups = classify_vibrational_groups(structure, cutoff, former_types, o_type)
    masses = structure.get_masses()

    equilibrated, _ = _run_lammps_md(
        structure=structure,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=10_000,
        timestep=timestep,
        n_print_thermo=1000,
        initial_temperature=temperature_sim,
        ensemble="nvt_langevin",
        seed=seed,
        server_kwargs=server_kwargs,
    )
    if equilibration_steps > 0:
        equilibrated, _ = _run_lammps_md(
            structure=equilibrated,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_sim,
            n_ionic_steps=equilibration_steps,
            timestep=timestep,
            n_print_thermo=1000,
            initial_temperature=temperature_sim,
            ensemble=nvt_ensemble,
            seed=seed,
            server_kwargs=server_kwargs,
        )
    # No temperature and no initial temperature: plain fix nve from the velocities of the previous stage.
    nve_kwargs = {
        "potential": potential,
        "temperature": None,
        "initial_temperature": None,
        "n_ionic_steps": block_steps,
        "timestep": timestep,
        "n_dump": dump_every,
        "ensemble": "nve",
        "seed": seed,
        "server_kwargs": server_kwargs,
    }
    spectral_groups = {"total": {"all": np.arange(len(structure))}, **groups}
    structure_final = equilibrated
    spectra, temperatures, energies, step0_energies = [], [], [], []
    for block in range(1, n_blocks + 1):
        # Own directory per block, so its dump can be found and deleted; a random suffix avoids collisions on reuse.
        block_dir = (
            None
            if tmp_working_directory is None
            else tempfile.mkdtemp(prefix=f"vdos_nve_block_{block:03d}_", dir=tmp_working_directory)
        )
        structure_final, block_spectra, temperature, energy_tot, step0_energy = _nve_block(
            structure_final,
            nve_kwargs,
            block,
            n_blocks,
            block_frames,
            masses,
            dt_fs,
            spectral_groups,
            normalization,
            window,
            block_dir,
            keep_trajectory=keep_trajectory,
        )
        spectra.append(block_spectra)
        temperatures.append(temperature)
        energies.append(energy_tot)
        step0_energies.append(step0_energy)

    vdos, vdos_sem = _block_statistics([block_spectra["vdos"] for block_spectra in spectra])
    vacf, vacf_sem = _block_statistics([block_spectra["vacf"] for block_spectra in spectra])
    temperature, energy_tot = np.concatenate(temperatures), np.concatenate(energies)

    return {
        "frequencies_thz": spectra[0]["frequencies_thz"],
        "vdos": vdos,
        "vdos_sem": vdos_sem,
        "lag_ps": spectra[0]["lag_ps"],
        "vacf": vacf,
        "vacf_sem": vacf_sem,
        "groups": groups,
        "structure": structure_final,
        "temperature": temperature,
        "energy_tot": energy_tot,
        "nve_drift": _nve_drift(temperature, energy_tot, step0_energies, len(structure), dt_fs),
        "dt_fs": dt_fs,
        "n_blocks": n_blocks,
        "n_frames": block_frames,
        "resolution_thz": 1.0 / block_ps,
        "nyquist_thz": nyquist_thz,
        "normalization": normalization,
        "window": window,
    }

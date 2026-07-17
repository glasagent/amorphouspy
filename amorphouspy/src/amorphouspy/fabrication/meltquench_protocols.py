"""Simulation protocols for melt-quench workflows.

Implementations of various melt-quench protocols for different potentials.

Author
------
Achraf Atila (achraf.atila@bam.de)
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from ase.atoms import Atoms

from amorphouspy.fabrication.pre_equilibration import pre_equilibration_fix_override

# Default melt temperatures per protocol (K)
DEFAULT_MELT_TEMPERATURES: dict[str, float] = {
    "pmmcs": 5000.0,
    "bmp": 4000.0,
    "bjp": 5000.0,
    "shik": 4000.0,
    "du/teter": 5000.0,
    "yang2026": 4000.0,
}

# Dump interval (in fs) for the structural-averaging sampling stage.
# 1000 fs = 1 ps between frames -- well beyond the longest bond-vibration
# periods in oxide glasses:
#   Si-O / P-O stretch  ~28-33 fs  (1000-1200 cm^-1)
#   Al-O stretch        ~42-48 fs  (700-800 cm^-1)
#   O-X-O bending       ~67-83 fs  (400-500 cm^-1)
# analysis and removing trajectory frames from melt-quench completely.


@dataclass
class MeltQuenchParams:
    """Parameters for melt-quench simulation protocols.

    Attributes:
        structure: Initial atomic structure.
        potential: Potential parameters.
        temperature_high: High temperature for melting.
        temperature_low: Low temperature for quenching.
        cooling_steps: Number of steps for cooling phase.
        timestep: MD timestep.
        n_dump: Dump frequency in MD steps.
        n_print_thermo: Thermodynamic print frequency in MD steps. If None,
            defaults to n_dump.
        langevin: Whether to use Langevin dynamics.
        seed: Random seed.
        server_kwargs: Server configuration.
        tmp_working_directory: Temporary directory path.
        equilibration_steps: Override for all fixed equilibration stages inside a protocol.
            If None, each protocol uses its own hardcoded defaults.
        pre_equilibrate: Run the Langevin + nve/limit pre-equilibration block at
            ``temperature_high`` before the first stage. Needed for randomly
            placed structures; set to ``False`` when the starting structure is
            already equilibrated.

    """

    structure: Atoms
    potential: pd.DataFrame
    temperature_high: float
    temperature_low: float
    cooling_steps: int
    timestep: float
    langevin: bool
    seed: int
    n_dump: int | None = None
    n_print_thermo: int | None = None
    server_kwargs: dict | None = None
    tmp_working_directory: str | Path | None = None
    equilibration_steps: int | None = None
    pre_equilibrate: bool = True


def _pre_equilibration_stage(
    run: Callable[..., Any],
    params: MeltQuenchParams,
    history: list[dict | None],
) -> Atoms:
    """Run the optional stage 0: Langevin + nve/limit pre-equilibration at high T.

    Relaxes the randomly placed structure at ``temperature_high`` before the
    protocol's first real stage. The runner's generated integrator fix is
    replaced via the input-control override (same pathway as the pressure
    ramp), so the potential Config stays untouched. ``initial_temperature=0``
    keeps the velocity field uninitialized: atoms start at rest and the
    Langevin thermostat heats them. ``langevin=False`` is required so exactly
    one generated fix line exists to be replaced. When ``pre_equilibrate`` is
    False the stage is skipped and a ``None`` placeholder is appended so stage
    indices stay stable.

    Args:
        run: Protocol runner partial with the potential already bound.
        params: MeltQuenchParams dataclass containing all simulation parameters.
        history: Per-stage thermo list; the stage result (or ``None``) is appended.

    Returns:
        The pre-equilibrated structure, or ``params.structure`` when skipped.

    """
    if not params.pre_equilibrate:
        history.append(None)
        return params.structure

    structure, parsed = run(
        structure=params.structure,
        temperature=params.temperature_high,
        n_ionic_steps=10_000,
        initial_temperature=0,
        pressure=None,
        langevin=False,
        input_control_file={"fix": pre_equilibration_fix_override(params.temperature_high)},
    )
    history.append(parsed.get("generic", None))
    return structure


def pmmcs_protocol(runner: Callable[..., Any], params: MeltQuenchParams) -> tuple[Atoms, list[dict | None]]:
    """Execute the simulation PMMCS protocol.

    Stage 0 optionally pre-equilibrates the random structure at
    ``temperature_high`` (see :func:`_pre_equilibration_stage`); stage 1 then
    equilibrates the liquid directly at ``temperature_high`` -- there is no
    heating ramp.

    Args:
        runner: The function to run LAMMPS MD simulations.
        params: MeltQuenchParams dataclass containing all simulation parameters.

    Returns:
        Final structure and list of per-stage thermo dicts (one per stage, in order).

    """
    run = partial(
        runner,
        potential=params.potential,
        tmp_working_directory=params.tmp_working_directory,
        timestep=params.timestep,
        n_dump=params.n_dump,
        n_print_thermo=params.n_print_thermo,
        langevin=params.langevin,
        server_kwargs=params.server_kwargs,
    )

    history: list[dict | None] = []

    # Stage 0: Pre-equilibration of the random structure at high T
    structure = _pre_equilibration_stage(run, params, history)

    # Stage 1: Equilibration at high T
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        n_ionic_steps=params.equilibration_steps if params.equilibration_steps is not None else 1_000_000,
        initial_temperature=params.temperature_high,
        seed=params.seed,
    )
    history.append(parsed.get("generic", None))

    # Stage 2: Cooling from high to low T
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        temperature_end=params.temperature_low,
        n_ionic_steps=params.cooling_steps,
        initial_temperature=0,
    )
    history.append(parsed.get("generic", None))

    # Stage 3: Pressure release at low T
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=params.equilibration_steps if params.equilibration_steps is not None else 1_000_000,
        initial_temperature=0,
        pressure=0.0,
    )
    history.append(parsed.get("generic", None))

    return structure, history


def bmp_protocol(runner: Callable[..., Any], params: MeltQuenchParams) -> tuple[Atoms, list[dict | None]]:
    """Execute the simulation BMP protocol.

    Stage 0 optionally pre-equilibrates the random structure at
    ``temperature_high`` (see :func:`_pre_equilibration_stage`); stage 1 then
    equilibrates the liquid directly at ``temperature_high`` -- there is no
    heating ramp.

    Args:
        runner: The function to run LAMMPS MD simulations.
        params: MeltQuenchParams dataclass containing all simulation parameters.

    Returns:
        Final structure and list of per-stage thermo dicts (one per stage, in order).

    """
    run = partial(
        runner,
        potential=params.potential,
        tmp_working_directory=params.tmp_working_directory,
        timestep=params.timestep,
        n_dump=params.n_dump,
        n_print_thermo=params.n_print_thermo,
        langevin=params.langevin,
        server_kwargs=params.server_kwargs,
    )

    history: list[dict | None] = []

    # Stage 0: Pre-equilibration of the random structure at high T
    structure = _pre_equilibration_stage(run, params, history)

    # Stage 1: Equilibration at high T
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        n_ionic_steps=params.equilibration_steps if params.equilibration_steps is not None else 1_000_000,
        initial_temperature=params.temperature_high,
        seed=params.seed,
    )
    history.append(parsed.get("generic", None))

    # Stage 2: Cooling from high to low T
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        temperature_end=params.temperature_low,
        n_ionic_steps=params.cooling_steps,
        initial_temperature=0,
    )
    history.append(parsed.get("generic", None))

    # Stage 3: Pressure release at low T
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=params.equilibration_steps if params.equilibration_steps is not None else 1_000_000,
        initial_temperature=0,
        pressure=0.0,
    )
    history.append(parsed.get("generic", None))

    return structure, history


def bjp_protocol(runner: Callable[..., Any], params: MeltQuenchParams) -> tuple[Atoms, list[dict | None]]:
    """Execute the simulation BJP protocol.

    Stage 0 optionally pre-equilibrates the random structure at
    ``temperature_high`` (see :func:`_pre_equilibration_stage`); stage 1 then
    equilibrates the liquid directly at ``temperature_high`` in NPT -- there
    is no heating ramp.

    Args:
        runner: The function to run LAMMPS MD simulations.
        params: MeltQuenchParams dataclass containing all simulation parameters.

    Returns:
        Final structure and list of per-stage thermo dicts (one per stage, in order).

    """
    run = partial(
        runner,
        potential=params.potential,
        tmp_working_directory=params.tmp_working_directory,
        timestep=params.timestep,
        n_dump=params.n_dump,
        n_print_thermo=params.n_print_thermo,
        langevin=params.langevin,
        server_kwargs=params.server_kwargs,
    )

    history: list[dict | None] = []

    # Stage 0: Pre-equilibration of the random structure at high T
    structure = _pre_equilibration_stage(run, params, history)

    # Stage 1: Equilibration at high T
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        n_ionic_steps=params.equilibration_steps if params.equilibration_steps is not None else 100_000,
        initial_temperature=params.temperature_high,
        pressure=0.0,
        seed=params.seed,
    )
    history.append(parsed.get("generic", None))

    # Stage 2: Cooling from high to low T
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        temperature_end=params.temperature_low,
        n_ionic_steps=params.cooling_steps,
        initial_temperature=0,
        pressure=0.0,
    )
    history.append(parsed.get("generic", None))

    # Stage 3: Pressure release at low T
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=params.equilibration_steps if params.equilibration_steps is not None else 100_000,
        initial_temperature=0,
        pressure=0.0,
    )
    history.append(parsed.get("generic", None))

    return structure, history


def shik_protocol(runner: Callable[..., Any], params: MeltQuenchParams) -> tuple[Atoms, list[dict | None]]:
    """Execute the simulation SHIK protocol.

    Prepares the liquid directly at ``temperature_high`` (Sundararaman et al.
    recipe): stage 0 optionally pre-equilibrates the random structure (see
    :func:`_pre_equilibration_stage`), followed by NVT and NPT equilibration
    at the melt temperature, an NPT quench with a pressure ramp, and a final
    anneal. There is no heating ramp.

    Args:
        runner: The function to run LAMMPS MD simulations.
        params: MeltQuenchParams dataclass containing all simulation parameters.

    Returns:
        Final structure and list of per-stage thermo dicts (one per stage, in order).

    """
    run = partial(
        runner,
        potential=params.potential,
        tmp_working_directory=params.tmp_working_directory,
        timestep=params.timestep,
        n_dump=params.n_dump,
        n_print_thermo=params.n_print_thermo,
        langevin=params.langevin,
        server_kwargs=params.server_kwargs,
    )

    history: list[dict | None] = []

    # Stage 0: Pre-equilibration of the random structure at high T
    structure = _pre_equilibration_stage(run, params, history)

    # Stage 1: NVT equilibration at temperature_high for 100 ps
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        n_ionic_steps=params.equilibration_steps
        if params.equilibration_steps is not None
        else int(100_000 / params.timestep),  # 100 ps / (1 fs timestep) = 1e5 steps
        initial_temperature=params.temperature_high,
        pressure=None,  # NVT ensemble
        seed=params.seed,
    )
    history.append(parsed.get("generic", None))

    # Stage 2: NPT equilibration at temperature_high and 0.1 GPa for 700 ps
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        n_ionic_steps=params.equilibration_steps
        if params.equilibration_steps is not None
        else int(700_000 / params.timestep),  # 700 ps
        initial_temperature=0,
        pressure=0.1,  # GPa
    )
    history.append(parsed.get("generic", None))

    # Stage 3: Quenching temperature_high -> temperature_low in NPT
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        temperature_end=params.temperature_low,
        n_ionic_steps=params.cooling_steps,
        initial_temperature=0,
        pressure=0.1,
        pressure_end=0.0,  # ramp pressure from 0.1 -> 0 GPa
    )
    history.append(parsed.get("generic", None))

    # Stage 4: Annealing at temperature_low and 0 GPa for 100 ps in NPT
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=params.equilibration_steps
        if params.equilibration_steps is not None
        else int(100_000 / params.timestep),  # 100 ps
        initial_temperature=0,
        pressure=0.0,
    )
    history.append(parsed.get("generic", None))

    return structure, history


def du_teter_protocol(runner: Callable[..., Any], params: MeltQuenchParams) -> tuple[Atoms, list[dict | None]]:
    """Execute the simulation Du/Teter protocol.

    Uses the same stage sequence as the PMMCS protocol: optional stage 0
    pre-equilibration at ``temperature_high`` (see
    :func:`_pre_equilibration_stage`), NVT equilibration directly at
    ``temperature_high``, NVT cooling, and NPT pressure release.

    Args:
        runner: The function to run LAMMPS MD simulations.
        params: MeltQuenchParams dataclass containing all simulation parameters.

    Returns:
        Final structure and list of per-stage thermo dicts (one per stage, in order).

    """
    run = partial(
        runner,
        potential=params.potential,
        tmp_working_directory=params.tmp_working_directory,
        timestep=params.timestep,
        n_dump=params.n_dump,
        n_print_thermo=params.n_print_thermo,
        langevin=params.langevin,
        server_kwargs=params.server_kwargs,
    )

    history: list[dict | None] = []

    # Stage 0: Pre-equilibration of the random structure at high T
    structure = _pre_equilibration_stage(run, params, history)

    # Stage 1: Equilibration at high T
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        n_ionic_steps=params.equilibration_steps if params.equilibration_steps is not None else 1_000_000,
        initial_temperature=params.temperature_high,
        seed=params.seed,
    )
    history.append(parsed.get("generic", None))

    # Stage 2: Cooling from high to low T
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        temperature_end=params.temperature_low,
        n_ionic_steps=params.cooling_steps,
        initial_temperature=0,
    )
    history.append(parsed.get("generic", None))

    # Stage 3: Pressure release at low T
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=params.equilibration_steps if params.equilibration_steps is not None else 1_000_000,
        initial_temperature=0,
        pressure=0.0,
    )
    history.append(parsed.get("generic", None))

    return structure, history


_YANG_MELT_PRESSURE_ATM = 20000  # atm
_YANG_MELT_PRESSURE_GPA = _YANG_MELT_PRESSURE_ATM * 101325e-9  # ~ GPa


def yang2026_protocol(runner: Callable[..., Any], params: MeltQuenchParams) -> tuple[Atoms, list[dict | None]]:
    """Execute the simulation protocol for the Yang2026 potential.

    Stage 0 optionally pre-equilibrates the random structure at
    ``temperature_high`` (see :func:`_pre_equilibration_stage`). Stages 1-6
    follow the protocol described in Yang et al., J. Non-Cryst. Solids 684,
    124104 (2026):
        1.  20 ps NVT  300 K
        2.  20 ps NPT  300 K, P = 0
        3. 100 ps NPT  T_high, P = 20000 atm
        4. 100 ps NPT  T_high, P = 0
        5. cooling T_high → 300 K at 1 K/ps, NPT P = 0  (cooling_steps controls duration)
        6. 100 ps NPT  300 K, P = 0

    Args:
        runner: The function to run LAMMPS MD simulations.
        params: MeltQuenchParams dataclass containing all simulation parameters.

    Returns:
        Final structure and list of per-stage thermo dicts (one per stage, in order).

    """
    run = partial(
        runner,
        potential=params.potential,
        tmp_working_directory=params.tmp_working_directory,
        timestep=params.timestep,
        n_dump=params.n_dump,
        n_print_thermo=params.n_print_thermo,
        langevin=params.langevin,
        server_kwargs=params.server_kwargs,
    )

    history: list[dict | None] = []

    # Stage 0: Pre-equilibration of the random structure at high T
    structure = _pre_equilibration_stage(run, params, history)

    eq_steps = params.equilibration_steps
    steps_20ps = eq_steps if eq_steps is not None else int(20_000 / params.timestep)
    steps_100ps = eq_steps if eq_steps is not None else int(100_000 / params.timestep)

    # Stage 1: NVT 300 K for 20 ps
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=steps_20ps,
        initial_temperature=params.temperature_low,
        pressure=None,
        seed=params.seed,
    )
    history.append(parsed.get("generic", None))

    # Stage 2: NPT 300 K, P = 0 for 20 ps
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=steps_20ps,
        initial_temperature=0,
        pressure=0.0,
    )
    history.append(parsed.get("generic", None))

    # Stage 3: NPT T_high, P = 20000 atm for 100 ps
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        n_ionic_steps=steps_100ps,
        initial_temperature=0,
        pressure=_YANG_MELT_PRESSURE_GPA,
    )
    history.append(parsed.get("generic", None))

    # Stage 4: NPT T_high, P = 0 for 100 ps
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        n_ionic_steps=steps_100ps,
        initial_temperature=0,
        pressure=0.0,
    )
    history.append(parsed.get("generic", None))

    # Stage 5: cooling T_high → 300 K at 1 K/ps, NPT P = 0
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_high,
        temperature_end=params.temperature_low,
        n_ionic_steps=params.cooling_steps,
        initial_temperature=0,
        pressure=0.0,
    )
    history.append(parsed.get("generic", None))

    # Stage 6: NPT 300 K, P = 0 for 100 ps
    structure, parsed = run(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=steps_100ps,
        initial_temperature=0,
        pressure=0.0,
    )
    history.append(parsed.get("generic", None))

    return structure, history


# Map potential names to protocol functions
PROTOCOL_MAP: dict[str, Callable[..., tuple[Atoms, list[dict | None]]]] = {
    "pmmcs": pmmcs_protocol,
    "bjp": bjp_protocol,
    "shik": shik_protocol,
    "du/teter": du_teter_protocol,
    "bmp": bmp_protocol,  # Use the same protocol for both BMP variants, which only differ in the harmonic vs. SHRM
    "yang2026": yang2026_protocol,
}

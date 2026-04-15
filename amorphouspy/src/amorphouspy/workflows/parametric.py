"""Parametric study workflow for system size and cooling rate sweeps.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from pathlib import Path

from amorphouspy.potentials.potential import generate_potential
from amorphouspy.structure import get_ase_structure, get_structure_dict
from amorphouspy.workflows.meltquench import melt_quench_simulation


def parametric_melt_quench(
    composition: dict[str, float],
    potential_type: str,
    study: dict[int, list[float]],
    temperature_high: float | None = None,
    temperature_low: float = 300.0,
    timestep: float = 1.0,
    heating_rate: float | None = 1e13,
    n_print: int = 1000,
    equilibration_steps: int | None = None,
    *,
    server_kwargs: dict | None = None,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> list[dict]:
    """Run a parametric melt-quench study over system sizes and cooling rates.

    Generates one initial structure per unique system size, then runs a melt-quench
    simulation for each cooling rate assigned to that size.  Each size can have its
    own subset of cooling rates, so the run matrix does not have to be a full grid.

    Args:
        composition: Oxide glass composition mapping oxide formula to mol%.
            Example: ``{"SiO2": 70, "Na2O": 15, "CaO": 15}``.
        potential_type: Potential name (``"pmmcs"``, ``"bjp"``, or ``"shik"``).
        study: Mapping of target atom count to the cooling rates (K/s) to run for
            that size.  Example::

                {
                    1000:  [1e11, 1e12, 1e13],
                    10000: [1e11],
                }

        temperature_high: Melt temperature in K. Defaults to the protocol-specific
            value when ``None`` (4000 K for SHIK, 5000 K for PMMCS/BJP).
        temperature_low: Final glass temperature in K (default 300 K).
        timestep: MD timestep in femtoseconds (default 1.0 fs).
        heating_rate: Heating rate in K/s (default 1e13 K/s). Pass ``None`` to
            defer to the protocol default inside ``melt_quench_simulation``.
        n_print: Output frequency in simulation steps (default 1000).
        equilibration_steps: Override for all fixed equilibration stages inside the
            protocol. ``None`` uses protocol-specific defaults.
        server_kwargs: LAMMPS server kwargs (e.g. ``{"cores": 4}``).
        seed: Random seed for velocity initialisation (default 12345).
        tmp_working_directory: Directory for LAMMPS temporary files.

    Returns:
        Flat list of result dicts, one per (n_atoms, cooling_rate) pair, in the
        order they appear in ``study``.  Each dict contains:

        - ``n_atoms`` — actual atom count after integer formula-unit rounding
        - ``target_n_atoms`` — the requested target (may differ slightly)
        - ``cooling_rate`` — cooling rate used (K/s)
        - ``heating_rate`` — heating rate used (K/s)
        - ``structure`` — final quenched ASE ``Atoms`` object
        - ``result`` — per-stage thermodynamic history from ``melt_quench_simulation``

    Example:
        >>> results = parametric_melt_quench(
        ...     composition={"SiO2": 70, "Na2O": 15, "CaO": 15},
        ...     potential_type="pmmcs",
        ...     study={
        ...         1000:  [1e11, 1e12, 1e13],
        ...         10000: [1e11],
        ...     },
        ... )
        >>> for r in results:
        ...     print(r["target_n_atoms"], r["cooling_rate"], len(r["structure"]))

    """
    heating_rate = heating_rate if heating_rate is not None else 1e13

    # Build one (structure, potential) per unique system size.
    structure_cache: dict[int, tuple] = {}
    for n_atoms in study:
        atoms_dict = get_structure_dict(composition=composition, target_atoms=n_atoms)
        structure = get_ase_structure(atoms_dict=atoms_dict)
        potential = generate_potential(atoms_dict=atoms_dict, potential_type=potential_type)
        structure_cache[n_atoms] = (structure, potential)

    results = []
    for n_atoms, cooling_rates in study.items():
        structure, potential = structure_cache[n_atoms]
        for cooling_rate in cooling_rates:
            mq = melt_quench_simulation(
                structure=structure,
                potential=potential,
                temperature_high=temperature_high,
                temperature_low=temperature_low,
                timestep=timestep,
                heating_rate=heating_rate,
                cooling_rate=cooling_rate,
                n_print=n_print,
                equilibration_steps=equilibration_steps,
                server_kwargs=server_kwargs,
                seed=seed,
                tmp_working_directory=tmp_working_directory,
            )

            results.append(
                {
                    "n_atoms": len(structure),
                    "target_n_atoms": n_atoms,
                    "cooling_rate": cooling_rate,
                    "heating_rate": heating_rate,
                    "structure": mq["structure"],
                    "result": mq["result"],
                }
            )

    return results

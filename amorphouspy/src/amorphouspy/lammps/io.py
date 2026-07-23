"""LAMMPS I/O utilities for amorphouspy: reading LAMMPS output and dump files.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from pathlib import Path
from typing import Any, cast

import ase.io
from ase import Atoms


def load_lammps_dump(
    path: str | Path,
    type_map: dict[int, str] | None = None,
    *,
    frame: int | None = None,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    return_atoms_dict: bool = False,
) -> Atoms | list[Atoms] | tuple[Atoms, dict[str, Any]] | list[tuple[Atoms, dict[str, Any]]]:
    """Read a LAMMPS dump file and return ASE Atoms object(s) with correct chemical symbols.

    By default the full trajectory is returned as a list.  Pass *frame* to get a
    single frame, or *start* / *stop* / *step* to slice the trajectory.

    Args:
        path: Path to the LAMMPS dump file.
        type_map: Mapping from LAMMPS integer type to element symbol,
            e.g. ``{1: "O", 2: "Si"}``. When ``None``, the ``element``
            column stored in the dump file is used.
        frame: Zero-based index of a single frame to read.  Mutually exclusive
            with *start* / *stop* / *step*.
        start: First frame index of the slice (inclusive, default 0).
        stop: Last frame index of the slice (exclusive, default end of file).
        step: Stride between selected frames (default 1).
        return_atoms_dict: When ``True``, also return amorphouspy ``atoms_dict``
            objects alongside the ``ase.Atoms`` objects (default ``False``).

    Returns:
        - Single frame (``frame`` given, *return_atoms_dict* ``False``): ``Atoms``
        - Single frame (``frame`` given, *return_atoms_dict* ``True``): ``(Atoms, dict)``
        - Multiple frames (*return_atoms_dict* ``False``): ``list[Atoms]``
        - Multiple frames (*return_atoms_dict* ``True``): ``list[tuple[Atoms, dict]]``

        Each ``atoms_dict`` contains ``"atoms"``, ``"box"``, and ``"total_atoms"``.

    Raises:
        ValueError: If *frame* is combined with *start* / *stop* / *step*.
        ValueError: If *type_map* is ``None`` and the dump file does not contain
            an ``element`` column.

    Example:
        >>> # Full trajectory
        >>> frames = load_lammps_dump("run.lammpstrj", type_map={1: "O", 2: "Si"})
        >>> # Single frame
        >>> ase_atoms = load_lammps_dump("run.lammpstrj", type_map={1: "O", 2: "Si"}, frame=0)
        >>> # Frames 30-49
        >>> frames = load_lammps_dump("run.lammpstrj", type_map={1: "O", 2: "Si"}, start=30, stop=50)
        >>> # Every 10th frame from 0 to 99
        >>> frames = load_lammps_dump("run.lammpstrj", type_map={1: "O", 2: "Si"}, start=0, stop=100, step=10)

    """
    if frame is not None and any(x is not None for x in (start, stop, step)):
        msg = "'frame' cannot be combined with 'start', 'stop', or 'step'"
        raise ValueError(msg)

    single_frame = frame is not None
    index: int | slice = frame if frame is not None else slice(start, stop, step)

    raw = ase.io.read(str(path), format="lammps-dump-text", index=index)
    if single_frame:
        frames_list: list[Atoms] = [cast("Atoms", raw)]
    else:
        frames_list = [cast("Atoms", a) for a in (raw if isinstance(raw, list) else [raw])]

    def _apply_symbols(atoms: Atoms) -> None:
        if type_map is not None:
            atoms.set_chemical_symbols([type_map[int(t)] for t in atoms.arrays["type"]])
        elif "element" in atoms.arrays:
            atoms.set_chemical_symbols(list(atoms.arrays["element"]))
        else:
            msg = (
                "type_map is required when the dump file does not contain an 'element' column. "
                "Either pass type_map={1: 'O', 2: 'Si', ...} or add "
                "'dump_modify element O Si ...' to your LAMMPS input script."
            )
            raise ValueError(msg)

    for atoms in frames_list:
        _apply_symbols(atoms)

    if not return_atoms_dict:
        return frames_list[0] if single_frame else frames_list

    def _to_dict(atoms: Atoms) -> dict[str, Any]:
        box_length = float(atoms.get_cell()[0][0])
        atoms_list = [
            {"element": sym, "position": list(pos)}
            for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions(), strict=True)
        ]
        return {"atoms": atoms_list, "box": box_length, "total_atoms": len(atoms_list)}

    pairs = [(atoms, _to_dict(atoms)) for atoms in frames_list]
    return pairs[0] if single_frame else pairs


def frames_from_melt_quench_result(
    result: dict,
    initial_structure: Atoms,
    *,
    stage: int = -1,
    stride: int = 1,
) -> list[Atoms]:
    """Extract trajectory frames from a ``melt_quench_simulation`` result dict.

    Args:
        result: The dict returned by :func:`melt_quench_simulation`
            (keys ``"structure"`` and ``"result"``).
        initial_structure: The ``Atoms`` object passed as input to the simulation.
            It carries the correct atomic numbers, which are used to label each frame.
        stage: Index into ``result["result"]`` selecting which protocol stage to
            extract. Defaults to ``-1`` (the final low-temperature equilibration).
        stride: Take every *stride*-th frame. Defaults to ``1`` (all frames).

    Returns:
        A list of :class:`ase.Atoms` objects, one per selected frame, with
        per-frame thermo data stored in ``atoms.info`` and per-atom arrays
        (forces, velocities, unwrapped positions) in ``atoms.arrays``.

    Example:
        >>> result = melt_quench_simulation(atoms, potential)
        >>> frames = frames_from_melt_quench_result(result, atoms)
        >>> (r, rdfs_mean, cumcn_mean, rdfs_sem, cumcn_sem), _ = average_over_frames(
        ...     compute_rdf, frames, r_max=8.0
        ... )

    """
    stage_data = result["result"][stage]
    n_frames = len(stage_data["positions"])
    indices = range(0, n_frames, stride)
    frames: list[Atoms] = []
    for i in indices:
        atoms = initial_structure.copy()
        atoms.set_positions(stage_data["positions"][i])
        atoms.set_cell(stage_data["cells"][i])
        atoms.set_pbc(True)
        if "velocities" in stage_data:
            atoms.set_velocities(stage_data["velocities"][i])
        if "indices" in stage_data:
            atoms.set_array("indices", stage_data["indices"][i])
        # Thermo-derived scalars are logged on a separate (thermo) cadence, so they can be shorter than
        # the per-frame dump arrays (e.g. with a log-spaced dump schedule); guard each access.
        for info_key, data_key in (
            ("temperature", "temperature"),
            ("energy_pot", "energy_pot"),
            ("energy_tot", "energy_tot"),
            ("volume", "volume"),
            ("pressure", "pressures"),
            ("step", "steps"),
        ):
            if data_key in stage_data and i < len(stage_data[data_key]):
                atoms.info[info_key] = stage_data[data_key][i]
        if "forces" in stage_data:
            atoms.arrays["forces"] = stage_data["forces"][i]
        if "unwrapped_positions" in stage_data:
            atoms.arrays["unwrapped_positions"] = stage_data["unwrapped_positions"][i]
        frames.append(atoms)
    return frames


def structure_from_parsed_output(initial_structure: Atoms, parsed_output: dict, *, wrap: bool = False) -> Atoms:
    """Construct an `Atoms` object from parsed output data.

    Args:
        initial_structure: The initial atomic structure to use as a template.
        parsed_output: Parsed output containing atomic positions, cell, and indices.
        wrap: Whether to wrap the atomic positions to the simulation cell (default is False).
            Keeping the unwrapped positions is more beneficial if structures are passed between
            different LAMMPS simulations in one workflow to ensure continuity.

    Returns:
        An `Atoms` object with updated positions and cell.

    Example:
        >>> new_atoms = structure_from_parsed_output(atoms, lammps_output)

    """
    # Take a copy of the initial structure as template and update the relevant properties
    atoms_copy = initial_structure.copy()
    atoms_copy.set_array("indices", parsed_output["generic"]["indices"][-1])
    atoms_copy.set_positions(parsed_output["generic"]["positions"][-1])
    atoms_copy.set_velocities(parsed_output["generic"]["velocities"][-1])
    atoms_copy.set_cell(parsed_output["generic"]["cells"][-1])
    atoms_copy.set_pbc(True)
    if wrap:
        atoms_copy.wrap()

    return atoms_copy

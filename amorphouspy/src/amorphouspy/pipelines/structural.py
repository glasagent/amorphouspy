"""Structural analysis pipeline.

Runs structural characterisation (RDF, coordination, bond angles, etc.)
on a quenched glass structure, optionally averaging over trajectory frames
from the final equilibration stage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from amorphouspy.properties.structural.all import StructureData, analyze_structure

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)


def extract_equilibration_frames(
    final_structure: Atoms,
    simulation_history: list[dict[str, Any]] | None = None,
) -> list[Atoms]:
    """Reconstruct Atoms snapshots from the final equilibration stage.

    Falls back to a single-element list with *final_structure* when no
    simulation history is available or the history contains no position data.

    Args:
        final_structure: The quenched structure from the melt-quench pipeline.
        simulation_history: Full stage-by-stage MD history (optional).

    Returns:
        List of ASE Atoms frames suitable for averaging.
    """
    if not simulation_history:
        return [final_structure]

    last_stage = next((s for s in reversed(simulation_history) if s is not None), None)
    if last_stage is None or "positions" not in last_stage:
        return [final_structure]

    positions = last_stage["positions"]
    cells = last_stage["cells"]
    n_frames = len(positions)

    if n_frames <= 1:
        return [final_structure]

    frames: list[Atoms] = []
    for i in range(n_frames):
        frame = final_structure.copy()
        frame.set_positions(positions[i])
        frame.set_cell(cells[i])
        frame.set_pbc(True)
        frame.wrap()
        frames.append(frame)

    return frames


def run_structural_analysis(
    final_structure: Atoms,
    simulation_history: list[dict[str, Any]] | None = None,
) -> tuple[StructureData, StructureData | None, int]:
    """Run structural analysis, optionally averaging over trajectory frames.

    Args:
        final_structure: Quenched ASE Atoms object.
        simulation_history: Full stage-by-stage MD history for frame averaging.

    Returns:
        Tuple of ``(mean_data, sem_data, n_frames)`` where *sem_data* is
        ``None`` when only one frame is used.
    """
    frames = extract_equilibration_frames(final_structure, simulation_history)
    n_frames = len(frames)

    if n_frames > 1:
        logger.info("Frame-averaging structural analysis over %d frames", n_frames)
        mean_data, sem_data = analyze_structure(atoms=frames, frame_averaging=True)
    else:
        mean_data, sem_data = analyze_structure(atoms=frames[0])

    return mean_data, sem_data, n_frames

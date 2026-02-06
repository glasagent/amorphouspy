"""Test module for glass simulation workflow."""

import shutil
from pathlib import Path

from ase.io import read
from pyiron_base import Project, job

from amorphouspy import (
    generate_potential as _generate_potential,
)
from amorphouspy import (
    get_structure_dict as _get_structure_dict,
)
from amorphouspy import (
    md_simulation as _md_simulation,
)

from . import DATA_DIR

generate_potential = job(_generate_potential)
get_structure_dict = job(_get_structure_dict)
md_simulation = job(_md_simulation)


def test_glass_simulation() -> None:
    """Test the complete glass simulation workflow."""
    project_path = Path("glass")
    pr = Project(str(project_path))

    try:
        atoms_dict = get_structure_dict(
            composition="100SiO2",
            n_molecules=None,
            target_atoms=9,
            mode="molar",
            density=None,
            min_distance=2.0,
            max_attempts_per_atom=10000,
            pyiron_project=pr,
        )

        structure = read(DATA_DIR / "SiO2_glass_300_atoms.xyz")

        server_kwargs = {"cores": 2}

        generated_potential = generate_potential(
            atoms_dict=atoms_dict,
            potential_type="shik",
            pyiron_project=pr,
        )

        delayed = md_simulation(
            structure=structure,
            potential=generated_potential,
            temperature_sim=300.0,
            timestep=1.0,
            production_steps=1_000,
            n_print=100,
            server_kwargs=server_kwargs,
            pressure=0,
            pyiron_project=pr,
        )

        _ = delayed.pull()

    finally:
        pr.remove_jobs(silently=True, recursive=True)

        for tbl in (
            Path("table_O_O.tbl"),
            Path("table_O_Si.tbl"),
            Path("table_Si_Si.tbl"),
        ):
            tbl.unlink(missing_ok=True)

        shutil.rmtree(project_path, ignore_errors=True)

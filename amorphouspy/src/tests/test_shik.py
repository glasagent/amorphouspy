"""Test module for glass simulation workflow.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from ase.io import read
from executorlib import SingleNodeExecutor

from amorphouspy import (
    generate_potential,
    get_structure_dict,
    md_simulation,
)

from . import DATA_DIR


def test_glass_simulation() -> None:
    """Test the complete glass simulation workflow."""
    with SingleNodeExecutor() as exe:
        atoms_dict_future = exe.submit(
            get_structure_dict,
            composition={"SiO2": 100},
            n_molecules=None,
            target_atoms=9,
            mode="molar",
            density=None,
            min_distance=2.0,
            max_attempts_per_atom=10000,
        )
        structure = read(DATA_DIR / "SiO2_glass_300_atoms.xyz")
        server_kwargs = {"cores": 2}
        generated_potential_future = exe.submit(
            generate_potential,
            atoms_dict=atoms_dict_future,
            potential_type="shik",
        )

        delayed_future = exe.submit(
            md_simulation,
            structure=structure,
            potential=generated_potential_future,
            temperature_sim=300.0,
            timestep=1.0,
            production_steps=1_000,
            n_print=100,
            server_kwargs=server_kwargs,
            pressure=0,
        )
        _ = delayed_future.result()

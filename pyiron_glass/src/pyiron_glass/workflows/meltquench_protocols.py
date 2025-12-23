"""Simulation protocols for melt-quench workflows.

Implementations of various melt-quench protocols for different potentials.

Author
------
Achraf Atila (achraf.atila@bam.de)
"""

from abc import ABC, abstractmethod

import pandas as pd
from ase.atoms import Atoms


class MeltQuenchProtocol(ABC):
    """Abstract base class for melt-quench simulation protocols."""

    def __init__(self, runner: callable) -> None:
        """Initialize the protocol.

        Args:
            runner: The function to run LAMMPS MD simulations.
                    Signature should match `_run_lammps_md`.

        """
        self.runner = runner

    @abstractmethod
    def run(
        self,
        structure: Atoms,
        potential: pd.DataFrame | dict,
        temperature_high: float,
        temperature_low: float,
        heating_steps: int,
        cooling_steps: int,
        timestep: float,
        n_print: int,
        *,
        langevin: bool,
        seed: int,
        server_kwargs: dict | None,
        tmp_working_directory: str | None,
    ) -> tuple[Atoms, dict]:
        """Execute the simulation protocol."""


class PMMCSProtocol(MeltQuenchProtocol):
    """Protocol for PMMCS potential."""

    def run(
        self,
        structure: Atoms,
        potential: pd.DataFrame | dict,
        temperature_high: float,
        temperature_low: float,
        heating_steps: int,
        cooling_steps: int,
        timestep: float,
        n_print: int,
        *,
        langevin: bool,
        seed: int,
        server_kwargs: dict | None,
        tmp_working_directory: str | None,
    ) -> tuple[Atoms, dict]:
        """Execute the simulation PMMCS protocol."""
        # Stage 1: Heating from low to high T
        structure, _ = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            temperature_end=temperature_high,
            n_ionic_steps=heating_steps,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=temperature_low,
            langevin=langevin,
            seed=seed,
            server_kwargs=server_kwargs,
        )

        # Stage 2: Equilibration at high T
        structure, _ = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,
            n_ionic_steps=10_000,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # Stage 3: Cooling from high to low T
        structure, _ = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,
            temperature_end=temperature_low,
            n_ionic_steps=cooling_steps,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # Stage 4: Pressure release at low T
        structure, _ = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            n_ionic_steps=10_000,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=0.0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # Stage 5: Long equilibration at low T
        structure_final, parsed_output = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            n_ionic_steps=100_000,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        return structure_final, parsed_output


class BJPProtocol(MeltQuenchProtocol):
    """Protocol for BJP potential."""

    def run(
        self,
        structure: Atoms,
        potential: pd.DataFrame | dict,
        temperature_high: float,
        temperature_low: float,
        heating_steps: int,
        cooling_steps: int,
        timestep: float,
        n_print: int,
        *,
        langevin: bool,
        seed: int,
        server_kwargs: dict | None,
        tmp_working_directory: str | None,
    ) -> tuple[Atoms, dict]:
        """Execute the simulation BJP protocol."""
        # Stage 1: Heating from low to high T
        structure, _ = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            temperature_end=temperature_high,
            n_ionic_steps=heating_steps,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=temperature_low,
            pressure=0.0,
            langevin=langevin,
            seed=seed,
            server_kwargs=server_kwargs,
        )

        # Stage 2: Equilibration at high T
        structure, _ = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,
            n_ionic_steps=100_000,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=0.0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # Stage 3: Cooling from high to low T
        structure, _ = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,
            temperature_end=temperature_low,
            n_ionic_steps=cooling_steps,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=0.0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # Stage 4: Pressure release at low T
        structure, _ = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            n_ionic_steps=100_000,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=0.0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # Stage 5: Long equilibration at low T
        structure_final, parsed_output = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            n_ionic_steps=100_000,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        return structure_final, parsed_output


class SHIKProtocol(MeltQuenchProtocol):
    """Protocol for SHIK potential."""

    def run(
        self,
        structure: Atoms,
        potential: pd.DataFrame | dict,
        temperature_high: float,
        temperature_low: float,
        heating_steps: int,
        cooling_steps: int,
        timestep: float,
        n_print: int,
        *,
        langevin: bool,
        seed: int,
        server_kwargs: dict | None,
        tmp_working_directory: str | None,
    ) -> tuple[Atoms, dict]:
        """Execute the simulation SHIK protocol."""
        # Stage 1: heating from 300 to 5000 K for 100 ps
        structure, _ = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,  # 5000 K
            n_ionic_steps=heating_steps,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=temperature_high,
            pressure=None,  # NVT ensemble
            langevin=langevin,
            seed=seed,
            server_kwargs=server_kwargs,
        )

        exclude_patterns = [
            "fix langevin all langevin 5000 5000 0.01 48279",
            "fix ensemble all nve/limit 0.5",
            "run 10000",
            "unfix langevin",
            "unfix ensemble",
        ]

        potential["Config"] = potential["Config"].apply(
            lambda lines: [line for line in lines if not any(p in line for p in exclude_patterns)]
        )

        # Stage 2: NVT equilibration at 5000 K for 100 ps
        structure, _ = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,  # 5000 K
            n_ionic_steps=int(100_000 / timestep),  # 100 ps / (1 fs timestep) = 1e5 steps
            timestep=timestep,
            n_print=n_print,
            initial_temperature=temperature_high,
            pressure=None,  # NVT ensemble
            langevin=langevin,
            seed=seed,
            server_kwargs=server_kwargs,
        )

        # Stage 3: NPT equilibration at 5000 K and 0.1 GPa for 700 ps
        structure, _ = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,
            n_ionic_steps=int(700_000 / timestep),  # 700 ps
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=0.1,  # GPa
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # Stage 4: Quenching 5000 K -> 300 K in NPT
        structure, _ = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,
            temperature_end=temperature_low,
            n_ionic_steps=cooling_steps,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=[0.1, 0.0],  # ramp pressure from 0.1 -> 0 GPa
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # Stage 5: Annealing at 300 K and 0 GPa for 100 ps in NPT
        structure_final, parsed_output = self.runner(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            n_ionic_steps=int(100_000 / timestep),  # 100 ps
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=0.0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        return structure_final, parsed_output

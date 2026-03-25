"""Pydantic models for the amorphouspy API.

Defines request/response schemas for the ``/jobs`` and ``/glasses`` endpoints.
"""

from enum import StrEnum
from io import StringIO
from typing import Annotated, Literal

from ase import Atoms
from ase.io import read, write
from pydantic import BaseModel, Field, PlainSerializer, PlainValidator

from amorphouspy_api.composition import Composition

# ---------------------------------------------------------------------------
# ASE Atoms serialisation helpers (used by database & visualization)
# ---------------------------------------------------------------------------


def serialize_atoms(atoms: Atoms) -> str:
    """Serialize ASE Atoms to JSON string."""
    buf = StringIO()
    write(buf, atoms, format="json")
    return buf.getvalue()


def validate_atoms(v: Atoms | dict | str | None) -> Atoms | None:
    """Validate and convert input to ASE Atoms object."""
    if v is None:
        return None
    if isinstance(v, Atoms):
        return v
    if isinstance(v, dict):
        try:
            return Atoms(**v)
        except Exception as e:
            msg = f"Could not reconstruct Atoms from dict: {e}"
            raise ValueError(msg) from e
    if isinstance(v, str):
        try:
            return read(StringIO(v), format="json")
        except Exception as e:
            msg = f"Could not parse Atoms from string: {e}"
            raise ValueError(msg) from e
    msg = f"Expected ASE Atoms, dict, str, or None — got {type(v)}"
    raise TypeError(msg)


AtomsType = Annotated[
    Atoms | None,
    PlainValidator(validate_atoms),
    PlainSerializer(serialize_atoms, return_type=str, when_used="unless-none"),
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Potential(StrEnum):
    """Supported interatomic potentials."""

    pmmcs = "pmmcs"
    bjp = "bjp"
    shik = "shik"


class StepStatus(StrEnum):
    """Status of an individual pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(StrEnum):
    """Overall status of a simulation job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Analysis configurations (discriminated union)
# ---------------------------------------------------------------------------


class StructureAnalysis(BaseModel):
    """Configuration for structural analysis (RDF, coordination, bond angles)."""

    type: Literal["structure"] = "structure"
    rdf_cutoff: float = Field(default=8.0, description="RDF cutoff in Å")
    bin_width: float = Field(default=0.02, description="RDF bin width in Å")


class ElasticAnalysis(BaseModel):
    """Configuration for elastic moduli analysis."""

    type: Literal["elastic"] = "elastic"
    strain_magnitude: float = Field(default=0.01, description="Applied strain for elastic constants")
    n_steps: int = Field(default=10000, description="Equilibration steps before measurement")


class ViscosityAnalysis(BaseModel):
    """Configuration for viscosity analysis (Green-Kubo).

    Viscosity is computed by running additional MD production runs at each
    requested temperature.  The melt-quench structure is sequentially cooled
    from high to low temperature, and at each step a Green-Kubo viscosity
    calculation is performed.
    """

    type: Literal["viscosity"] = "viscosity"
    temperatures: list[float] = Field(default=[1500, 2000, 2500], description="Simulation temperatures in K")
    timestep: float = Field(default=1.0, description="MD timestep in fs for the viscosity production run")
    n_timesteps: int = Field(default=10_000_000, description="Number of MD steps per viscosity production run")
    n_print: int = Field(default=1, description="Thermodynamic output frequency")
    max_lag: int | None = Field(
        default=1_000_000,
        description="Maximum correlation lag (steps) for Green-Kubo post-processing; None uses full trajectory",
    )


class CTEAnalysis(BaseModel):
    """Configuration for coefficient of thermal expansion analysis."""

    type: Literal["cte"] = "cte"
    temp_range: tuple[float, float] = Field(default=(300, 900), description="Temperature range in K")
    heating_rate: float = Field(default=1e12, description="Heating rate in K/s")


Analysis = Annotated[
    StructureAnalysis | ElasticAnalysis | ViscosityAnalysis | CTEAnalysis,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Viscosity result data (stored inside result_data["viscosity"])
# ---------------------------------------------------------------------------


class ViscosityResultData(BaseModel):
    """Result of a multi-temperature viscosity analysis."""

    temperatures: list[float] = Field(..., description="Simulation temperatures (K)")
    viscosities: list[float] = Field(..., description="Viscosities at each temperature (Pa·s)")
    max_lag: list[float] = Field(..., description="Max cutoff correlation time per temperature (ps)")
    simulation_steps: list[int] = Field(..., description="MD steps per temperature")
    lag_times_ps: list[list[float]] = Field(default_factory=list, description="Lag time arrays per temperature (ps)")
    sacf_data: list[list[float]] = Field(default_factory=list, description="Averaged normalised SACF per temperature")
    viscosity_running: list[list[float]] = Field(
        default_factory=list, description="Running viscosity integral per temperature (Pa·s)"
    )


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------


class MeltQuenchParams(BaseModel):
    """Parameters for the melt-quench MD simulation."""

    melt_temperature: float = Field(default=5000, description="Melt temperature in K")
    quench_rate: float = Field(default=1e12, description="Quench rate in K/s")
    n_atoms: int = Field(default=3000, description="Number of atoms")
    timestep: float = Field(default=1.0, description="MD timestep in fs")


# ---------------------------------------------------------------------------
# Job submission / response
# ---------------------------------------------------------------------------


class JobSubmission(BaseModel):
    """Request body for ``POST /jobs``."""

    composition: Composition = Field(
        ...,
        description=(
            "Oxide glass composition as a mapping of oxide formula to mol%. "
            "Values are rescaled to sum to 100%. "
            "Example: {'SiO2': 70, 'Na2O': 15, 'CaO': 15}"
        ),
    )
    potential: Potential = Field(default=Potential.pmmcs)
    simulation: MeltQuenchParams = Field(default_factory=MeltQuenchParams)
    analyses: list[Analysis] = Field(
        default_factory=lambda: [StructureAnalysis()],
        description="Analyses to run. Each can carry its own parameters.",
    )


class JobCreatedResponse(BaseModel):
    """Response for ``POST /jobs``."""

    id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(default=JobStatus.PENDING)
    composition: Composition
    potential: Potential
    created_at: str


class JobProgress(BaseModel):
    """Per-step progress for ``GET /jobs/{id}``."""

    structure_generation: StepStatus = StepStatus.PENDING
    melt_quench: StepStatus = StepStatus.PENDING
    structure_analysis: StepStatus = StepStatus.PENDING
    analyses: dict[str, StepStatus] = Field(
        default_factory=dict,
        description="Progress of additional analyses (viscosity, cte, elastic, …)",
    )


class JobStatusResponse(BaseModel):
    """Response for ``GET /jobs/{id}``."""

    id: str
    status: JobStatus
    composition: Composition
    potential: Potential
    progress: JobProgress
    errors: dict[str, str] = Field(default_factory=dict)
    created_at: str
    completed_at: str | None = None


class JobResultsResponse(BaseModel):
    """Response for ``GET /jobs/{id}/results``."""

    job_id: str
    composition: Composition
    structure: dict | None = None
    analyses: dict[str, dict] = Field(
        default_factory=dict,
        description="Results keyed by analysis type (viscosity, cte, elastic, …)",
    )


class JobSearchRequest(BaseModel):
    """Request body for ``POST /jobs:search``."""

    composition: Composition = Field(
        ...,
        description=(
            "Oxide glass composition as a mapping of oxide formula to mol%. "
            "Values are rescaled to sum to 100%. "
            "Example: {'SiO2': 70, 'Na2O': 15, 'CaO': 15}"
        ),
    )
    potential: Potential | None = None
    analyses: list[str] | None = None


class JobSearchMatch(BaseModel):
    """A single match from a job search."""

    job_id: str
    composition: Composition
    potential: Potential
    analyses: list[str]
    similarity: float = 1.0
    completed_at: str | None = None


class JobSearchResponse(BaseModel):
    """Response for ``POST /jobs:search``."""

    matches: list[JobSearchMatch]


# ---------------------------------------------------------------------------
# Glasses (materials) layer
# ---------------------------------------------------------------------------


class GlassSummary(BaseModel):
    """Summary entry for one glass composition."""

    composition: Composition
    n_jobs: int


class GlassListResponse(BaseModel):
    """Response for ``GET /glasses``."""

    glasses: list[GlassSummary]


class GlassPropertySource(BaseModel):
    """Provenance info linking a property back to its source job."""

    source_job: str
    potential: Potential
    computed_at: str | None = None


class AvailableStructure(BaseModel):
    """A quenched structure available for download."""

    job_id: str
    potential: Potential
    n_atoms: int


class GlassLookupRequest(BaseModel):
    """Request body for ``POST /glasses:lookup``."""

    composition: Composition = Field(
        ...,
        description=(
            "Oxide glass composition as a mapping of oxide formula to mol%. "
            "Example: {'SiO2': 70, 'Na2O': 15, 'CaO': 15}"
        ),
    )


class GlassPropertiesResponse(BaseModel):
    """Aggregated properties for ``POST /glasses:lookup``."""

    composition: Composition
    properties: dict[str, dict] = Field(default_factory=dict)
    available_structures: list[AvailableStructure] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)

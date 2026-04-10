"""Pydantic models for the amorphouspy API.

Defines request/response schemas for the ``/jobs`` and ``/glasses`` endpoints.
"""

from enum import StrEnum
from io import StringIO
from typing import Annotated, Literal

from ase import Atoms
from ase.io import read, write
from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    PlainSerializer,
    PlainValidator,
    RootModel,
    Tag,
)

# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _fmt_value(v: float) -> str:
    rounded = round(v, 2)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:g}"


class Composition(RootModel[dict[str, float]]):
    """Oxide glass composition (mol%).

    Accepts and serialises as a plain ``dict[str, float]``.
    Values represent mol% and will be rescaled to sum to 100% where needed.

    Examples:
        >>> c = Composition({"Na2O": 15, "SiO2": 70, "CaO": 15})
    >>> c.canonical
    'CaO 15 - Na2O 15 - SiO2 70'
    """

    @property
    def canonical(self) -> str:
        """Canonical string for DB storage and exact-match comparison.

        Components sorted alphabetically; values rounded to 2 dp,
        trailing zeros stripped.
        """
        components = sorted(self.root.items())
        return " - ".join(f"{oxide} {_fmt_value(val)}" for oxide, val in components)

    @classmethod
    def from_canonical(cls, canonical: str) -> "Composition":
        """Construct from a canonical DB string.

        >>> Composition.from_canonical("CaO 15 - Na2O 15 - SiO2 70")
        Composition({'CaO': 15.0, 'Na2O': 15.0, 'SiO2': 70.0})
        """
        result: dict[str, float] = {}
        for part in canonical.split(" - "):
            token = part.strip()
            if not token:
                continue
            oxide, value_str = token.rsplit(" ", 1)
            result[oxide] = float(value_str)
        return cls(result)


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

    type: Literal["structure_characterization"] = "structure_characterization"
    rdf_cutoff: float = Field(default=8.0, description="RDF cutoff in Å")
    bin_width: float = Field(default=0.02, description="RDF bin width in Å")


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
    n_timesteps: int = Field(
        default=10_000_000,
        description="Number of MD steps per viscosity production run",
    )
    n_print: int = Field(default=1, description="Thermodynamic output frequency")
    max_lag: int | None = Field(
        default=1_000_000,
        description="Maximum correlation lag (steps) for Green-Kubo post-processing; None uses full trajectory",
    )


class ElasticAnalysis(BaseModel):
    """Configuration for elastic moduli analysis (stress-strain finite differences).

    Calculates the full Cij stiffness tensor via central differences and
    derives isotropic moduli (B, G, E, nu) using Voigt-Reuss-Hill averaging.
    """

    type: Literal["elastic"] = "elastic"
    temperature: float = Field(default=300.0, description="Simulation temperature in K")
    pressure: float | None = Field(default=None, description="Target pressure for equilibration (None = NVT)")
    timestep: float = Field(default=1.0, description="MD timestep in fs")
    equilibration_steps: int = Field(default=1_000_000, description="Equilibration MD steps")
    production_steps: int = Field(default=10_000, description="Production MD steps per strain direction")
    n_print: int = Field(default=1, description="Thermodynamic output frequency")
    strain: float = Field(default=1e-3, description="Strain magnitude for finite differences")


class _CTEBase(BaseModel):
    """Shared simulation parameters for both CTE methods."""

    type: Literal["cte"] = "cte"
    pressure: float = Field(default=1e-4, description="Target pressure in GPa (default ≈ 1 bar)")
    timestep: float = Field(default=1.0, description="MD timestep in fs")
    equilibration_steps: int = Field(default=100_000, description="Equilibration MD steps")
    production_steps: int = Field(default=200_000, description="Production MD steps per run")


class CTEFluctuations(_CTEBase):
    """CTE via enthalpy-volume fluctuations at a single temperature.

    Iteratively runs production MD until convergence criteria are met,
    returning CTE values with uncertainty estimates.
    """

    method: Literal["fluctuations"] = "fluctuations"
    temperature: float = Field(default=300.0, description="Simulation temperature in K")
    min_production_runs: int = Field(
        default=2,
        description="Minimum production runs before convergence check",
    )
    max_production_runs: int = Field(
        default=25,
        description="Maximum production runs",
    )
    cte_uncertainty_criterion: float = Field(
        default=1e-6,
        description="Convergence criterion for linear CTE uncertainty in 1/K",
    )


class CTETemperatureScan(_CTEBase):
    """CTE via NPT production runs at multiple temperatures.

    Returns raw volume / box-length data at each temperature for
    user-side CTE fitting (e.g. linear or polynomial V-T fit).
    """

    method: Literal["temperature_scan"] = "temperature_scan"
    temperatures: list[float] = Field(
        default=[300, 400, 500, 600],
        description="Temperatures in K",
    )


CTEAnalysis = Annotated[
    CTEFluctuations | CTETemperatureScan,
    Field(discriminator="method"),
]


def _analysis_tag(v: object) -> str:
    """Return a unique tag for each Analysis variant.

    Most types are identified by their ``type`` field alone.  CTE variants
    share ``type="cte"`` and are further distinguished by ``method``.
    """
    if isinstance(v, dict):
        t = v.get("type", "")
        if t == "cte":
            return f"cte_{v.get('method', 'fluctuations')}"
        return t
    t = getattr(v, "type", "")
    if t == "cte":
        return f"cte_{getattr(v, 'method', 'fluctuations')}"
    return t


Analysis = Annotated[
    Annotated[StructureAnalysis, Tag("structure_characterization")]
    | Annotated[ViscosityAnalysis, Tag("viscosity")]
    | Annotated[ElasticAnalysis, Tag("elastic")]
    | Annotated[CTEFluctuations, Tag("cte_fluctuations")]
    | Annotated[CTETemperatureScan, Tag("cte_temperature_scan")],
    Discriminator(_analysis_tag),
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
    lag_times_ps: list[list[float]] = Field(
        default_factory=list, description="Downsampled lag time arrays per temperature (ps)"
    )
    viscosity_integral: list[list[float]] = Field(
        default_factory=list,
        description="Cumulative viscosity integral per temperature (Pa·s)",
    )


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------


class MeltQuenchParams(BaseModel):
    """Parameters for the melt-quench MD simulation."""

    melt_temperature: float = Field(
        default=None,
        description="Melt temperature in K. If None, protocol-specific melt temperature is used. ",
    )
    quench_rate: float = Field(default=1e12, description="Quench rate in K/s")
    n_atoms: int = Field(default=6000, description="Number of atoms")
    timestep: float = Field(default=1.0, description="MD timestep in fs")
    equilibration_steps: int | None = Field(
        default=None,
        description="Override for all fixed equilibration stages. If None, protocol-specific defaults are used.",
    )


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
        default_factory=lambda: [StructureAnalysis(), ViscosityAnalysis(), CTEFluctuations(), ElasticAnalysis()],
        description="Analyses to run. Each can carry its own parameters. Defaults to all available analyses.",
    )


def _job_urls(job_id: str) -> dict[str, str]:
    """Build user-facing URLs for a job.

    Uses the ``API_BASE_URL`` environment variable.  When unset the URLs
    will contain relative paths only (empty base).
    """
    from amorphouspy_api.config import API_BASE_URL

    base = API_BASE_URL.rstrip("/")
    return {
        "status": f"{base}/jobs/{job_id}",
        "results": f"{base}/jobs/{job_id}/results",
        "visualization": f"{base}/jobs/{job_id}/visualize",
        "structure": f"{base}/jobs/{job_id}/structure",
    }


class JobCreatedResponse(BaseModel):
    """Response for ``POST /jobs``."""

    id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(default=JobStatus.PENDING)
    composition: Composition
    potential: Potential
    created_at: str
    urls: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Useful URLs for this job: 'status' to poll progress, "
            "'results' for analysis data, 'visualization' for an interactive "
            "HTML dashboard, 'structure' to download the quenched structure."
        ),
    )


class JobProgress(BaseModel):
    """Per-step progress for ``GET /jobs/{id}``."""

    structure_generation: StepStatus = StepStatus.PENDING
    melt_quench: StepStatus = StepStatus.PENDING
    analyses: dict[str, StepStatus] = Field(
        default_factory=dict,
        description="Progress of each analysis (structure_characterization, viscosity, cte, elastic, …)",
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
    urls: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Useful URLs for this job: 'status' to poll progress, "
            "'results' for analysis data, 'visualization' for an interactive "
            "HTML dashboard, 'structure' to download the quenched structure."
        ),
    )


class JobResultsResponse(BaseModel):
    """Response for ``GET /jobs/{id}/results``."""

    job_id: str
    composition: Composition
    analyses: dict[str, dict] = Field(
        default_factory=dict,
        description="Results keyed by analysis type (structure_characterization, viscosity, cte, elastic, …)",
    )
    visualization_url: str = Field(
        default="",
        description="URL for an interactive HTML visualization dashboard of these results.",
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
    threshold: float = Field(
        default=0.05,
        description=(
            "Maximum Euclidean distance in elemental atom-fraction space for a "
            "composition to be included as a close match. Set to 0 for exact "
            "matches only. Typical values: 0.01-0.1."
        ),
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of close matches to return.",
    )


class JobSearchMatch(BaseModel):
    """A single match from a job search."""

    job_id: str
    composition: Composition
    potential: Potential
    analyses: list[str]
    similarity: float = 1.0
    match_type: str = Field(
        default="exact",
        description="'exact' for identical composition, 'close' for nearby.",
    )
    distance: float = Field(
        default=0.0,
        description="Euclidean distance in elemental atom-fraction space (0 for exact matches).",
    )
    completed_at: str | None = None
    visualization_url: str = Field(
        default="",
        description="URL for an interactive HTML visualization dashboard of this job's results.",
    )


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
    visualization_url: str = Field(
        default="",
        description="URL for an interactive HTML visualization dashboard of this job's results.",
    )


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

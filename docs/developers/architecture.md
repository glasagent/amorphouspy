# Architecture

For more information on the internal workings, this section provides an overview of how `amorphouspy` is organized internally.

## Workflow Overview

The following diagram shows the full simulation pipeline, from user input through workflows to output properties:

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 60}} }%%
flowchart LR

   subgraph "User Input"
      UserStructure[User Structure]
      Database["Internal/External<br/>Database"]
      InputStructure["Initial<br/>Structure"]
      AdhocGeneration["Structure Generation"]
      Composition["Composition"]
      Density["Density"]
   end


    TemperatureProgram["Temperature<br/>Program"]
    Strain[Strain/Stress]
    GenericSimulationSettings["Generic Simulation<br/>Settings"]
    Others[...]
    FF[Interatomic Potential]

   subgraph "Workflows"
      WorkflowSettings["Workflow<br/>Settings"]
      MeltQuench["Melt-Quench<br/>Simulation"]
      StructureAnalysis["Structure<br/>Analysis"]
      ElasticModuliSimulation["Elastic Moduli<br/>Simulation"]
      ViscositySimulation["Viscosity<br/>Simulation"]
      CTESimulation["CTE<br/>Simulation"]
      AnisotropyAnalysis["Anisotropy<br/>Analysis"]
      OthersWorkflow[...]
   end

   subgraph "Output"
      GlassStructure["Glass<br/>Structure"]
      RDF[RDF]
      BondAngleDistribution["Bond Angle<br/>Distribution"]
      QnDistribution["Qn Values"]
      NetworkAnalysis["Network<br/>Analysis"]
      ElasticModuli["Elastic<br/>Moduli"]
      Viscosity[Viscosity]
      CTE[CTE]
      Anisotropy[Anisotropy]
      OthersOutput[...]
   end

   subgraph "Legend"
      Future[Future]
      Underway[Underway]
      Implemented[Implemented]
      Validated[Validated]
   end


   
   UserStructure --> InputStructure
   Database --> InputStructure
   AdhocGeneration --> InputStructure
   Density --> AdhocGeneration
   Composition --> AdhocGeneration
   
   InputStructure --> WorkflowSettings
   FF --> WorkflowSettings
   GenericSimulationSettings --> WorkflowSettings
   TemperatureProgram --> WorkflowSettings
   Strain --> WorkflowSettings
   Others --> WorkflowSettings
   

   GlassStructure --> InputStructure
   WorkflowSettings --> MeltQuench --> GlassStructure 
   WorkflowSettings --> StructureAnalysis
   WorkflowSettings --> ElasticModuliSimulation --> ElasticModuli
   WorkflowSettings --> ViscositySimulation --> Viscosity
   WorkflowSettings --> CTESimulation --> CTE
   WorkflowSettings --> AnisotropyAnalysis --> Anisotropy
   WorkflowSettings --> OthersWorkflow --> OthersOutput
   StructureAnalysis --> RDF
   StructureAnalysis --> BondAngleDistribution
   StructureAnalysis --> NetworkAnalysis
   StructureAnalysis --> QnDistribution 


%% Styling
   classDef future fill:#ea580c,stroke:#f97316,stroke-width:2px,color:#fff,font-weight:bold,font-size:22px;
   classDef implemented fill:#bbf7d0,stroke:#10b981,stroke-width:2px,color:#166534,font-weight:bold,font-size:22px;
   classDef validated fill:#059669,stroke:#10b981,stroke-width:2px,color:#fff,font-weight:bold,font-size:22px;
   classDef underway fill:#7c3aed,stroke:#8b5cf6,stroke-width:2px,color:#fff,font-weight:bold,font-size:22px;
   classDef none fill:#333333,stroke:#ffffff,stroke-width:2px,color:#fff,font-weight:bold,font-size:22px;


class Implemented,Density,Composition,AdhocGeneration,InputStructure,TemperatureProgram,WorkflowSettings,StructureAnalysis,NetworkAnalysis,BondAngleDistribution,RDF,QnDistribution,MeltQuench,GlassStructure,UserStructure,SystemSize,GenericSimulationSettings implemented
class Validated validated
class Underway,ViscositySimulation,ElasticModuliSimulation,Strain,FF underway
class Future,CTESimulation,AnisotropyAnalysis,Database, future
class ElasticModuli,Viscosity,CTE,Anisotropy,Others,OthersWorkflow,OthersOutput none
```

## Package Organization

```
amorphouspy/
├── structure.py          # Composition parsing, structure generation, density model
├── mass.py               # Atomic mass utilities (wraps ASE data)
├── neighbors.py          # Cell-list neighbor search with periodic boundary conditions
├── io_utils.py           # LAMMPS I/O, XYZ writer, ASE Atoms helpers
├── shared.py             # Element type mapping, distribution counting utilities
├── potentials/
│   ├── potential.py      # Unified potential generator interface
│   ├── pmmcs_potential.py  # Pedone (PMMCS) Morse + Coulomb
│   ├── bjp_potential.py    # Bouhadja Born-Mayer-Huggins + Coulomb
│   └── shik_potential.py   # SHIK Buckingham + r⁻²⁴ + Coulomb
├── analysis/
│   ├── radial_distribution_functions.py  # RDF g(r) and coordination n(r)
│   ├── qn_network_connectivity.py        # Qⁿ distribution and network connectivity
│   ├── bond_angle_distribution.py        # O-X-O and X-O-X bond angle histograms
│   ├── rings.py                          # Guttman ring statistics (via sovapy)
│   ├── cavities.py                       # Void/cavity volume analysis (via sovapy)
│   └── cte.py                            # CTE from NPT fluctuations
└── workflows/
    ├── meltquench.py           # Core melt-quench simulation logic
    ├── meltquench_protocols.py # Potential-specific multi-stage protocols
    ├── md.py                   # Single-point NVT/NPT molecular dynamics
    ├── elastic_mod.py          # Elastic moduli via stress-strain finite differences
    ├── viscosity.py            # Viscosity via Green-Kubo (SACF integration)
    ├── cte.py                  # CTE simulation with convergence checking
    ├── structural_analysis.py  # Comprehensive analysis pipeline + Plotly plotting
    └── shared.py               # LAMMPS command builder utility
```

# amorphouspy

This project provides workflows to perform atomistic simulations for glasses.
This concerns, but is not limited to, generating initial structural models, performing melt-quenching simulations as well as analysing relevant properties. 

Later, these workflows serve as blueprints intended to be used by "Otto", the GlasAgent. 

## Contents

- `amorphouspy`: Workflows for atomistic modeling of oxide glasses
- `notebooks`: Jupyter notebooks for atomistic modeling of oxide glasses	

## Installation

```
conda env create -f environment.yml
conda activate amorphouspy
pip install -e amorphouspy
jupyter notebook notebooks/Meltquench.ipynb
```

## Developer setup
In addition, install
```
pip install -r amorphouspy/requirements-dev.txt
pre-commit install
```


## Upcoming milestones

- MS12 (end of 2025): Integration of existing GlasDigital workflows (DFT and classical MD) for determining density and elastic moduli
- MS12 (end of 2025): Workflows (classical MD) to determine high-temperature viscosity (TODO) and generation of structural models via melt-quenching (DONE)

----------------------------------------------------------

- MS42 (end of June 2028): Feature generation for ML and semi-empirical models based on glass structure​

- MS42 (end of June 2028): State-of-the art MLIP available​ (Testing started)
- MS42 (end of June 2028): Workflows for complex property analyses​. 
  - Qn values (DONE/TODO)
  - RDFs (DONE/TODO)
  - Network analysis (DONE/TODO)
  - Anisotropy analysis (TODO)
- MS60 (end of 2030): Demonstrator ready​

## Material systems to start with (proposed by Leopold (@ltalirz))

First, as an easy start we can work with crystalline materials that also exist as glasses:
1. NaAlSi3​O8​ (Albite)
2. CaAl2​Si2​O8​ (Anorthite)
3. CaB2​Si2​O8​ (Danburite)
4. (Mg,Fe)2​Al4​Si5​O18​ (Cordierite; NB: Achraf mentioned that Fe might be tricky due to different charge states)

Later, more complicated glasses from Schott can be considered. The following are only the approximate compositions, taken from the internet:

5. DGG3:
   - SiO2​ (~78.56%)
   - B2​O3​ (~12.7%)
   - Al2O3 (~2.76%)
   - Na2​O (~3.43%)
   - K2O (~0.94%)

6. FIOLAX clear:
   - SiO2​ (~75%)
   - B2​O3​ (~10.5%)
   - Na2​O (~7%)
   - Al2​O3​ (~5%)
   - CaO (~1.5%)

# Approximate amorphouspy workflow diagram 

```mermaid
%% Mermaid live editor: https://mermaid-js.github.io/mermaid-live-editor

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
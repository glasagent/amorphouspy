# pyiron workflows for atomistic modeling of oxide glasses

This project aims at setting up pyiron-workflows to perform atomistic simulations for glasses.
This concerns, but is not limited to, generating initial structural models, performing melt-quenching simulations as well as analysing relevant properties. 

Later, these workflows serve as blueprints intended to be used by "Otto", the GlasAgent. 

## Contents

- `pyiron_glass`: pyiron workflows for atomistic modeling of oxide glasses
- `notebooks`: Jupyter notebooks for atomistic modeling of oxide glasses	

## Installation

```
conda env create -f environment.yml
conda activate pyiron-glass
pip install -e pyiron_glass
jupyter notebook notebooks/Meltquench.ipynb
```

## Developer setup
In addition, install
```
pip install -r pyiron_glass/requirements-dev.txt
pre-commit install
```


## Upcoming milestones

- Integration of existing GlasDigital workflows (DFT and classical MD) for determining density and elastic moduli into current pyiorn version and adaptation to glasses (BAM2,MS12)
- Defining pyiron workflows (classical MD) to determine high-temperature viscosity and generation of structural models via melt-quenching (BAM2 with input from Schott, MS12)

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

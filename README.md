# Scope of the project
This project aims at setting up pyiron-workflows to perform atomistic simulations for glasses.
This concerns, but is not limited to, generating initial structural models, performing melt-quenching simulations as well as analysing relevant properties. 
Later, these workflows serve as blueprints intended to be used by "Otto". 

# Current To Do's
In the "GlasAgent Atomistik" meeting on 14.04.2025 with Jan, Achraf, Tilmann, Leopold and Marcel we decided on the following first steps:
- Achraf (@Atilaac) sets up a first jupyter notebook or python script with the workflows he typically used so far to get things started
- Jan (@jan-janssen) checks how this can be translated to pyiron (at best, some parts of this serve as templates for following workflows/simulation procedures)

# Update by Achraf (@Atilaac) 30.04.2025
- I uploaded some scripts that I made this week.
- The scripts are to be described below:
   - Folder preLAMMPS have an example of a LAMMPS script used to create a lithium aluminosilicate glass. The SHIK potential is used with a slight modification of using DSF model for Coulomb instead of wolf used in the original SHIK. This should not lead to significate changes in the properties and here used to show an example. Scientific robustness of the results is to be discussed at a later stage. The folder contain a jupyter notebook "GlassMaking.ipynb" and another forlder called POTFILES that have the potential table generated for using SHIK parameters.

   - Folder pyiron contains the jupyter notebook "Meltquench.ipynb" and the output of the simulation I performed using the parameters from Pedone JPCB 2006 (DSF is used for Coulomb as well). The I used a small sample with a 100 molecules and the number of steps is reduced a lot for testing puposes and not to be used in a real simulation. 
   - The pyiron jupyter notebook, have many functions that take a glass composition and create a cubic box with atoms inside that corresponds to 100 molecules. The number of atoms and box size are calculated automatically from the composition and density.
   - Jan (@jan-janssen) should definetly have a look at the provided script and will have a better way to translate the pureLAMMPS script to the pyiron properly.

- I have more comments inside the box and I tried my best to make the code as readable and structured as possible. Moreover, I created bunch of function to make the code reusable elsewhere as well.
- pyiron atomistics and lammmps must be installed before running the scripts.

# General aspects
- The code should have a good documentation
- As soon as it becomes necessary, tests should be implemented

# Upcoming milestones
- Integration of existing GlasDigital workflows (DFT and classical MD) for determining density and elastic moduli into current pyiorn version and adaptation to glasses (BAM2,MS12)
- Defining pyiron workflows (classical MD) to determine high-temperature viscosity and generation of structural models via melt-quenching (BAM2 with input from Schott, MS12)

# Material systems to start with (proposed by Leopold (@ltalirz))
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

# FitAndPlot
`FitAndPlot` is a routine for fitting and plotting dark matter halo profiles stored in HDF5 files obtained from processing of ROCKSTAR halo finder output.
The code fits binned halo profiles with a 3D and 2D NFW model and a 3D gNFW model and computes the following:

- Distributions of the fit parameters NFW and gNFW $r_{200}$, $r_s$, $\gamma$.
- 3D stacked profiles of mass, density, circular velocity, velocity dispersion and velocity anisotropy.
- 2D stacked profiles of mass, density and surface density excess.
- Concentration-mass relation using both the 3D and 2D NFW fits.
- Halo sparsities and their relation with concentration and mass.

The code can be used to analyze simulations of multiple cosmological models at a time, and it can fit 3D mass, density and circular velocity, along with projected mass and density, profiles with NFW and gNFW models. The **\"Global code parameters\"** section allows to change global settings such as the location of the hdf5 files, the number of cosmological models to consider and the folder for saving all generated plots. The `HaloReadH5.py` module contains all individual reading and fitting functions.

The $R_{500}$ and $R_{200}$ from the simulation and halo IDs corresponding to the saved fits are also saved.

## Basic functioning

**The HDF5 files corresponding to each simulation must be arranged in a specific directory structure, where each file is contained in a folder named after the simulation, including folders for each region considered. The name of the HDF5 files themselves does not matter, but the names of the directories do.**

For example, to consider three models LCDM, Model_1 and Model_2, including two different simulation regions Region_1 and Region_2, then the HDF5 files must be positioned in the corresponding simulation folders as follows:

    hdf5_folder/simulation_name
    ├───Region_1
    │   ├───LCDM
    │   │   └───LCDM_file_R1.hdf5
    │   ├───Model_1
    │   │   └───M1_file_R1.hdf5
    │   └───Model_2
    │       └───M2_file_R1.hdf5
    └───Region_2
        ├───LCDM
        │   └───LCDM_file_R2.hdf5
        ├───Model_1
        │   └───M1_file_R2.hdf5
        └───Model_2
            └───M2_file_R2.hdf5

Once the HDF5 file reading and fitting is completed, all results are stored in nested dictionaries.

The **halo_profiles** dictionary contains all halo binned profiles, including their Poissonian uncertainties and the radial bin centers, for both the 3D case and all supplied 2D projections. 
The dictionary is structured as follows:

    halo_profiles
    └───sim_type
        ├───3D
        │   └───BINNED_PROFILE_3D
        └───2D
            └───projections
                └───BINNED_PROFILE_2D

    Possible key values:
    --------------------
    BINNED_PROFILE_3D = {"MASS", "DENSITY", "VCIRC", "BETA", "NUM", "CUM_NUM", "SIGMAr", "SIGMAt", "SIGMAp", "ERR_MASS", "ERR_DENSITY", "ERR_VCIRC", "R"}

    BINNED_PROFILE_2D = {"MASS", "DENSITY", "DEN_CUM", "DELTA_SIGMA", "NUM", "CUM_NUM", "ERR_MASS", "ERR_DENSITY", "ERR_DEN_CUM", "ERR_DSIGMA", "R"}

The **fit_pars** and **fit_cov** contain all halo best-fit parameters and covariances respectively, for each profile model and fit quantity supplied, and for both the 3D and 2D fits. 

**The fit parameters are returned by the dictionaries in log $-$ except for chi2 and M200 $-$ and the covariances are those of the log parameters**.

The dictionary is structured as follows:

    fit_pars/fit_cov
    └───sim_type
        ├───3D
        │   └───PROFILE_MODEL
        │       └───FIT_QUANTITY_3D
        │           └───fit_parameter/fit_covariance             
        └───2D
            └───projections
                └───PROFILE_MODEL
                    └───FIT_QUANTITY_2D
                        └───fit_parameter/fit_covariance

    Possible key values:
    --------------------
    PROFILE_MODEL = {"NFW", "gNFW"}
    FIT_QUANTITY_3D = {"MASS", "DENSITY", "VCIRC"}
    FIT_QUANTITY_2D = {"MASS", "DENSITY"}

    For NFW profile:
    -   fit_parameter = {"r200", "rs", "M200", "chi2"}
    -   fit_covariance = {"r200", "rs", "r200_rs"}   

    For gNFW profile:
    -   fit_parameter = {"r200", "rs", "gamma", "M200", "chi2"}
    -   fit_covariance = {"r200", "rs", "r200_rs", "r200_gamma", "rs_gamma"}  

Finally, the **halo_props** and **sim_props** are dictionaries containing properties for each halo and simulation properties, respectively:

    - halo_props[sim_type].keys() = {"ID", "REGION", "R500", "R200"}

    - sim_props[sim_type].keys() = {"HALO_NUM_TOT", "HALO_NUM_REGION", "MPART", "COSM_PARS"}

**All halo profiles and fit parameters are saved in physical units, except the 3D and 2D radial bins which are in units of $r_{500}$. All lengths and masses are in Mpc and M$_\odot$ respectively, except for the velocity dispersions and circular velocities, which are measured in km/s.**

## Enable savestates
If a simulation is composed of multiple HDF5 files, the routine can be instructed to only read up to a certain number of files, specified in the global parameters section. Furthermore, each batch of profiles and fit parameters obtained by files read in this way will be saved to a "progress" folder, to keep track of the files already read and to backup the fit progress. 

## Multiprocessing
`FitAndPlot` also possesses basic multiprocessing functionality: if multiple simulation types are supplied, the reading and fitting of each one can be assigned to different Python processes to allow for parallel computation.
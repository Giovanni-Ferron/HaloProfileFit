# HaloProfileFit
`HaloProfileFit` is a Python3 routine for fitting and plotting dark matter halo profiles stored in HDF5 files.

By default, the code fits halo mass profiles with 3D and 2D, NFW and gNFW models, and computes the following:

- Distributions of the fit parameters NFW and gNFW $r_{200c}$, $r_s$, $\gamma$.
- 3D stacked profiles of mass, density, circular velocity, velocity dispersion components and velocity anisotropy.
- 2D stacked profiles of mass, density and surface density excess.
- Concentration-mass relation and sparsities.

The `FitAndPlot.ipynb` notebook can be used as a starting point to analyze halo profiles from multiple cosmological simulations at a time, and it can be used to fit 3D mass, density and circular velocity, along with projected mass and density profiles with NFW and gNFW models. 

The **\"Global code parameters\"** section allows to change global settings such as the location of the hdf5 files, the number of cosmological models to consider, the type of profile models and halo profile quantities to fit, and the folder for saving all generated plots. 

The `HaloReadH5.py` module contains all individual reading and fitting functions, along with a HaloModel class which allows to add custom profile models for fitting.

## Basic functioning

**The HDF5 files corresponding to each simulation must be arranged in a specific directory structure, where each file is contained in a folder named after the simulation model (names in the sim_type list), including folders for each region considered (names in the files_to_read list). The name of the HDF5 files themselves does not matter, but the names of the directories do.**

For example, to consider three models LCDM, Model_1 and Model_2 (these names would be inserted into the sim_type list), including two different simulation regions Region_1 and Region_2, then the HDF5 files must be positioned in the corresponding simulation folders as follows:

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

Also note that the region folders do not necessarily have to correspond to actual simulation regions, but can be used for other kinds of subdivision: for example, they can represent the simulations at various redshifts, where each region folder corresponds to a certain snapshot.
Once the HDF5 file reading and fitting is completed, all results are stored in nested dictionaries.

**In `FitAndPlot.ibynb` all quantites stored in the HDF5 files are assumed to be in physical units, except for the radial bins and densities which are assumed to be in units of the halo $r_{500c}$ (change this using the scale_lengths argument in GetSimProfiles).**

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

**The dictionaries contain the log10 of the fitted free parameters $-$ except for chi2 and M200 $-$ and the covariances are those of the log10 parameters**.

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

Finally, the **halo_props** and **sim_props** are dictionaries containing properties for each halo and global simulation properties, respectively:

    - halo_props[sim_type].keys() = {"ID", "REGION", "R500", "R200", "M500", "M200", "MASS_TOT"}

    - sim_props[sim_type].keys() = {"HALO_NUM_TOT", "HALO_NUM_REGION", "MPART", "COSM_PARS"}

**All halo profiles and fit parameters are saved in physical units, except the 3D and 2D radial bins which are in units of $r_{500c}$. All overdensity radii and masses are computed with respect to the critical density of the Universe, rather than the background value. All lengths and masses are in Mpc and $\text{M}_\odot$ respectively, except for the velocity dispersions and circular velocities, which are measured in km/s.**

## Enable savestates
If a simulation is composed of many HDF5 files, the routine allows to select the number of files to read at a time, along with the region from which to read them, allowing to read all HDF5 files in batches.

Furthermore, each batch of profiles read in this way can optionally be saved to a "progress" folder, to keep track of the files already read and to backup the current progress. The routine can create savestates for multiple individual HDF5 files at once, saving all current progress into a single npz file containing all information stored in the halo_profiles, halo_props and sim_props dictionaries for that batch. Instead, the progress for the fitting is only saved on a per-dimension basis, that is, the routine can only save the current progress after completing the fitting for an individual dimension (i.e. 3D, 2Dx, 2Dy, 2Dz). The names of the HDF5 files already read are stored in txt files inside the savestates/filenames_done folder. 

Finally, since the routine keeps track of the progress, every time it is run it will continue reading or fitting from the last created savestate, until it will have read or fit every remaining halo.
If savestates are allowed, the routine will automatically read any created savestates instead of reading or fitting from the beginning,. Therefore, to make the code start reading or fitting the profiles from zero simply delete the "savestates" and "savestates_fits" folders from the "progress" directory.

## Multiprocessing
`HaloProfileFit` also possesses basic multiprocessing functionality: if multiple simulation types are supplied, the reading and fitting of each one can be assigned to different Python processes to allow for parallel computation.

## Modifying the code
The code can be freely and easily modified as needed, for example in order to include more halo fit models or change the HDF5 group names. To add a custom fit model it is most convenient to modify the HaloModel class, adding the required parameters of the custom model as done for the default NFW and gNFW cases. Instead, for modifications to the HDF5 reading or profile fitting it is best to see the GetProfiles and FitProfiles functions.   

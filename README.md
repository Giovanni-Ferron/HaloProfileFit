# FitAndPlot
`FitAndPlot` is a routine for fitting and plotting dark matter halo profiles stored in HDF5 files obtained from processing of ROCKSTAR halo finder output.
The code fits binned halo profiles with a 3D and 2D NFW model and a 3D gNFW model and computes the following:

- Distributions of the fit parameters $r_{200}$, $r_s$, $\\gamma$, $r_{-2}$.
- 3D stacked profiles of mass, density, circular velocity, velocity dispersion and velocity anisotropy.
- 2D stacked profiles of mass, density and density excess.
- Concentration-mass relation using both the 3D and 2D NFW fits.
- Halo sparsities and their relation with concentration and mass.

The code can be used to analyze simulations of multiple cosmological models at a time, and it can fit 3D mass, density and circular velocity profiles with NFW and gNFW. The 2D fits can only be performed with NFW, for both projected mass and projected density. The **\"Global code parameters\"** section allows to change settings such as the location of the hdf5 files, the number of cosmological models to consider, the fit bounds and the folder for saving all generated plots. The `MP_fitplot.py` module is used for multiprocessing, which can be enabled through the global code parameters. If enabled, multiple processes are opened to read and fit the HDF5 files of multiple different input simulations in parallel. If a simulation is composed of multiple HDF5 files, the routine can be instructed to only read up to a certain number of files, specified in the global parameters section. Furthermore, each batch of fits obtained by files read in this way will be saved to a "progress" folder, to keep track of the files already read and to backup the fit progress. 

The function that fits all the 3D profiles with NFW and gNFW models for a given simulation is FitSimulationData. By looping over every region and halo the parameters of the individual fits, their uncertainties and the fit chi2 are saved, but only if the NFW mass fit $M_{200}$ is greater or equal than a threshold mass, mThresh, and if the halo $r_s < r_{200}$. The 3D mass, density, circular velocity, velocity dispersion, velocity anisotropy and cumulative particle number profiles, their uncertainties and the bin radii are also saved. The fit parameters, chi2, uncertainties and profiles are saved in nested dictionaries where the keys correspond to the simulation type, dimension (for 2D only), profile model, profile type and parameters. For example, to access the 3D fit parameter "r200" from the model_pars dict, for the CDM model, gNFW mass profile: 

    par_name = model_pars["CDM"]["gNFW"]["MASS"]["r200"].

**The fit parameters $r_{200}$, $r_s$ and $\\gamma$ are returned by the dictionaries in log, and the covariances are those of the log parameters**.
- The parameters saved in **model_pars** and **model_pars_2D** have keys: **"r200", "rs", "gamma", "chi2", "M200"**.
- The uncertainties in **model_cov** and **model_cov_2D** have keys: **"r200", "rs", "gamma", "r200_rs", "r200_gamma", "rs_gamma"**.
- The binned 3D profiles in **model_profiles** have keys: **"MASS", "DENSITY", "VCIRC", "BETA", "NUM", "CUMNUM", "SIGMAr", "SIGMAt", "SIGMAp", "ERR_MASS", "ERR_DENSITY", "ERR_VCIRC", "R"**.
- The binned 2D profiles in **model_profiles_2D** have keys, for each dimension: **"MASS", "DENSITY", "DEN_CUM", "NUM", "CUMNUM", "ERR_MASS", "ERR_DENSITY", "ERR_DEN_CUM", "R"**.

The $R_{500}$ and $R_{200}$ from the simulation and halo IDs corresponding to the saved fits are also saved. **The parameters $r_{200}$, $r_s$, $R_{500}$ and $R_{200}$ are in Mpc, the bin radii R are in units of $r_{500}$.**

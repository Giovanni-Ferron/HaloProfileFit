import numpy as np
import scipy.special as spec
from scipy.optimize import curve_fit
import h5py
import os
import sys
import glob

#Define physical constants
G_mpc = 4.302e-9    #G in Msun^-1 * Mpc * km^2 * s^-2

#Function used to fit and compute the chi2
def FitAndChi2(func, data_x, data_y, y_err, par_0, par_bounds, method="trf"):
    """
    This function fits a given function to the provided data and computes the chi2.
    
    Args:
    -----
    func:     [function]
              Function to fit to the data
    data_x:   [float]
              Data for the independent variable
    data_y:   [float]
              Data for the dependent variable
    y_err:    [float]
              Uncertainties on the dependent variable
    par_0:     [array_like]
               Initial guess for each fit parameter
    par_bounds: [2-tuple of array_like]
                Lower and upper bounds on the fit parameters

    Returns:
    --------
    popt: [array]
          Best-fit parameters
    cov:  [2D array]
          Parameter covariance matrix
    chi2: [float]
          chi2 of the fit
    """
    
    popt, cov = curve_fit(func, data_x, data_y, p0=par_0, sigma=y_err, 
                          bounds=par_bounds, absolute_sigma=True, maxfev=10000, method=method)

    ndof = len(data_x) - len(popt)
    y_fit = func(data_x, *popt) 
    chi2 = np.sum(((data_y - y_fit) / y_err)**2) / ndof
    
    return popt, cov, chi2
    
    
#Function used to compute the 3D halo profile, either NFW or gNFW, mass, density or circular velocity
def HaloProfile(lr, lr200, lrs, lgamma, cosm_pars, profile_model="NFW", quantity_type="MASS"):
    """
    This function returns a halo mass or density profile at the specified radius, either NFW or gNFW.
    
    Args:
    -----
    r:     [float]
           Halo profile radii in Mpc
    lr200:  [float]
            Log of the halo r200 in Mpc
    lrs:    [float]
            Log of the alo scale radius in Mpc
    lgamma: [float]
            Log of the gNFW gamma exponent (it's 0. for NFW)
    cosm_pars:     [float array]
                   Array containing cosmological parameters (h, omega matter, omega lambda, redshift)
    profile_model: [string]
                   Profile model to return: NFW, GNFW
    quantity_type: [string]
                   Profile type to return: MASS, DENSITY

    Returns:
    --------
    profile: [float]
             Value of the selected profile at a given radius, in Msun for MASS profile, Msun/Mpc^3 for DENSITY profile and km/s for VCIRC profile
    """

    h, Om, Ol, z = cosm_pars
    H0 = 100 * h

    r = 10**lr
    r200 = 10**lr200
    rs = 10**lrs
    gamma = 10**lgamma
    
    c200 = r200 / rs
    x = r / rs
    H2z = H0**2 * (Om * (1 + z)**3 + Ol)
    M200 = 100 * H2z / G_mpc * r200**3
    fac200 = np.log(1 + c200) - c200 / (1 + c200)

    #Compute either mass, density or circular velocity for the NFW profile   
    if profile_model.upper() == "NFW":
        if quantity_type.upper() == "MASS":
            return (np.log(1 + x) - x / (1 + x)) / fac200 * M200

        elif quantity_type.upper() == "DENSITY":
            return M200 / (4 * np.pi * fac200) * (r * (rs + r)**2)**-1

        elif quantity_type.upper() == "VCIRC":
            return np.sqrt((np.log(1 + x) - x / (1 + x)) / fac200 * M200 * G_mpc / r)

    #Compute either mass, density or circular velocity for the gNFW profile
    elif profile_model.upper() == "GNFW":
        if quantity_type.upper() == "MASS":
            h2x = spec.hyp2f1(3-gamma, 3-gamma, 4-gamma, -r200/rs)
            h2y = spec.hyp2f1(3-gamma, 3-gamma, 4-gamma, -r/rs)

            return (r / r200)**(3 - gamma) * M200 * h2y / h2x

        elif quantity_type.upper() == "DENSITY":
            h2x = spec.hyp2f1(3-gamma, 3-gamma, 4-gamma, -r200/rs)
            rho_0 = M200 / (4 * np.pi * rs**3) * ((r200/rs)**(3 - gamma) / (3 - gamma) * h2x)**-1

            return rho_0 / (x**gamma * (1 + x)**(3 - gamma))

        elif quantity_type.upper() == "VCIRC":
            h2x = spec.hyp2f1(3-gamma, 3-gamma, 4-gamma, -r200/rs)
            h2y = spec.hyp2f1(3-gamma, 3-gamma, 4-gamma, -r/rs)
            
            return np.sqrt((r / r200)**(3 - gamma) * M200 * h2y / h2x * G_mpc / r)
   

#NFW surface density profile
def nfw_sigma(R, rho_s, r_s):
    """
    Computes the projected surface density Σ(R) of the NFW profile.
    Parameters:
        R     : projected radius (scalar or array) [same units as r_s]
        rho_s : scale density
        r_s   : scale radius
    Returns:
        Σ(R)  : surface density at radius R
    """
    
    x = np.array(R) / r_s
    sigma = np.zeros_like(x)

    # x < 1
    mask1 = (x < 1)
    sqrt1 = np.sqrt(1 - x[mask1]**2)
    sigma[mask1] = (1 / (x[mask1]**2 - 1)) * (1 - (2 / sqrt1) * np.arctanh(sqrt1 / (1 + x[mask1])))

    # x == 1
    mask2 = (x == 1)
    sigma[mask2] = 1.0 / 3.0

    # x > 1
    mask3 = (x > 1)
    sqrt2 = np.sqrt(x[mask3]**2 - 1)
    sigma[mask3] = (1 / (x[mask3]**2 - 1)) * (1 - (2 / sqrt2) * np.arctan(sqrt2 / (1 + x[mask3])))

    return 2 * rho_s * r_s * sigma


#NFW projected mass profile
def nfw_projected_mass(R, rho_s, r_s):
    """
    Computes the projected mass M_proj(R) inside a cylinder of radius R for the NFW profile.
    Parameters:
        R     : projected radius (scalar or array) [same units as r_s]
        rho_s : scale density
        r_s   : scale radius
    Returns:
        M_proj(R) : projected mass inside radius R
    """

    x = np.array(R) / r_s
    M_proj = np.zeros_like(x)

    # x < 1
    mask1 = x < 1
    acos_arg = 1 / x[mask1]
    term = np.arccosh(acos_arg)
    F = (term**1) / np.sqrt(1 - x[mask1]**2)
    M_proj[mask1] = np.log(x[mask1] / 2) + F

    # x == 1
    mask2 = x == 1
    M_proj[mask2] = 1 - np.log(2)  #This simplifies to 0

    # x > 1
    mask3 = x > 1
    a = 1 / x[mask3]
    term = np.arccos(a)
    F = (term**1) / np.sqrt(x[mask3]**2 - 1)
    M_proj[mask3] = np.log(x[mask3] / 2) + F

    return 4 * np.pi * rho_s * r_s**3 * M_proj


#Function used to compute the 2D halo profile, either NFW or gNFW, mass, density or circular velocity
def HaloProfile2D(lr, lr200, lrs, cosm_pars, profile_model="NFW", quantity_type="MASS"):
    """
    This function returns a halo 2D mass or 2D density profile at the specified radius, either NFW or gNFW.
    
    Args:
    -----
    r:     [float]
           Halo profile radii in Mpc
    r200:  [float]
           Halo r200 in Mpc
    rs:    [float]
           Halo scale radius in Mpc
    gamma: [float]
           gNFW gamma exponent (defined in the interval [0, 2], it's 1 for NFW)
    cosm_pars:     [float array]
                   Array containing cosmological parameters (h, omega matter, omega lambda, redshift)
    profile_model: [string]
                   Profile model to return: NFW, GNFW
    quantity_type: [string]
                   Profile type to return: MASS, DENSITY

    Returns:
    --------
    profile: [float]
             Value of the selected profile at a given radius, in Msun for MASS profile, and Msun/Mpc^3 for DENSITY profile
    """

    h, Om, Ol, z = cosm_pars
    H0 = 100 * h

    r = 10**lr
    r200 = 10**lr200
    rs = 10**lrs

    c200 = r200 / rs
    x = r / rs
    H2z = H0**2 * (Om * (1 + z)**3 + Ol)
    M200 = 100 * H2z / G_mpc * r200**3
    fac200 = np.log(1 + c200) - c200 / (1 + c200)

    #Compute either mass or density for the NFW profile   
    if profile_model.upper() == "NFW":
        rho_s = M200 / (4 * np.pi * rs**3) / fac200
        
        if quantity_type.upper() == "MASS":
            return nfw_projected_mass(r, rho_s, rs)

        elif quantity_type.upper() == "DENSITY":
            return nfw_sigma(r, rho_s, rs)   


def FitSimulationDataMP(simulation_type, sim_regions, fit_quantities, basename, Ntofit, mThresh, Rfit_bounds=None, Rfit_bounds_gnfw=None, 
                      nfw_bounds=((-3., -3.), (2., 2.)), gnfw_bounds=((-3., -3., -15.), (2., 2., np.log10(2.)))):
    """
    This function performs 3D NFW and gNFW fits of halos of a given simulation, saving only the fits and binned profiles of halos whose NFW mass fit
    returns M200 > mThresh. The fits are performed using non-linear least squares through the SciPy function curve_fit.
    
    Args:
    -----
    simulation_type:      [string]
                          Input simulation model, e.g. CDM
    fit_quantities:       [string array]
                          Profiles to fit the NFW and gNFW models to, e.g. ["MASS", "DENSITY", "VCIRC"]
    Rfit_bounds:          [array_like of tuples]
                          Each tuple should contain the lower and upper bounds on the radius over which to fit NFW for a given simulation_type
    Rfit_bounds_gnfw:     [array_like of tuples]
                          Each tuple should contain the lower and upper bounds on the radius over which to fit gNFW for a given simulation_type
    nfw_bounds:           [2-tuple of array_like]
                          Lower and upper bounds on the NFW fit parameters
    gnfw_bounds:          [2-tuple of array_like]
                          Lower and upper bounds on the gNFW fit parameters

    Returns:
    -------
    fit_pars:         [dict]
                      Nested dictionary containing all NFW and gNFW fit parameters, chi2 and bin radius for both mass and, if enabled, density profiles
    fit_cov:          [dict]
                      Nested dictionary containing all NFW and gNFW fit parameters uncertainties and covariances for both mass and, if enabled, density 
                      and circular velocity profiles
    halo_profiles:    [dict]
                      Nested dictionary containing all mass, density and velocity anisotropy profiles
    R500_SIM:         [dict]
                      Contains the R500c obtained directly from the HDF5 output for the input model, in Mpc
    R200_SIM:         [dict]
                      Contains the R200c obtained directly from the HDF5 output for the input model, in Mpc
    halo ids:         [dict]
                      Contains the ROCKSTAR halo IDs of the saved fits for the input model
    cosm_pars:        [dict]
                      Contains a numpy array [h, Om, Ol, z] of cosmological parameters for the input model
    sim_props:        [dict]
                      Contains a tuple (Nhalo_tot, Mpart, rs_large, gamma_low) containing total number of halos in the simulation and particle mass
    """

    #Define the relevant quantities for every profile
    #####################################################################################################
    
    #Store the halo IDs for each saved fit
    halo_ids = []
    cosm_pars = []

    #Total number of halos and number of large rs
    Nhalos_file = 0
    rs_large = 0
    Mpart = 0
    
    #Halo r500c and r200c from simulation
    r500c_SIM = []
    r200c_SIM = []
    
    #Binned halo profiles     
    #Insert all halo profiles corresponding to the saved NFW and gNFW fits into a dictionary to be returned by the function
    halo_profiles = {"MASS": [], "DENSITY": [], "VCIRC": [], "BETA": [], "NUM": [], "CUMNUM": [], 
                     "SIGMAr": [], "SIGMAt": [], "SIGMAp": [], "ERR_MASS": [], "ERR_DENSITY": [], "ERR_VCIRC": [], "R": []}

    #Dicts containing all NFW and gNFW fit parameters, chi2 and uncertainties for mass and, if enabled, density and circular velocity profiles
    fit_pars = {"NFW": dict(), "gNFW": dict()}
    fit_cov = {"NFW": dict(), "gNFW": dict()}

    for quantity in fit_quantities:
        fit_pars["NFW"][quantity] = {"r200" : [], "rs": [], "chi2": [], "M200": []}
        fit_pars["gNFW"][quantity] = {"r200" : [], "rs": [], "gamma": [], "chi2": [], "M200": [], "rm2": []}

        fit_cov["NFW"][quantity] = {"r200" : [], "rs": [], "r200_rs": []}
        fit_cov["gNFW"][quantity] = {"r200" : [], "rs": [], "gamma": [], "r200_rs": [], "r200_gamma": [], "rs_gamma": []}

    #Extract the binned profiles and other relevant quantities from the HDF5 files
    #####################################################################################################

    print("CURRENT SIMULATION: " + simulation_type)
    
    #Loop over the simulation regions
    for D in sim_regions[:]:
        #Define the folder where all progress is saved
        distdir = os.getcwd()
        distname = basename + "/" + simulation_type + D + "/progress/3D/"
        distot = os.path.join(distdir, distname)
        
        #If the directory doesn't exist, create it
        if not os.path.exists(distot):
            os.makedirs(distot) 
            
        #Path of all hdf5 files in a given simulation folder
        filenames = [os.path.normpath(i) for i in glob.glob(basename + "/" + simulation_type + D + "/*.hdf5")]
        filenames_remaining = []

        if len(glob.glob(basename + "/" + simulation_type + D + "/progress/3D/filenames.txt")) == 0:
            filenames_done = []

        else:
            filenames_done = np.loadtxt(basename + "/" + simulation_type + D + "/progress/3D/filenames.txt", dtype="str", ndmin=1)

        if len(filenames_done) != 0:
            for fnames in filenames:
                if (filenames_done == fnames).any() == False:
                    filenames_remaining.append(fnames)

            #Load current progress from file
            filenames_done = filenames_done.tolist()

        else:
            filenames_remaining = filenames

        #Count the total number of halos in the region across all files
        Nhalos_region = 0

        if Ntofit == None:
            file_max = len(filenames_remaining) + 1

        else:
            file_max = Ntofit

        #Read the hdf5 files
        for f_i, h5name in enumerate(filenames_remaining[:file_max]):
            print(f"GLOBAL PROGRESS: {len(filenames_done) + 1} / {len(filenames)}\n")
            
            with h5py.File(h5name, "r") as hdf:
                print("READING FILE: " + h5name)
                
                #Get the cosmological parameters
                h = hdf['Header'].attrs['h']
                Om = hdf['Header'].attrs['Om']
                Ol = hdf['Header'].attrs['Ol']
                z = hdf['Header'].attrs['z']
                cosm_pars = np.array([h, Om, Ol, z])
                H2z = (100 * h)**2 * (Om * (1 + z)**3 + Ol)

                #Only print cosmological parameters once
                if h5name == filenames[0]:
                    print("Cosmology:\n h = {0:3.4f}, Om = {1:3.4f}, Ol = {2:3.4f}, z = {3:3.4f}\n".format(h, Om , Ol, z))

                #Halo properties
                ids = hdf['Header/Group_IDs'][:]    #Group IDs
                data = hdf['RadialProfiles']   #Halo profiles
                Mpart = hdf['Header'].attrs['Mpart']    #Particle masses
                Nhalos_file += len(ids)    #Add the number of halos in the file to the total number of halos the a region

                #Display global progress
                print(f" ----- Halos to fit: {len(ids)}")
            
                #Loop over the halo IDs
                for id_count, ii in enumerate(ids):
                    #Display halo fitting progress
                    if int(id_count / Nhalos_file * 100) % 5 == 0:
                        sys.stdout.write("\r")
                        sys.stdout.write(f"{simulation_type} ---- PROGRESS: {id_count / Nhalos_file * 100:.0f}% ---- ")
                        sys.stdout.flush()
                    
                    #Get r200 and r500 (in Mpc)
                    r500c = data['Group_%i_R500'%ii][0]
                    r200c = data['Group_%i_R200'%ii][0]
                    M200c = 200 / (2 * G_mpc) * H2z * (r200c)**3
            
                    #Get radial bin centers in Mpc and halo binned profiles
                    bin_centers = data["Group_%i_Radius"%ii][:]  #In units of r500
                    radius_Mpc = bin_centers * r500c   #In Mpc
                    Mass = data["Group_%i_Mass"%ii][:]    #Total mass inside a bin
                    MassCum = data["Group_%i_MassCum"%ii][:]    #In M_sun
                    Den = data['Group_%i_Density'%ii][:] / r500c**3    #Density inside a bin in M_sun/Mpc^3
                    Vel_circ = np.sqrt(G_mpc * MassCum / radius_Mpc)    #In km/s
                    Npart = data['Group_%i_Npart'%ii][:]    #Number of particles inside a bin
                    NpartCum = data['Group_%i_NpartCum'%ii][:]    #Cumulative number of particles
            
                    #Compute mass Poisson errors
                    err_mass = Mpart * np.sqrt(NpartCum)
            
                    #Compute density Poisson errors
                    #Since the bin edges are equally spaced in log space, to compute the volume shells add half of the difference between log bin centers to 
                    #the log bin center to obtain all the bin edges except the first, which is added manually
                    bin_edges = 10**np.append(np.log10(bin_centers[0]) - 0.5 * np.diff(np.log10(bin_centers))[0], 
                                               np.log10(bin_centers) + 0.5 * np.diff(np.log10(bin_centers))[0])
                    volume_bins = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
                    err_den = Mpart * np.sqrt(Npart) / (volume_bins * r500c**3)
        
                    #Compute circular velocity Poisson errors
                    err_vel = 0.5 * np.sqrt(G_mpc / radius_Mpc) * MassCum**(-1/2) * err_mass
            
                    #Get 3D velocity dispersions and compute velocity anisotropy beta
                    sr2, st2, sp2 = data["Group_%i_VelocityDispersion"%ii][3:6]
                    beta_r = 1 - (st2 + sp2) / (2 * sr2)
                    beta_r = np.nan_to_num(beta_r, 0.)

                    #Fit the binned profiles with an NFW and a gNFW model
                    #####################################################################################################

                    try:
                        #Store the fit parameters, covariance matrix and chi2 for a single fit
                        fit_parameters_dict = {p_type: {q: {"popt": None, "cov": None, "chi2": None} for q in fit_quantities} for p_type in ["NFW", "gNFW"]}

                        #Initial guess for curve_fit
                        initial_pars = [[np.log10(1.5*r500c), np.log10(0.5*r500c)], [np.log10(1.5*r500c), np.log10(0.5*r500c), 0.]]

                        #Profile models to fit
                        profile_type = ["NFW", "gNFW"]
                        
                        for p_type, par_0, bounds in zip(profile_type, initial_pars, [nfw_bounds, gnfw_bounds]):
                            for quantity in fit_quantities:
                                if quantity == "MASS":
                                    binned_profile = MassCum
                                    err_profile = err_mass

                                elif quantity == "DENSITY":
                                    binned_profile = Den
                                    err_profile = err_den

                                elif quantity == "VCIRC":
                                    binned_profile = Vel_circ
                                    err_profile = err_vel
                                    
                                #Fit the mass (and, if enabled, density and circular velocity) profiles and compute the chi2
                                try:
                                    if p_type == "NFW":
                                        #Condition on radius: select only R (in units of r500) inside the fit_bounds, and not corresponding to an empty bin
                                        R_cond = (bin_centers >= Rfit_bounds[0]) & (bin_centers <= Rfit_bounds[1]) & (Mass != 0)

                                        #Define the NFW and gNFW fit functions with the current cosmology
                                        fnfw = lambda lr, lr200, lrs: np.log10(HaloProfile(lr, lr200, lrs, 0., cosm_pars, p_type, quantity))
                                        y_data = np.log10(binned_profile[R_cond])
                                        y_err = err_profile[R_cond] / binned_profile[R_cond] / np.log(10)
                                        x_data = np.log10(radius_Mpc[R_cond])

                                        #Always fit the NFW model first, since it is used to create the condition for saving the halo
                                        cond_save = True

                                    elif p_type == "gNFW":
                                        #Condition on radius: select only R (in units of r500) inside the fit_bounds, and not corresponding to an empty bin
                                        R_cond = (bin_centers >= Rfit_bounds_gnfw[0]) & (bin_centers <= Rfit_bounds_gnfw[1]) & (Mass != 0)
                                        
                                        #Define the NFW and gNFW fit functions with the current cosmology
                                        fnfw = lambda lr, lr200, lrs, lgamma: np.log10(HaloProfile(lr, lr200, lrs, lgamma, cosm_pars, p_type, quantity))
                                        y_data = np.log10(binned_profile[R_cond])
                                        y_err = err_profile[R_cond] / binned_profile[R_cond] / np.log(10)
                                        x_data = np.log10(radius_Mpc[R_cond])

                                    #Only fit non-NFW models if the halo would be saved (cond_save == True), to save on computation time
                                    #Fit using non-linear least squares
                                    if cond_save:
                                        fit_parameters_dict[p_type][quantity]["popt"],\
                                        fit_parameters_dict[p_type][quantity]["cov"],\
                                        fit_parameters_dict[p_type][quantity]["chi2"] = FitAndChi2(fnfw, x_data, y_data, y_err,
                                                                                       par_0=par_0, par_bounds=bounds)

                                    #Update cond_save with information from the NFW fit
                                    if p_type == "NFW":
                                        r200_MassFit = 10**fit_parameters_dict["NFW"]["MASS"]["popt"][0]
                                        rs_MassFit = 10**fit_parameters_dict["NFW"]["MASS"]["popt"][1]
                                        M200_MassFit = r200_MassFit**3 * 100 * H2z / G_mpc
                                        chi2_MassFit = 10**fit_parameters_dict["NFW"]["MASS"]["chi2"]
        
                                        cond_save = (M200_MassFit >= mThresh) and (chi2_MassFit < np.inf) and (rs_MassFit <= r200_MassFit)

                                except:
                                    print("\tERROR IN " + p_type + " " + quantity + "FIT")

                        #Save the fits and binned profiles only for halos with NFW mass fitM200 >= mThresh, gamma > gamma_min, rs < r200 and r200c > 0
                        #####################################################################################################
                                         
                        #Save NFW and gNFW fit parameters and binned profiles
                        if cond_save:
                            #Save halo ID and NFW cut radius
                            halo_ids.append(ii)
                            
                            #Save r500c and r200c from simulation
                            r500c_SIM.append(r500c)
                            r200c_SIM.append(r200c)
            
                            #Save mass, density, circular velocity and anisotropy profiles and their uncertainties, and the radii in units of r500
                            saved_quantities = [MassCum, Den, Vel_circ, beta_r, Npart, NpartCum, sr2, st2, sp2, err_mass, err_den, err_vel, bin_centers]
                            
                            for key, quantity in zip(list(halo_profiles.keys()), saved_quantities):
                                halo_profiles[key].append(quantity)

                            #####################################################################################################

                            #NFW fit
                            #Save the fit parameters and uncertainties, the chi2 and M200
                            for quantity in fit_quantities:
                                fit_NFW = [*fit_parameters_dict["NFW"][quantity]["popt"],
                                           fit_parameters_dict["NFW"][quantity]["chi2"],
                                           (10**fit_parameters_dict["NFW"][quantity]["popt"][0])**3 * 100 * H2z / G_mpc]

                                cov_NFW = [*np.diag(fit_parameters_dict["NFW"][quantity]["cov"]),
                                           fit_parameters_dict["NFW"][quantity]["cov"][0, 1]]

                                #Store the fit parameters in a dictionary
                                for p_i, key in enumerate(list(fit_pars["NFW"][quantity].keys())):
                                    fit_pars["NFW"][quantity][key].append(fit_NFW[p_i])

                                #Store the fit covariances in a dictionary
                                for p_i, key in enumerate(list(fit_cov["NFW"][quantity].keys())):
                                    fit_cov["NFW"][quantity][key].append(cov_NFW[p_i])

                            #####################################################################################################
                    
                            #gNFW fit
                            #Save the fit parameters and uncertainties, the chi2, M200 and rm2
                            for quantity in fit_quantities:
                                fit_gNFW = [*fit_parameters_dict["gNFW"][quantity]["popt"], 
                                           fit_parameters_dict["gNFW"][quantity]["chi2"],
                                           (10**fit_parameters_dict["gNFW"][quantity]["popt"][0])**3 * 100 * H2z / G_mpc,
                                           10**fit_parameters_dict["gNFW"][quantity]["popt"][1] * (2 - 10**fit_parameters_dict["gNFW"][quantity]["popt"][2])]

                                cov_gNFW = [*np.diag(fit_parameters_dict["gNFW"][quantity]["cov"]),
                                           fit_parameters_dict["gNFW"][quantity]["cov"][0, 1],
                                           fit_parameters_dict["gNFW"][quantity]["cov"][0, 2],
                                           fit_parameters_dict["gNFW"][quantity]["cov"][1, 2]]

                                #Store the fit parameters in a dictionary
                                for p_i, key in enumerate(list(fit_pars["gNFW"][quantity].keys())):
                                    fit_pars["gNFW"][quantity][key].append(fit_gNFW[p_i])

                                #Store the fit covariances in a dictionary
                                for p_i, key in enumerate(list(fit_cov["gNFW"][quantity].keys())):
                                    fit_cov["gNFW"][quantity][key].append(cov_gNFW[p_i])

                        #Update the number of large rs
                        if M200_MassFit >= mThresh and rs_MassFit > r200_MassFit:
                            rs_large += 1
                                
                    except ValueError:
                        print("\tSKIP FIT FOR HALO " + str(ii) + " IN REGION " + str(D))

            print("\nTotal " + simulation_type + f" halos saved = {len(halo_ids)}")
            print("--------------------------------------------------------------\n")
            
            #Save current file as already read
            filenames_done.append(h5name)
            
            #Create a save file with the current progress
            np.savetxt(basename + "/" + simulation_type + D + "/progress/3D/filenames.txt", filenames_done, fmt="%s")
            
        #Save current progress to file in the "progress" folder
        save_name = "save_state_N" + str(len(filenames_done))
        np.savez(basename + "/" + simulation_type + D + "/progress/3D/" + save_name, fit_pars=fit_pars, fit_cov=fit_cov, 
                                                                                     halo_profiles=halo_profiles, halo_ids=halo_ids,
                                                                                     r500c_SIM=r500c_SIM, r200c_SIM=r200c_SIM,
                                                                                     cosm_pars=cosm_pars, 
                                                                                     Nhalos_file=Nhalos_file, Mpart=Mpart)


def FitSimulationDataMP2D(simulation_type, sim_regions, fit_quantities, dimensions, save_IDs, 
                        basename, Ntofit, Rfit_bounds=None, nfw_bounds=((-3, -3), (2, 2))):
    """
    This function performs 2D NFW fits of halos of a given simulation, selecting only the halos saved by the FitSimulationData function in the 3D case. 
    The fits are performed using non-linear least squares through the SciPy function curve_fit.
    
    Args:
    -----
    simulation_type: [string]
                     Type of simulation, e.g. CDM
    fit_quantities:  [string array]
                     Profiles to fit the NFW and gNFW models to, e.g. ["MASS", "DENSITY"]
    dimensions:      [string array]
                     Axes along which the projected profiles were computed
    save_IDs:        [int array]
                     IDs of the halos to save. If None, save the halos based on their fit M200 NFW
    Rfit_bounds:     [array_like of tuples]
                     Each tuple should contain the lower and upper bounds on the radius over which to fit for a given simulation_type
    nfw_bounds:      [2-tuple of array_like]
                     Lower and upper bounds on the NFW fit parameters
    gnfw_bounds:     [2-tuple of array_like]
                     Lower and upper bounds on the gNFW fit parameters

    Return:
    -------
    model_pars_2D:        [dict]
                          Nested dictionary containing all NFW and gNFW fit parameters, chi2 and bin radius for both mass 
                          and, if enabled, density profiles
    model_cov_2D:         [dict]
                          Nested dictionary containing all NFW and gNFW fit parameters uncertainties and covariances for both mass 
                          and, if enabled, density profiles
    model_profiles_2D:    [dict]
                          Nested dictionary containing all mass, density and velocity anisotropy profiles
    """
    
    #Define the relevant quantities for every profile
    #####################################################################################################
    
    Nhalos_file = 0
    fit_pars = {"NFW": dict()}
    fit_cov = {"NFW": dict()}

    for quantity in fit_quantities:
        fit_pars["NFW"][quantity], fit_cov["NFW"][quantity] = dict(), dict()
        
        for dim in dimensions:
            #Fit parameters for every dimension
            fit_pars["NFW"][quantity][dim] = {"r200" : [], "rs": [], "chi2": [], "M200": []}
            fit_cov["NFW"][quantity][dim] = {"r200" : [], "rs": [], "r200_rs": []}

    #Mass, density and anisotropy profiles for every projected dimension
    halo_profiles = {dim: {"MASS": [], "DENSITY": [], "DEN_CUM": [], "NUM": [], "CUMNUM": [], "ERR_MASS": [], 
                              "ERR_DENSITY": [], "ERR_DEN_CUM": [], "R": []} for dim in dimensions}
                              
    #Extract the binned profiles and other relevant quantities from the HDF5 files
    #####################################################################################################
    
    #Loop over the simulation regions
    print("CURRENT SIMULATION: " + simulation_type)
    
    for D in sim_regions[:]:
        #Define the folder where all progress is saved
        distdir = os.getcwd()
        distname = basename + "/" + simulation_type + D + "/progress/2D/"
        distot = os.path.join(distdir, distname)
        
        #If the directory doesn't exist, create it
        if not os.path.exists(distot):
            os.makedirs(distot) 
            
        #Path of all hdf5 files in a given simulation folder
        filenames = [os.path.normpath(i) for i in glob.glob(basename + "/" + simulation_type + D + "/*.hdf5")]
        filenames_remaining = []

        if len(glob.glob(basename + "/" + simulation_type + D + "/progress/2D/filenames.txt")) == 0:
            filenames_done = []

        else:
            filenames_done = np.loadtxt(basename + "/" + simulation_type + D + "/progress/2D/filenames.txt", dtype="str", ndmin=1)

        if len(filenames_done) != 0:
            for fnames in filenames:
                if (filenames_done == fnames).any() == False:
                    filenames_remaining.append(fnames)

            #Load current progress from file
            filenames_done = filenames_done.tolist()

        else:
            filenames_remaining = filenames

        #Count the total number of halos in the region across all files
        Nhalos_region = 0

        if Ntofit == None:
            file_max = len(filenames_remaining) + 1

        else:
            file_max = Ntofit

        #Read the hdf5 files
        for f_i, h5name in enumerate(filenames_remaining[:file_max]):
            print(f"GLOBAL PROGRESS: {len(filenames_done) + 1} / {len(filenames)}\n")
            
            with h5py.File(h5name, "r") as hdf:
                print("READING FILE: " + h5name)
                
                #Get the cosmological parameters
                h = hdf['Header'].attrs['h']
                Om = hdf['Header'].attrs['Om']
                Ol = hdf['Header'].attrs['Ol']
                z = hdf['Header'].attrs['z']
                cosm_pars = np.array([h, Om, Ol, z])
                H2z = (100 * h)**2 * (Om * (1 + z)**3 + Ol)
            
                #Only fit the saved halos corresponding to the 3D IDs in the file
                ids = hdf['Header/Group_IDs'][:]    #Group IDs
                file_IDs = np.intersect1d(save_IDs, ids)
                
                #Halo properties
                data = hdf['RadialProfiles']   #Halo profiles
                Mpart = hdf['Header'].attrs['Mpart']    #Particle masses
                Nhalos_file += len(file_IDs)    #Add the number of halos in the file to the total number of halos the a region

                for id_count, ii in enumerate(file_IDs):
                    #Display halo fitting progress
                    if int(id_count / Nhalos_file * 100) % 5 == 0:
                        sys.stdout.write("\r")
                        sys.stdout.write(f"{simulation_type} ---- PROGRESS: {id_count / Nhalos_file * 100:.0f}% ---- ")
                        sys.stdout.flush()
                        
                    #Get halo parameters, r200 and r500
                    r500c = data['Group_%i_R500'%ii][0]

                    for dim in dimensions:
                        #Get projected profiles
                        bin_centers = data["Group_%i_Radius2D"%ii + dim][:]  #In units of r500
                        radius_Mpc = bin_centers * r500c   #In Mpc
                        Mass_2D = data["Group_%i_Mass2D"%ii + dim][:]    #Total mass inside a bin
                        MassCum_2D = data["Group_%i_MassCum2D"%ii + dim][:]
                        Den_2D = data["Group_%i_Density2D"%ii + dim][:] / r500c**2    #In M_sun/Mpc^2
                        Ncum_2D = data["Group_%i_NpartCum2D"%ii + dim][:]
                        Nbin_2D = data["Group_%i_Npart2D"%ii + dim][:]

                        #Compute mass Poisson errors
                        err_mass = Mpart * np.sqrt(Ncum_2D)
                
                        #Compute density Poisson errors, if enabled
                        #Since the bin edges are equally spaced in log space, to compute the volume shells add half of the difference between log bin centers to 
                        #the log bin center to obtain all the bin edges except the first, which is added manually
                        bin_edges = 10**np.append(np.log10(bin_centers[0]) - 0.5 * np.diff(np.log10(bin_centers))[0], 
                                                   np.log10(bin_centers) + 0.5 * np.diff(np.log10(bin_centers))[0])
                        area_bins = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
                        err_den = np.sqrt(Nbin_2D) / (area_bins * r500c**2)

                        #Cumulative surface density, in M_sun/Mpc^2
                        DenCum_2D = MassCum_2D / (np.pi * bin_edges[1:]**2 * r500c**2)
                        err_den_cum = Mpart * np.sqrt(Ncum_2D) / (np.pi * bin_edges[1:]**2 * r500c**2)

                        #####################################################################################################

                        # Radius interval to fit
                        R_cut = Rfit_bounds[0]
                        R_cond = np.logical_and(np.logical_and(bin_centers >= R_cut, bin_centers < Rfit_bounds[1]), Mass_2D != 0)
                        
                        #####################################################################################################

                        #Fit the binned profiles with NFW and gNFW
                        try:
                            profile_type = np.array(["NFW"])
                            fit_parameters_dict = {p_type: {q: {"popt": None, "cov": None, "chi2": None} for q in fit_quantities} 
                                                   for p_type in profile_type}

                            initial_pars = [[np.log10(1.5*r500c), np.log10(0.5*r500c)], [np.log10(1.5*r500c), np.log10(0.5*r500c), 0]]
                            
                            for p_type in profile_type:                           
                                for quantity in fit_quantities:
                                    if quantity == "MASS":
                                        binned_profile = MassCum_2D
                                        err_profile = err_mass
        
                                    elif quantity == "DENSITY":
                                        binned_profile = Den_2D
                                        err_profile = err_den
                                    
                                    #Define the NFW fit functions with the current cosmology
                                    if p_type == "NFW":
                                        fnfw = lambda lr, lr200, lrs: np.log10(HaloProfile2D(lr, lr200, lrs, cosm_pars, p_type, quantity))
                                        y_data = np.log10(binned_profile[R_cond])
                                        y_err = err_profile[R_cond] / binned_profile[R_cond] / np.log(10)
                                        par_0 = initial_pars[0]
                                        bounds = nfw_bounds

                                    #Fit the profile
                                    fit_parameters_dict[p_type][quantity]["popt"],\
                                    fit_parameters_dict[p_type][quantity]["cov"],\
                                    fit_parameters_dict[p_type][quantity]["chi2"] = FitAndChi2(fnfw, np.log10(radius_Mpc[R_cond]), y_data, y_err,
                                                                                                par_0=par_0, par_bounds=bounds)
        
                            #####################################################################################################

                            #Save the fit parameters and binned profiles only if the mass fit results in rs < r200
                            r200_MassFit = 10**fit_parameters_dict[p_type][quantity]["popt"][0]
                            rs_MassFit = 10**fit_parameters_dict[p_type][quantity]["popt"][1]
                            
                            if rs_MassFit < r200_MassFit:                    
                                #Save mass, density, and number profiles and their uncertainties, and the radii in units of r500
                                profiles_list = [MassCum_2D, Den_2D, DenCum_2D, Nbin_2D, Ncum_2D, err_mass, err_den, err_den_cum, bin_centers]
                                
                                for key, quantity in zip(list(halo_profiles[dim].keys()), profiles_list):
                                    halo_profiles[dim][key].append(quantity)
                                    
                                #####################################################################################################
            
                                #NFW fit
                                #Save the fit parameters and uncertainties, the chi2 and M200
                                if np.any(profile_type == "NFW"):
                                    for quantity in fit_quantities:
                                        fit_NFW = [*fit_parameters_dict["NFW"][quantity]["popt"],
                                                   fit_parameters_dict["NFW"][quantity]["chi2"],
                                                   (10**fit_parameters_dict["NFW"][quantity]["popt"][0])**3 * 100 * H2z / G_mpc]
            
                                        cov_NFW = [*np.diag(fit_parameters_dict["NFW"][quantity]["cov"]),
                                                   fit_parameters_dict["NFW"][quantity]["cov"][0, 1]]
            
                                        #Store the fit parameters in a dictionary
                                        for p_i, key in enumerate(list(fit_pars["NFW"][quantity][dim].keys())):
                                            fit_pars["NFW"][quantity][dim][key].append(fit_NFW[p_i])
            
                                        #Store the fit covariances in a dictionary
                                        for p_i, key in enumerate(list(fit_cov["NFW"][quantity][dim].keys())):
                                            fit_cov["NFW"][quantity][dim][key].append(cov_NFW[p_i])
                                
                        except ValueError:
                            print("SKIP FIT FOR HALO " + str(ii) + " IN REGION " + str(D))

                print("\nTotal " + simulation_type + f" halos saved = {len(file_IDs)}")
                print("--------------------------------------------------------------\n")
                
                #Save current file as already read
                filenames_done.append(h5name)
                
                #Create a save file with the current progress
                np.savetxt(basename + "/" + simulation_type + D + "/progress/2D/filenames.txt", filenames_done, fmt="%s")
            
        #Save current progress to file in the "progress" folder
        save_name = "save_state_N" + str(len(filenames_done))
        np.savez(basename + "/" + simulation_type + D + "/progress/2D/" + save_name, fit_pars=fit_pars, fit_cov=fit_cov, 
                                                                                     halo_profiles=halo_profiles)


def multiprocessing_fit(sim_type, sim_regions, fit_quantities, basename, Ntofit, mThresh, Rfit_bounds, Rfit_bounds_gnfw):
    print("STARTED PROCESS " + str(os.getpid()) + " WORKING ON " + sim_type)
    
    FitSimulationDataMP(sim_type, sim_regions, fit_quantities, basename, Ntofit, mThresh, Rfit_bounds, Rfit_bounds_gnfw)

    print("FINISHED PROCESS " + str(os.getpid()) + " WORKING ON " + sim_type)


def multiprocessing_fit_2D(sim_type, sim_regions, fit_quantities, dimensions, fit_ids, basename, Ntofit, Rfit_bounds):
    print("STARTED PROCESS " + str(os.getpid()) + " WORKING ON " + sim_type)
    
    FitSimulationDataMP2D(sim_type, sim_regions, fit_quantities, dimensions, fit_ids, basename, Ntofit, Rfit_bounds)

    print("FINISHED PROCESS " + str(os.getpid()) + " WORKING ON " + sim_type)
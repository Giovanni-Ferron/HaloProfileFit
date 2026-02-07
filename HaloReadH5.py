import numpy as np
import scipy.special as spec
from scipy.optimize import curve_fit
from itertools import repeat
import multiprocessing.pool as mp
import shutil
import h5py
import glob
import os
import sys
import time


class HaloModel:
    """
    Class used to store information about the halo profile models used in fitting. 
    By default only NFW and gNFW are available, but more can be added by inserting them here and in the HaloProfile function.
    """
    
    def __init__(self, name, custom_profile=None, free_par_names=None, fit_bounds=None):
        self.name = name
        self.custom_profile = custom_profile
        
        if name.upper() == "NFW":
            free_par_names = ["r200", "rs"]
            self.fit_bounds = [(-3, -3), (2, 2)]
            
        elif name.upper() == "GNFW":
            free_par_names = ["r200", "rs", "gamma"]
            self.fit_bounds = [(-3, -3, -15), (2, 2, np.log10(2))]
        
        self.free_par_names = free_par_names


    def _profile(self, lr, *params, cosm_params=None, quantity=None, projection=False):
        """
        Function used to generate the halo profile.
        
        Args:
        -----
        lr:             [array-like]
                        Log of the radii at which to compute the halo profile, in Mpc
        *params         []
                        Log of the free parmeters of the profile model
        cosm_params     [array-like]
                        List containing cosmological parameters h, Om, Ol, z of the simulations
        quantity        [string]
                        Quantity type of the profile, e.g. MASS or DENSITY
        projection      [bool]
                        Set to True to consider the 2D projected version of the profile, otherwise consider the 3D one
                        
        Returns:
        --------
        profile         [array]
                        Halo profile computed at the specified radii
        """
        
        if self.custom_profile is None:
            return HaloProfile(lr, *params, cosm_params=cosm_params, profile_model=self.name, quantity_type=quantity, projection=projection)
            
        else:
            return self.custom_profile(lr, *params, cosm_params, quantity)


def HaloProfile(lr, *free_params, cosm_params, profile_model="NFW", quantity_type="MASS", projection=False):
    """
    This function returns a halo mass, density or circular velocity profile at the specified radius, either NFW or gNFW.
    
    Args:
    -----
    lr:                     [array-like]
                            Log of the radii at which to compute the halo profile, in Mpc
    *free_params            []
                            Log of the free parmeters of the profile model
    cosm_params             [array-like]
                            List containing cosmological parameters [h, Om, Ol, z] of the simulations
    quantity                [string]
                            Quantity type of the profile, e.g. MASS or DENSITY
    projection              [bool]
                            Set to True to consider the 2D projected version of the profile, otherwise consider the 3D one

    Returns:
    --------
    profile:                [float]
                            Value of the selected profile at a given radius.
                            In Msun for MASS profile, Msun/Mpc^3 for DENSITY profile and km/s for VCIRC profile
    """
    
    #Define G
    G_mpc = 4.302e-9    #In Msun^-1 * Mpc * km^2 * s^-2
    
    h, Om, Ol, z = cosm_params
    H0 = 100 * h
    H2z = H0**2 * (Om * (1 + z)**3 + Ol)

    r = 10**lr
    free_pars = np.atleast_1d(free_params)

    #Compute either mass, density or circular velocity for the NFW profile, either 3D or 2D
    if profile_model.upper() == "NFW":
        r200, rs = 10**free_pars
        c200 = r200 / rs
        x = r / rs
        
        M200 = 100 * H2z / G_mpc * r200**3
        fac200 = np.log(1 + c200) - c200 / (1 + c200)
        
        rho_s = M200 / (4 * np.pi * rs**3) / fac200
        
        if quantity_type.upper() == "MASS":
            if projection:
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
            
                return 4 * np.pi * rho_s * rs**3 * M_proj
            
            else:
                return (np.log(1 + x) - x / (1 + x)) / fac200 * M200

        elif quantity_type.upper() == "DENSITY":
            if projection:
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

                return 2 * rho_s * rs * sigma
            
            else:
                return M200 / (4 * np.pi * fac200) * (r * (rs + r)**2)**-1

        elif quantity_type.upper() == "VCIRC":
            return np.sqrt((np.log(1 + x) - x / (1 + x)) / fac200 * M200 * G_mpc / r)

    #Compute either mass, density or circular velocity for the gNFW profile, either 3D or 2D
    elif profile_model.upper() == "GNFW":
        r200, rs, gamma = 10**free_pars
        M200 = 100 * H2z / G_mpc * r200**3
        x = r / rs
        
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


def RestoreSavestates(sim_name, sim_type):
    """
    Function that loads all savestate data for the binned profiles into dictionaries.

    Args:
    -----
    sim_name:           [string]
                        Name of the simulation to load (see README)
    sim_type:           [string]
                        Name of the simulation type to load (see README).
    
    Returns:
    -------
    halo_profiles:    [dict]
                      Nested dictionary containing all mass, density and velocity profiles, both 3D and 2D
    halo_props:       [dict]
                      Contains all halo IDs and the simulation regions they belong to, along with their r200 and r500 in Mpc
    sim_props:        [dict]
                      Contains the number of halos in each simulation region and in total, the mass of a particle 
                      and the cosmological parameters [h, Om, Ol, z]
    """
    
    profile_dims=["3D", "2D"]
    
    halo_profiles = {dim: dict() for dim in profile_dims}
    halo_props = dict()
    sim_props = dict()
    
    Nhalos_tot = 0
    
    basename = "progress/" + sim_name + "/savestates/" + sim_type
        
    for f, f_path in enumerate([os.path.normpath(i) for i in glob.glob(basename + "/*.npz")]):
            with np.load(f_path, allow_pickle=True) as save_state:
                profiles_saved_3D = save_state["halo_profiles_3D"].item()
                profiles_saved_2D = save_state["halo_profiles_2D"].item()
                hprops_saved = save_state["halo_props"].item()
                sprops_saved = save_state["sim_props"].item()
                
                if f == 0:
                    halo_profiles["3D"] = profiles_saved_3D
                    halo_profiles["2D"] = profiles_saved_2D
                    halo_props = hprops_saved
                    sim_props = sprops_saved
                    
                else:
                    #Append saved profiles
                    for quantity in profiles_saved_3D.keys():
                        halo_profiles["3D"][quantity] += profiles_saved_3D[quantity]
                        
                    for dim in profiles_saved_2D.keys():
                        for quantity in profiles_saved_2D[dim].keys():
                            halo_profiles["2D"][dim][quantity] += profiles_saved_2D[dim][quantity]
                                
                    for key in hprops_saved.keys():
                        halo_props[key] += hprops_saved[key]
                
                for region in list(sprops_saved["HALO_NUM_REGION"].keys()):
                    Nhalos_region = 0
                    
                    if (f == 0) and (sim_props["HALO_NUM_REGION"][region] is None):
                        sim_props["HALO_NUM_REGION"][region] = 0
                    
                    if f > 0:
                        if sprops_saved["HALO_NUM_REGION"][region] is not None:
                            Nhalos_region += sprops_saved["HALO_NUM_REGION"][region]
                            Nhalos_tot += Nhalos_region
                        
                        if region not in sim_props["HALO_NUM_REGION"]:
                            sim_props["HALO_NUM_REGION"][region] = 0
                        
                        sim_props["HALO_NUM_REGION"][region] += Nhalos_region

    sim_props["HALO_NUM_TOT"] = np.sum([sim_props["HALO_NUM_REGION"][r] for r in sim_props["HALO_NUM_REGION"].keys()], dtype=int)

    #Convert in numpy arrays
    for quantity in halo_profiles["3D"].keys():
        halo_profiles["3D"][quantity] = np.array(halo_profiles["3D"][quantity])
        
    for dim in profiles_saved_2D.keys():
        for quantity in halo_profiles["2D"][dim].keys():
            halo_profiles["2D"][dim][quantity] = np.array(halo_profiles["2D"][dim][quantity])
                
    for key in hprops_saved.keys():
        halo_props[key] = np.array(halo_props[key])
                        
    return halo_profiles, halo_props, sim_props


def GetProfiles(hdf5_path=None, sim_name=None, sim_type=None, sim_regions=[""], dimensions=[], 
                enable_savestates=False, Ntoread=None, enable_multiprocessing=False):
    """
    This function reads the HDF5 files of a given simulation and returns the corresponding profiles and simulation properties
    in nested dictionaries.
    
    Args:
    -----
    hdf5_path:                  [string]
                                Relative path of the directory containing the simulation folders (see README)
    sim_name:                   [string]
                                Name of the simulation, assigned to the top-level directory (see README)
    sim_type:                   [string]
                                Simulation model from which to extract the halo profiles (see README)
    sim_regions                 [string list]
                                Names of the region folders to read, leave empty [""] for no regions
    dimensions                  [string list]
                                Projections of the 2D profiles to extract, e.g. ["x", "y", "z"], leave empty [] for no 2D profiles
    enable_savestates           [bool]
                                True to enable saving progress after each batch reading (see README)
    Ntoread                     [int list]
                                Number of files to read at once in each region, can be a single int instead of a list to apply the same number 
                                to all regions
    enable_multiprocessing      [bool]
                                True to enable multiprocessing (see README), used in this function only for printing purposes
                                
    Returns:
    --------
    profiles:         [dict]
                      Nested dictionary containing all mass, density and velocity profiles, both 3D and 2D
    halo_props:       [dict]
                      Contains all halo IDs and the simulation regions they belong to, along with their r200 and r500 in Mpc
    sim_props:        [dict]
                      Contains the number of halos in each simulation region and in total, the mass of a particle 
                      and the cosmological parameters [h, Om, Ol, z]
    """
                
    if enable_multiprocessing:
        print("STARTED PROCESS " + str(os.getpid()) + " WORKING ON " + sim_type)

    #Define G
    G_mpc = 4.302e-9    #In Msun^-1 * Mpc * km^2 * s^-2
    
    #Halo profiles dictionaries
    halo_profiles_3D = {"MASS": [], "DENSITY": [], "VCIRC": [], "BETA": [], "NUM": [], "CUM_NUM": [], 
                     "SIGMAr": [], "SIGMAt": [], "SIGMAp": [], "ERR_MASS": [], "ERR_DENSITY": [], "ERR_VCIRC": [], "R": []}
    halo_profiles_2D = {dim: {"MASS": [], "DENSITY": [], "DEN_CUM": [], "DELTA_SIGMA": [], "NUM": [], "CUM_NUM": [], "ERR_MASS": [], 
                                   "ERR_DENSITY": [], "ERR_DEN_CUM": [], "ERR_DSIGMA": [], "R": []} for dim in dimensions}
    halo_props = {"ID": [], "REGION": [], "R500": [], "R200": []}
    sim_props = {"HALO_NUM_TOT": None, "HALO_NUM_REGION": {reg: None for reg in sim_regions}, "MPART": None, "COSM_PARS": None}

    #Define simulation properties
    Nhalos_tot = 0
    Nfiles_read = 0
    
    #Read all HDF5 files and save the profiles in a dictionary
    print("CURRENT SIMULATION: " + sim_type)
    
    #Create directory to store savestates and progress
    if not os.path.exists("progress/" + sim_name):
        os.makedirs("progress/" + sim_name)
        
    if (not os.path.exists("progress/" + sim_name + "/savestates/" + sim_type)) and enable_savestates:
        os.makedirs("progress/" + sim_name + "/savestates/" + sim_type)

    for n, region in enumerate(sim_regions):
        #Number of halos in a region
        Nhalos_region = 0
        
        #HDF5 file names
        basename = sim_name + "/" + region + "/" + sim_type
        file_names = [os.path.normpath(f) for f in glob.glob(basename + "/*.hdf5")]
        
        if enable_savestates:
            #Define the folder where all progress is saved
            distdir = os.getcwd()
            distname = "progress/" + basename
            distot = os.path.join(distdir, distname)
            
            #If the directory doesn't exist, create it
            if (not os.path.exists(distot)) and enable_savestates:
                os.makedirs(distot)

            #Path of all hdf5 files in a given simulation folder
            filenames_remaining = []

            if len(glob.glob(distname + "/" + "*.txt")) == 0:
                filenames_done = []

            else:
                filenames_done = np.loadtxt(distname + "/" + "filenames.txt", dtype="str", ndmin=1)

            if len(filenames_done) != 0:
                for fnames in file_names:
                    if (filenames_done == fnames).any() == False:
                        filenames_remaining.append(fnames)

                #Load current progress from file
                filenames_done = filenames_done.tolist()

            else:
                filenames_remaining = file_names
            
        else:
            filenames_remaining = file_names

        if Ntoread == None:
            file_max = len(filenames_remaining) + 1

        else:
            if len(np.atleast_1d(Ntoread)) == 1:
                file_max = Ntoread
                
            else:
                file_max = Ntoread[n]

    #####################################################################################################################

        for f_i, h5_file_name in enumerate(filenames_remaining[:file_max]):
            Nfiles_read += 1
            
            with h5py.File(h5_file_name, "r") as hdf:
                #Get the cosmological parameters
                h = hdf["Header"].attrs["h"]
                Om = hdf["Header"].attrs["Om"]
                Ol = hdf["Header"].attrs["Ol"]
                z = hdf["Header"].attrs["z"]
                cosm_pars = np.array([h, Om, Ol, z])

                #Only print the cosmological parameters the first time
                if h5_file_name == file_names[0]:
                    if region == sim_regions[0]:
                        if enable_multiprocessing:
                            print("Cosmology:\n h = {0:3.4f}, Om = {1:3.4f}, Ol = {2:3.4f}, z = {3:3.4f}".format(h, Om , Ol, z))
                    
                    print("\nREGION: " + region)
                    print("READING FILE:")
                    
                if enable_savestates:
                    print("\t" + h5_file_name + " ---- " + f"GLOBAL REGION PROGRESS: {len(filenames_done) + 1} / {len(file_names)}")
                    
                else:
                    print("\t" + h5_file_name)

                #Halo properties
                ids = hdf["Header/Group_IDs"][:].astype(int)    #Group IDs
                Mpart = hdf["Header"].attrs["Mpart"]    #Particle masses
                Nhalos_region += len(ids)
                Nhalos_tot += len(ids)    #Add the number of halos in the file to the total number of halos the region

                #Only save the simulation parameters the first time
                if h5_file_name == file_names[0]:
                    sim_props["COSM_PARS"] = cosm_pars
                    sim_props["MPART"] = Mpart

                #Read the halo profiles
                data = hdf["RadialProfiles"]   #Halo profiles group

                for ii in ids:
                    #Halo r500c and r200c
                    r500c = data["Group_%i_R500"%ii][:][0]
                    r200c = data["Group_%i_R200"%ii][:][0]

    #####################################################################################################################

                    #3D halo profiles and corresponding uncertainties
                    bin_centers = data["Group_%i_Radius"%ii][:]
                    MassCum = data["Group_%i_MassCum"%ii][:]    #Cumulative mass in M_sun
                    Den = data["Group_%i_Density"%ii][:] / r500c**3    #Density inside a bin in M_sun/Mpc^3
                    Vel_circ = np.sqrt(G_mpc * MassCum / bin_centers * r500c)    #Circular velocity in km/s
                    Npart = data["Group_%i_Npart"%ii][:]    #Number of particles inside a bin
                    NpartCum = data["Group_%i_NpartCum"%ii][:]    #Cumulative number of particles
                    sr2, st2, sp2 = data["Group_%i_VelocityDispersion"%ii][3:6]    #Velocity dispersions along r, theta, phi in km^2/s^2
                    
                    with np.errstate(invalid="ignore"):
                        beta_r = 1 - (st2 + sp2) / (2 * sr2)
                        beta_r = np.nan_to_num(beta_r, 0.)

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
                    with np.errstate(divide="ignore", invalid="ignore"):
                        err_vel = 0.5 * np.sqrt(G_mpc / (bin_centers * r500c)) * MassCum**(-1/2) * err_mass

                    #Save all data in dictionaries
                    saved_quantities = [MassCum, Den, Vel_circ, beta_r, Npart, NpartCum, sr2, st2, sp2, err_mass, err_den, err_vel, bin_centers]
                    
                    for key, quantity in zip(list(halo_profiles_3D.keys()), saved_quantities):
                        halo_profiles_3D[key].append(quantity)

    #####################################################################################################################
        
                    #2D halo profiles and corresponding uncertainties
                    for dim in dimensions:
                        bin_centers_2D = data["Group_%i_Radius2D"%ii + dim][:]  #In units of r500
                        MassCum_2D = data["Group_%i_MassCum2D"%ii + dim][:]    #2D cumulative mass in M_sun
                        Den_2D = data["Group_%i_Density2D"%ii + dim][:] / r500c**2    #2D density inside a bin in M_sun/Mpc^2
                        Ncum_2D = data["Group_%i_NpartCum2D"%ii + dim][:]    #2D number of particles inside a bin
                        Nbin_2D = data["Group_%i_Npart2D"%ii + dim][:]    #2D cumulative number of particles
    
                        #Compute mass Poisson errors
                        err_mass_2D = Mpart * np.sqrt(Ncum_2D)
                
                        #Compute density Poisson errors, if enabled
                        #Since the bin edges are equally spaced in log space, to compute the volume shells add half of the difference between log bin centers to 
                        #the log bin center to obtain all the bin edges except the first, which is added manually
                        bin_edges_2D = 10**np.append(np.log10(bin_centers_2D[0]) - 0.5 * np.diff(np.log10(bin_centers_2D))[0], 
                                                   np.log10(bin_centers_2D) + 0.5 * np.diff(np.log10(bin_centers_2D))[0])
                        area_bins = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
                        err_den_2D = Mpart * np.sqrt(Nbin_2D) / (area_bins * r500c**2)
    
                        #Cumulative surface density, in M_sun/Mpc^2
                        DenCum_2D = MassCum_2D / (np.pi * bin_edges_2D[1:]**2 * r500c**2)
                        err_den_cum_2D = Mpart * np.sqrt(Ncum_2D) / (np.pi * bin_edges_2D[1:]**2 * r500c**2)

                        #Delta Sigma profile, in Msun/Mpc^2
                        Dsigma = DenCum_2D - Den_2D
                        err_dsigma = np.sqrt(err_den_cum_2D**2 + err_den_2D**2)

                        #Save all data in dictionaries
                        saved_quantities_2D = [MassCum_2D, Den_2D, DenCum_2D, Dsigma, Nbin_2D, Ncum_2D, 
                                               err_mass_2D, err_den_2D, err_den_cum_2D, err_dsigma, bin_centers_2D]
                        
                        for key, quantity in zip(list(halo_profiles_2D[dim].keys()), saved_quantities_2D):
                            halo_profiles_2D[dim][key].append(quantity)

                    #Save the halo properties
                    halo_props["ID"].append(ii)
                    halo_props["REGION"].append(region)
                    halo_props["R500"].append(r500c)
                    halo_props["R200"].append(r200c)

    #####################################################################################################################

            if enable_savestates:
                #Save current file as already read
                filenames_done.append(h5_file_name)
                
                #Create a save file with the current progress
                np.savetxt("progress/" + sim_name + "/savestates/filenames.txt", filenames_done, fmt="%s")

    #####################################################################################################################

        #Save the number of halos in the region
        sim_props["HALO_NUM_REGION"][region] = Nhalos_region

    #Save the total number of halos
    sim_props["HALO_NUM_TOT"] = Nhalos_tot
    
    if enable_savestates and (Nfiles_read != 0):
        #Save current progress to file in the "progress" folder
        save_name = "save_state_" + f"{time.time():.0f}" + "_N" + str(Nfiles_read)
        np.savez("progress/" + sim_name + "/savestates/" + sim_type + "/" + save_name, halo_profiles_3D=halo_profiles_3D, 
                                                                             halo_profiles_2D=halo_profiles_2D, 
                                                                             halo_props=halo_props, sim_props=sim_props)

    print("------------------------------------------------------------------------------------\n")
    
    if enable_multiprocessing:
        print("FINISHED PROCESS " + str(os.getpid()) + " WORKING ON " + sim_type)
    
    if enable_savestates:
        return RestoreSavestates(sim_name, sim_type)
    
    else:
        #Convert the binned profiles in numpy arrays
        for key in list(halo_profiles_3D.keys()):
            halo_profiles_3D[key] = np.array(halo_profiles_3D[key])

        for dim in dimensions:
            for key in list(halo_profiles_2D[dim].keys()):
                halo_profiles_2D[dim][key] = np.array(halo_profiles_2D[dim][key])

        for key in list(halo_props.keys()):
            halo_props[key] = np.array(halo_props[key])
            
        profiles = {"3D": halo_profiles_3D, "2D": halo_profiles_2D}
        
        return profiles, halo_props, sim_props


def GetSimProfiles(hdf5_path=None, sim_name=None, simulation_type=None, sim_regions=[""], dimensions=[],
                    save_to_file=False, load_from_file=False, save_data_path="",
                    enable_savestates=False, Ntoread=None, enable_multiprocessing=False):
    """
    This function reads the HDF5 files of all supplied simulations and returns the corresponding profiles and simulation properties
    in nested dictionaries.
    It is a wrapper of the GetProfiles function, allowing for reading of multiple simulation types at once, 
    saving and loading all data to file and multiprocessing of multiple simulation types at once.
    
    Args:
    -----
    hdf5_path:                  [string]
                                Relative path of the directory containing the simulation folders (see README)
    sim_name:                   [string]
                                Name of the simulation, assigned to the top-level directory (see README)
    simulation_type:            [string]
                                Simulation model from which to extract the halo profiles (see README)
    sim_regions                 [string list]
                                Names of the region folders to read, leave empty [""] for no regions
    dimensions                  [string list]
                                Projections of the 2D profiles to extract, e.g. ["x", "y", "z"], leave empty [] for no 2D profiles
    save_to_file                [bool]
                                True to save all read data to a file, different from enable_savestates
    load_from_file              [bool]
                                Set to True to load halo profiles from a saved file instead of reading and from zero.
                                Only useful if the data was saved using save_data = True prior, otherwise does nothing
    save_data_path              [string]
                                Path to save the data to if save_data is enabled
    enable_savestates           [bool]
                                True to enable saving progress after each batch reading (see README)
    Ntoread                     [int list]
                                Number of files to read at once in each region, can be a single int instead of a list to apply the same number 
                                to all regions
    enable_multiprocessing      [bool]
                                True to enable multiprocessing (see README)

    Returns:
    --------
    halo_profiles:    [dict]
                      Nested dictionary containing all mass, density and velocity profiles, both 3D and 2D
    halo_props:       [dict]
                      Contains all halo IDs and the simulation regions they belong to, along with their r200 and r500 in Mpc
    sim_props:        [dict]
                      Contains the number of halos in each simulation region and in total, the mass of a particle 
                      and the cosmological parameters [h, Om, Ol, z]
    """

    halo_profiles = {sim_type: dict() for sim_type in simulation_type}
    halo_props = {sim_type: dict() for sim_type in simulation_type}
    sim_props = {sim_type: dict() for sim_type in simulation_type}
    
    if save_data_path == "":
        save_data_path = "progress/" + sim_name
    
    if load_from_file and os.path.isfile(save_data_path + "/halo_profiles.npz"):
        with np.load(save_data_path + "/halo_profiles.npz", allow_pickle=True) as file_saved:
            halo_profiles = file_saved["halo_profiles"].item()
            halo_props = file_saved["halo_props"].item()
            sim_props = file_saved["sim_props"].item()
            
    else:
        if enable_multiprocessing:
            with mp.Pool(len(np.atleast_1d(simulation_type))) as pool:
                results = pool.starmap(GetProfiles, [(hdf5_path, sim_name, sim_type, sim_regions, dimensions, 
                                                    enable_savestates, Ntoread, enable_multiprocessing) 
                                                    for sim_type in simulation_type])

                for sim_type, res in zip(simulation_type, results):
                    halo_profiles[sim_type], halo_props[sim_type], sim_props[sim_type] = res
        
        else:
            for sim_type in simulation_type:
                halo_profiles[sim_type],\
                halo_props[sim_type],\
                sim_props[sim_type] = GetProfiles(hdf5_path, sim_name, sim_type, sim_regions, dimensions, 
                                                    enable_savestates, Ntoread)
            
    #Save all profiles to file
    if save_to_file and ( (not load_from_file) or (not os.path.exists(save_data_path + "/halo_profiles.npz")) ):
        np.savez(save_data_path + "/halo_profiles.npz", halo_profiles=halo_profiles,
                                                        halo_props=halo_props, 
                                                        sim_props=sim_props)

    return halo_profiles, halo_props, sim_props


def RestoreSavestatesFits(sim_name, sim_type):
    """
    Function that loads all savestate data for the profile fits into dictionaries.

    Args:
    -----
    sim_name:           [string]
                        Name of the simulation name to load
    sim_type:           [string]
                        Name of the simulation type to load.
    
    Returns:
    -------
    fit_pars:         [dict]
                      Nested dictionary containing all free parameters of all fitted models, along with chi2 and M200 in Mpc, 
                      both for 3D and 2D fits
    fit_cov:          [dict]
                      Nested dictionary containing all covariances of all free parameters for evry fitted models, 
                      both for 3D and 2D fits
    """

    profile_dims=["3D", "2D"]

    fit_pars = {dim: dict() for dim in profile_dims}
    fit_cov = {dim: dict() for dim in profile_dims}
    
    basename = "progress/" + sim_name + "/savestates_fits/" + sim_type
    
    for n_dim in profile_dims:
        for f_path in [os.path.normpath(i) for i in glob.glob(basename + "/" + n_dim + "/*.npz")]:
                with np.load(f_path, allow_pickle=True) as save_state:
                    fit_pars_saved = save_state["fit_pars"].item()
                    fit_cov_saved = save_state["fit_cov"].item()
                    
                    if n_dim == "3D":
                        fit_pars[n_dim] = fit_pars_saved
                        fit_cov[n_dim] = fit_cov_saved
                        
                    elif n_dim == "2D":
                        dim = f_path[-5]
                        fit_pars[n_dim][dim] = fit_pars_saved
                        fit_cov[n_dim][dim] = fit_cov_saved

    return fit_pars, fit_cov


def FitProfiles(binned_profiles, profile_errors, bin_centers, num_profiles, R500, 
                  cosm_params, radius_fit_bounds=[(0., np.inf)], profile_type=["NFW"], profile_quantity=["MASS"], projection=False):
    """
    This function performs 3D and 2D fits of halo profiles from a supplied simulation type.
    The fits are performed using non-linear least squares through the SciPy function curve_fit.
    
    Args:
    -----
    binned_profiles:          [array]
                              Array of halo profiles to fit, either 3D or 2D mass, density or circular velocity profiles
    profile_errors:           [array]
                              Array of binned profiles uncertainties, either for 3D or 2D mass, density or circular velocity profiles
    bin_centers:              [array]
                              Array of radial bin centers associated to the binned profiles, in units of r500
    num_profiles:             [array]
                              Array of number of particles in each radial bin, used to discard any empty bins when fitting
    R500:                     [array]
                              Array of halo r500, in Mpc
    cosm_params:              [list]
                              Cosmological parameters [h, Om, Ol, z] for the supplied simulation type
    radius_fit_bounds:        [array_like of tuples]
                              Radius interval in which to fit, specified for each model to fit.
                              Each tuple should contain the lower and upper radial bounds over which to fit
    profile_type:             [string list]
                              List containing the types of halo models to fit, e.g. ["NFW", "gNFW"]
    profile_quantity          [string list]
                              List containing the quantities to fit, e.g. ["MASS", "DENSITY"]
    projection                [bool]
                              Set to True to fit the corresponding profile_type projected profile, instead of the 3D ones
                              
    Returns:
    --------
    fit_pars:         [dict]
                      Nested dictionary containing all free parameters of all fitted models, along with chi2 and M200 in Mpc
    fit_cov:          [dict]
                      Nested dictionary containing all covariances of all free parameters for evry fitted models
    """
    
    #Define G
    G_mpc = 4.302e-9    #In Msun^-1 * Mpc * km^2 * s^-2
    
    #Define the profiles to fit for every quantity specified
    binned_profiles_fit = {q: {"PROFILES": binned_profiles[i], "ERRORS": profile_errors[i]} for i, q in enumerate(profile_quantity)}
    
    #Dicts containing all NFW and gNFW fit parameters, chi2 and uncertainties for mass and, if enabled, density and circular velocity profiles
    fit_pars = {p_type: {quantity: dict() for quantity in profile_quantity} for p_type in profile_type}
    fit_cov = {p_type: {quantity: dict() for quantity in profile_quantity} for p_type in profile_type}

    #Get cosmological parameters
    Nhalos = int(binned_profiles[0].shape[0])
    h, Om, Ol, z = cosm_params
    H2z = (100 * h)**2 * (Om * (1 + z)**3 + Ol)

    for quantity in profile_quantity:
        print("\nNOW FITTING: " + quantity)
        
        for p_type in profile_type:
            halo_model = HaloModel(p_type)
            free_pars = halo_model.free_par_names
            
            fit_pars[p_type][quantity] = {p_name: [] for p_name in free_pars}
            fit_pars[p_type][quantity].update({"chi2": [], "M200": []})
            
            fit_cov[p_type][quantity] = {p_name: [] for p_name in free_pars}
            
            for i, p1 in enumerate(free_pars): 
                for j, p2 in enumerate(free_pars):
                    if j > i:
                        fit_cov[p_type][quantity].update({p1 + "_" + p2: []})
            
        #Fit all halo profiles
        for profile, profile_err, radius, num, r500, id_count in zip(binned_profiles_fit[quantity]["PROFILES"], 
                                                                     binned_profiles_fit[quantity]["ERRORS"], 
                                                                     bin_centers, num_profiles, R500, np.arange(0, len(R500), dtype=int)):
            
            if int(id_count / Nhalos * 100) % 5 == 0:
                sys.stdout.write("\r")
                sys.stdout.write(f"PROGRESS: {id_count / Nhalos * 100:.0f}%")
                sys.stdout.flush()

            #Store the fit parameters, covariance matrix and chi2 for a single fit
            fit_parameters_dict = {p_type: {q: {"popt": None, "cov": None, "chi2": None} for q in profile_quantity} 
                                   for p_type in profile_type}

            #Initial guess for curve_fit
            initial_pars = [[np.log10(1.5*r500), np.log10(0.5*r500)], [np.log10(1.5*r500), np.log10(0.5*r500), 0.]]

            for p_type, par_0, rad_bounds in zip(profile_type, initial_pars, radius_fit_bounds):
                #Condition on radius: select only R (in units of r500) inside the fit_bounds, and not corresponding to an empty bin
                R_cond = (radius >= rad_bounds[0]) & (radius <= rad_bounds[1]) & (num != 0)

                #Define the NFW and gNFW fit functions with the current cosmology
                halo_model = HaloModel(p_type)
                
                fit_func = lambda lr, *free_p: np.log10(halo_model._profile(lr, *free_p,
                                                        cosm_params=cosm_params, quantity=quantity, 
                                                        projection=projection))

                y_data = np.log10(profile[R_cond])
                y_err = profile_err[R_cond] / profile[R_cond] / np.log(10)
                x_data = np.log10(radius[R_cond] * r500)

                #Fit the profiles
                popt, cov = curve_fit(fit_func, x_data, y_data, sigma=y_err, p0=par_0, absolute_sigma=True, 
                                        bounds=halo_model.fit_bounds, max_nfev=10000)

                #Compute chi2 per degree of freedom
                ndof = len(x_data) - len(popt)
                y_fit = fit_func(x_data, *popt) 

                fit_parameters_dict[p_type][quantity]["popt"] = popt
                fit_parameters_dict[p_type][quantity]["cov"] = cov
                fit_parameters_dict[p_type][quantity]["chi2"] = np.sum(((y_data - y_fit) / y_err)**2) / ndof


                #Save the fit parameters and uncertainties, the chi2 and M200
                fit_result = [*fit_parameters_dict[p_type][quantity]["popt"], 
                               fit_parameters_dict[p_type][quantity]["chi2"],
                               (10**fit_parameters_dict[p_type][quantity]["popt"][0])**3 * 100 * H2z / G_mpc]

                cov_result = [*np.diag(fit_parameters_dict[p_type][quantity]["cov"])]
                           
                for i in range(fit_parameters_dict[p_type][quantity]["cov"].shape[0]):
                    for j in range(fit_parameters_dict[p_type][quantity]["cov"].shape[1]):
                        if j > i:
                            cov_result.append(fit_parameters_dict[p_type][quantity]["cov"][i, j])

                #Store the fit parameters in a dictionary
                for p_i, key in enumerate(list(fit_pars[p_type][quantity].keys())):
                    fit_pars[p_type][quantity][key].append(fit_result[p_i])

                #Store the fit covariances in a dictionary
                for p_i, key in enumerate(list(fit_cov[p_type][quantity].keys())):
                    fit_cov[p_type][quantity][key].append(cov_result[p_i])

    #Convert all parameter uncertainty lists inside the dictionary in numpy arrays 
    for quantity in profile_quantity:
        for p_type in profile_type:
            for key in list(fit_pars[p_type][quantity].keys()):
                fit_pars[p_type][quantity][key] = np.array(fit_pars[p_type][quantity][key])
    
            for key in list(fit_cov[p_type][quantity].keys()):
                fit_cov[p_type][quantity][key] = np.array(fit_cov[p_type][quantity][key])

    print("\n-------------------------------------------------------")

    return fit_pars, fit_cov


def FitSimProfiles(halo_profiles, halo_props, sim_props, simulation_type, simulation_name=None,
                    profile_type_3D=["NFW"], profile_type_2D=["NFW"], 
                    fit_quantities_3D=["MASS"], fit_quantities_2D=["MASS"],
                    radius_fit_bounds_3D=[(0., np.inf)], radius_fit_bounds_2D=[(0., np.inf)],
                    n_dim_fits=["3D", "2D"], dimensions=["x", "y", "z"],
                    save_to_file=False, load_from_file=False, save_data_path="", enable_savestates=False,
                    enable_multiprocessing=False):
    """
    This function performs 3D and 2D fits of halo profiles from a supplied simulation type.
    The fits are performed using non-linear least squares through the SciPy function curve_fit.
    It is a wrapper for the FitProfiles function, allowing for reading of multiple simulation types at once, 
    saving and loading all data to file, and creation of savestates.
    
    Args:
    -----
    halo_profiles:              [dict]
                                Nested dictionary containing all mass, density and velocity profiles, both 3D and 2D
    halo_props:                 [dict]
                                Contains all halo IDs and the simulation regions they belong to, along with their r200 and r500 in Mpc
    sim_props:                  [dict]
                                Contains the number of halos in each simulation region and in total, the mass of a particle 
                                and the cosmological parameters [h, Om, Ol, z]
    simulation_type:            [string list]
                                List containing the simulation types to consider (see README)
    simulation_name:            [array]
                                Name of the simulation to consider (see README)
    profile_type_3D:            [string list]
                                List containing the types of 3D halo models to fit, e.g. ["NFW", "gNFW"]
    profile_type_2D:            [string list]
                                List containing the types of 2D halo models to fit, e.g. ["NFW", "gNFW"]
    fit_quantities_3D:          [string list]
                                List containing the 3D quantities to fit, e.g. ["MASS", "DENSITY"]
    fit_quantities_2D:          [string list]
                                List containing the 2D quantities to fit, e.g. ["MASS", "DENSITY"]
    radius_fit_bounds_3D:       [array_like of tuples]
                                Radius interval in which to fit, specified for each 3D model to fit.
                                Each tuple should contain the lower and upper radial bounds over which to fit
    radius_fit_bounds_2D:       [array_like of tuples]
                                Radius interval in which to fit, specified for each 2D model to fit.
                                Each tuple should contain the lower and upper radial bounds over which to fit
    n_dim_fits:                 [string list]
                                Dimensions to consider, e.g. ["3D", "2D"]
    dimensions                  [string list]
                                Projections of the 2D profiles to consider, e.g. ["x", "y", "z"], leave empty [] for no 2D profiles
    save_to_file                [bool]
                                True to save all read data to a file, different from enable_savestates
    load_from_file              [bool]
                                Set to True to load halo profiles from a saved file instead of reading and from zero.
                                Only useful if the data was saved using save_data = True prior, otherwise does nothing
    save_data_path              [string]
                                Path to save the data to if save_data is enabled
    enable_savestates           [bool]
                                True to enable saving progress after each batch reading (see README)
    Ntoread                     [int list]
                                Number of files to read at once in each region, can be a single int instead of a list to apply the same number 
                                to all regions
    enable_multiprocessing      [bool]
                                True to enable multiprocessing (see README), used in this function for printing purposes only

    Returns:
    --------
    fit_pars:         [dict]
                      Nested dictionary containing all free parameters of all fitted models, along with chi2 and M200 in Mpc
    fit_cov:          [dict]
                      Nested dictionary containing all covariances of all free parameters for evry fitted models
    """
                
    if enable_multiprocessing:
        print("STARTED PROCESS " + str(os.getpid()) + " WORKING ON " + simulation_type[0])
                
    fit_pars = {sim_type: {"3D": dict(), "2D": {dim: dict() for dim in dimensions}} for sim_type in simulation_type}
    fit_cov = {sim_type: {"3D": dict(), "2D": {dim: dict() for dim in dimensions}} for sim_type in simulation_type}
    
    if save_data_path == "":
        save_data_path = "progress/" + simulation_name

    if load_from_file and os.path.isfile(save_data_path + "/halo_fits.npz"):
        with np.load(save_data_path + "/halo_fits.npz", allow_pickle=True) as file_saved:
            fit_pars = file_saved["fit_pars"].item()
            fit_cov = file_saved["fit_cov"].item()

    else:
        for sim_type in simulation_type:
            dirpath = "progress/" + simulation_name + "/savestates_fits/" + sim_type

            print("CURRENT SIMULATION: " + sim_type)
            
            #Create directory to store savestates
            for n_dim in n_dim_fits:
                if (not os.path.exists(dirpath + "/" + n_dim)) and enable_savestates:
                    os.makedirs(dirpath + "/" + n_dim)

            R500 = halo_props[sim_type]["R500"]
            cosm_params = sim_props[sim_type]["COSM_PARS"]
            
            for n_dim in n_dim_fits:
                binned_profiles = halo_profiles[sim_type][n_dim]
                
                if n_dim == "3D":
                    save_name = "save_state_3D.npz"

                    if (not os.path.isfile(dirpath + "/" + n_dim + "/" + save_name)):
                        print("DIMENSION: 3D")
                        
                        binned_profiles_fit = [binned_profiles[quantity] for quantity in fit_quantities]
                        binned_errors_fit = [binned_profiles["ERR_" + quantity] for quantity in fit_quantities]
                        
                        fit_pars[sim_type][n_dim],\
                        fit_cov[sim_type][n_dim] = FitProfiles(binned_profiles_fit, binned_errors_fit, 
                                                                binned_profiles["R"], binned_profiles["NUM"], R500, 
                                                                cosm_params, radius_fit_bounds_3D, profile_type_3D, 
                                                                profile_quantity=fit_quantities_3D, projection=False)
                                                            
                    if (not os.path.isfile(dirpath + "/" + n_dim + "/" + save_name)) and enable_savestates:
                        #Save current progress to file in the "progress" folder
                        np.savez(dirpath + "/" + n_dim + "/" + save_name, fit_pars=fit_pars[sim_type][n_dim], 
                                                                            fit_cov=fit_cov[sim_type][n_dim])

                elif n_dim == "2D":
                    for dim in dimensions:
                        save_name = "save_state_2D" + dim + ".npz"
                    
                        if (not os.path.isfile(dirpath + "/" + n_dim + "/" + save_name)):
                            print("DIMENSION: 2D" + dim)
                            
                            binned_profiles_fit = [binned_profiles[dim][quantity] for quantity in fit_quantities]
                            binned_errors_fit = [binned_profiles[dim]["ERR_" + quantity] for quantity in fit_quantities]
                            
                            fit_pars[sim_type][n_dim][dim],\
                            fit_cov[sim_type][n_dim][dim] = FitProfiles(binned_profiles_fit, binned_errors_fit, 
                                                                        binned_profiles[dim]["R"], binned_profiles[dim]["NUM"], 
                                                                        R500, cosm_params, radius_fit_bounds_2D, profile_type_2D, 
                                                                        profile_quantity=fit_quantities_2D, projection=True)
                                                                        
                        if (not os.path.isfile(dirpath + "/" + n_dim + "/" + save_name)) and enable_savestates:
                            #Save current progress to file in the "progress" folder
                            np.savez(dirpath + "/" + n_dim + "/" + save_name, fit_pars=fit_pars[sim_type][n_dim][dim],
                                                                            fit_cov=fit_cov[sim_type][n_dim][dim])

            if enable_savestates and os.path.exists(dirpath):
                fit_pars[sim_type], fit_cov[sim_type] = RestoreSavestatesFits(simulation_name, sim_type)

    if save_to_file and ( (not load_from_file) or (not os.path.isfile(save_data_path + "/halo_fits.npz")) ):
        np.savez(save_data_path + "/halo_fits.npz", fit_pars=fit_pars, fit_cov=fit_cov)
        
    if enable_multiprocessing:
        print("FINISHED PROCESS " + str(os.getpid()) + " WORKING ON " + simulation_type[0])

    return fit_pars, fit_cov


def FitSimProfilesMP(halo_profiles, halo_props, sim_props, simulation_type, simulation_name=None,
                    profile_type_3D=["NFW"], profile_type_2D=["NFW"],
                    fit_quantities_3D=["MASS"], fit_quantities_2D=["MASS"],
                    radius_fit_bounds_3D=[(0., np.inf)], radius_fit_bounds_2D=[(0., np.inf)],
                    n_dim_fits=["3D", "2D"], dimensions=["x", "y", "z"],
                    save_to_file=False, load_from_file=False, save_data_path="", enable_savestates=False, 
                    enable_multiprocessing=False):
    """
    This function performs 3D and 2D fits of halo profiles from a supplied simulation type.
    The fits are performed using non-linear least squares through the SciPy function curve_fit.
    It is a wrapper for the FitSimProfiles function, allowing for multiprocessing of multiple simulation types at once.
    
    Args:
    -----
    halo_profiles:              [dict]
                                Nested dictionary containing all mass, density and velocity profiles, both 3D and 2D
    halo_props:                 [dict]
                                Contains all halo IDs and the simulation regions they belong to, along with their r200 and r500 in Mpc
    sim_props:                  [dict]
                                Contains the number of halos in each simulation region and in total, the mass of a particle 
                                and the cosmological parameters [h, Om, Ol, z]
    simulation_type:            [string list]
                                List containing the simulation types to consider (see README)
    simulation_name:            [array]
                                Name of the simulation to consider (see README)
    profile_type_3D:            [string list]
                                List containing the types of 3D halo models to fit, e.g. ["NFW", "gNFW"]
    profile_type_2D:            [string list]
                                List containing the types of 2D halo models to fit, e.g. ["NFW", "gNFW"]
    fit_quantities_3D:          [string list]
                                List containing the 3D quantities to fit, e.g. ["MASS", "DENSITY"]
    fit_quantities_2D:          [string list]
                                List containing the 2D quantities to fit, e.g. ["MASS", "DENSITY"]
    radius_fit_bounds_3D:       [array_like of tuples]
                                Radius interval in which to fit, specified for each 3D model to fit.
                                Each tuple should contain the lower and upper radial bounds over which to fit
    radius_fit_bounds_2D:       [array_like of tuples]
                                Radius interval in which to fit, specified for each 2D model to fit.
                                Each tuple should contain the lower and upper radial bounds over which to fit
    n_dim_fits:                 [string list]
                                Dimensions to consider, e.g. ["3D", "2D"]
    dimensions                  [string list]
                                Projections of the 2D profiles to consider, e.g. ["x", "y", "z"], leave empty [] for no 2D profiles
    save_to_file                [bool]
                                True to save all read data to a file, different from enable_savestates
    load_from_file              [bool]
                                Set to True to load halo profiles from a saved file instead of reading and from zero.
                                Only useful if the data was saved using save_data = True prior, otherwise does nothing
    save_data_path              [string]
                                Path to save the data to if save_data is enabled
    enable_savestates           [bool]
                                True to enable saving progress after each batch reading (see README)
    Ntoread                     [int list]
                                Number of files to read at once in each region, can be a single int instead of a list to apply the same number 
                                to all regions
    enable_multiprocessing      [bool]
                                True to enable multiprocessing (see README)

    Returns:
    --------
    fit_pars:         [dict]
                      Nested dictionary containing all free parameters of all fitted models, along with chi2 and M200 in Mpc
    fit_cov:          [dict]
                      Nested dictionary containing all covariances of all free parameters for evry fitted models
    """
                    
    fit_pars = dict()
    fit_cov = dict()
                    
    if enable_multiprocessing:
        if save_data_path == "":
            save_data_path = "progress/" + simulation_name

        if load_from_file and os.path.isfile(save_data_path + "/halo_fits.npz"):
            with np.load(save_data_path + "/halo_fits.npz", allow_pickle=True) as file_saved:
                fit_pars = file_saved["fit_pars"].item()
                fit_cov = file_saved["fit_cov"].item()
            
        with mp.Pool(len(simulation_type)) as pool:
            results = pool.starmap(FitSimProfiles, [(halo_profiles, halo_props, sim_props, [sim_type], simulation_name,
                                                    profile_type_3D, profile_type_2D, fit_quantities_3D, fit_quantities_2D,
                                                    radius_fit_bounds_3D, radius_fit_bounds_2D,
                                                    n_dim_fits, dimensions,
                                                    False, False, "",
                                                    enable_savestates, enable_multiprocessing) 
                                                    for sim_type in simulation_type])
                                                    
            for sim_type, res in zip(simulation_type, results):
                fit_pars[sim_type] = res[0][sim_type]
                fit_cov[sim_type] = res[1][sim_type]
            
        if save_to_file and ( (not load_from_file) or (not os.path.isfile(save_data_path + "/halo_fits.npz")) ):
            np.savez(save_data_path + "/halo_fits.npz", fit_pars=fit_pars, fit_cov=fit_cov)
            
    else:
        fit_pars, fit_cov = HaloReadH5.FitSimProfiles(halo_profiles, halo_props, sim_props, simulation_type, simulation_name,
                                                      profile_type_3D, profile_type_2D, fit_quantities_3D, fit_quantities_2D,
                                                      radius_fit_bounds_3D, radius_fit_bounds_2D,
                                                      n_dim_fits, dimensions,
                                                      save_data, load_from_file, save_data_path, 
                                                      enable_savestates, enable_multiprocessing)
                                                      
    return fit_pars, fit_cov


def ApplyCondition(halo_profiles, fit_pars, fit_cov, halo_props, sim_props, condition, sim_type, dimensions=[]):
    """
    Function that applies a supplied condition to every dictionary of a given simulation.
    
    Args:
    -----
    halo_profiles:    [dict]
                      Nested dictionary containing all mass, density and velocity profiles, both 3D and 2D
    fit_pars:         [dict]
                      Nested dictionary containing all free parameters of all fitted models, along with chi2 and M200 in Mpc, 
                      both for 3D and 2D fits
    fit_cov:          [dict]
                      Nested dictionary containing all covariances of all free parameters for evry fitted models, 
                      both for 3D and 2D fits
    halo_props:       [dict]
                      Contains all halo IDs and the simulation regions they belong to, along with their r200 and r500 in Mpc
    sim_props:        [dict]
                      Contains the number of halos in each simulation region and in total, the mass of a particle 
                      and the cosmological parameters [h, Om, Ol, z]
    condition         [bool array]
                      Mask to apply to the samples, must have the same dimensions of the halo profile arrays
    sim_type:         [string]
                      Simulation to apply the condiditon to
    dimensions:       [string list]
                      Projections of the 2D profiles, e.g. ["x", "y", "z"], leave empty [] for no 2D profiles
                      
    Returns:
    -------
    out_profiles      [dict]
                      Similar to halo_profiles, but after applying the condition
    out_pars          [dict]
                      Similar to fit_pars, but after applying the condition
    out_cov           dict]
                      Similar to fit_cov, but after applying the condition
    out_props         [dict]
                      Similar to halo_props, but after applying the condition
    out_sim_props     [dict]
                      Similar to sim_props, but after applying the condition
    """

    #Define new profiles
    out_profiles = {sim_type: {"3D": dict(), "2D": {dim: dict() for dim in dimensions}}}
    out_pars = {sim_type: {"3D": dict(), "2D": {dim: dict() for dim in dimensions}}}
    out_cov = {sim_type: {"3D": dict(), "2D": {dim: dict() for dim in dimensions}}}
    out_props = {sim_type: dict()}
    out_sim_props = {sim_type: dict.fromkeys(sim_props[sim_type], dict())}
    
    #Apply condition
    n_dim_fits = list(halo_profiles[sim_type].keys())

    #3D profiles
    if "3D" in n_dim_fits:
        for key in list(halo_profiles[sim_type]["3D"].keys()):
            out_profiles[sim_type]["3D"][key] = halo_profiles[sim_type]["3D"][key][condition]
        
        #3D fits
        for p_type in list(fit_pars[sim_type]["3D"].keys()):
            profile_quantity = list(fit_pars[sim_type]["3D"][p_type].keys())
            
            out_pars[sim_type]["3D"][p_type] = dict()
            out_cov[sim_type]["3D"][p_type] = dict()
            
            for quantity in profile_quantity:
                out_pars[sim_type]["3D"][p_type][quantity] = dict()
                out_cov[sim_type]["3D"][p_type][quantity] = dict()
                
                for key in list(fit_pars[sim_type]["3D"][p_type][quantity].keys()):
                    out_pars[sim_type]["3D"][p_type][quantity][key] = fit_pars[sim_type]["3D"][p_type][quantity][key][condition]

                for key in list(fit_cov[sim_type]["3D"][p_type][quantity].keys()):
                    out_cov[sim_type]["3D"][p_type][quantity][key] = fit_cov[sim_type]["3D"][p_type][quantity][key][condition]

    #2D profiles
    for dim in dimensions:
        for key in list(halo_profiles[sim_type]["2D"][dim].keys()):
            out_profiles[sim_type]["2D"][dim][key] = halo_profiles[sim_type]["2D"][dim][key][condition]

        if "2D" in n_dim_fits:
            #2D fits
            for p_type in list(fit_pars[sim_type]["2D"][dim].keys()):
                profile_quantity = list(fit_pars[sim_type]["2D"][dim][p_type].keys())
                
                out_pars[sim_type]["2D"][dim][p_type] = dict()
                out_cov[sim_type]["2D"][dim][p_type] = dict()
            
                for quantity in profile_quantity:
                    out_pars[sim_type]["2D"][dim][p_type][quantity] = dict()
                    out_cov[sim_type]["2D"][dim][p_type][quantity] = dict()
                
                    for key in list(fit_pars[sim_type]["2D"][dim][p_type][quantity].keys()):
                        out_pars[sim_type]["2D"][dim][p_type][quantity][key] = fit_pars[sim_type]["2D"][dim][p_type][quantity][key][condition]
                
                    for key in list(fit_cov[sim_type]["2D"][dim][p_type][quantity].keys()):
                        out_cov[sim_type]["2D"][dim][p_type][quantity][key] = fit_cov[sim_type]["2D"][dim][p_type][quantity][key][condition]

    #Halo and simulation properties
    for key in list(halo_props[sim_type].keys()):
        out_props[sim_type][key] = halo_props[sim_type][key][condition]

    for key in list(sim_props[sim_type].keys()):
        if key == "HALO_NUM_REGION":
            for region in list(sim_props[sim_type]["HALO_NUM_REGION"].keys()):
                Nhalos_region = np.sum(out_props[sim_type]["REGION"] == region)
                out_sim_props[sim_type]["HALO_NUM_REGION"][region] = Nhalos_region
                
        elif key == "HALO_NUM_TOT":
            out_sim_props[sim_type]["HALO_NUM_TOT"] = len(out_props[sim_type]["ID"])
            
        else:
            out_sim_props[sim_type][key] = sim_props[sim_type]["COSM_PARS"]

    return out_profiles, out_pars, out_cov, out_props, out_sim_props

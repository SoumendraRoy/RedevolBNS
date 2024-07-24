import numpy as np
import h5py
from astropy.table import Table
from scipy import stats
import bounded_3d_kde as bounded_3d_kde
from astropy.cosmology import Planck18
import astropy.units as u
import corner
import matplotlib.pyplot as plt

seed = 102312328

def read_DCO(var, rate_key, redshift_interest  = 0.2):
    
    ####################################
    data_dir = '/mnt/home/lvanson/ceph/CompasOutput/v02.35.02/'

    # Read the rates data
    RateFile = h5py.File(data_dir+var+'/CosmicIntegration'+'/Rate_info.h5', 'r')    

    # Print the keys in the simulation data
    print(RateFile[rate_key].keys())

    # Print the shape of each dataset in the simulation data
    for key in RateFile[rate_key].keys():
        print(np.shape(RateFile[rate_key][key][()]))

    ####################################
    # Read COMPAS simulation data
    dataFile = h5py.File(data_dir+var+'/MainRun'+'/COMPAS_Output_wWeights.h5', 'r')

    # Set the key for the double compact object (DCO) data
    DCO_key = 'BSE_Double_Compact_Objects'
    # Set the list of keys to extract from the DCO data
    key_list = ['SEED', 'Mass(1)','Mass(2)','mixture_weight', 'Stellar_Type(1)', 'Stellar_Type(2)', 'Merges_Hubble_Time'] #'Metallicity@ZAMS(1)', 'Immediate_RLOF>CE', 'Optimistic_CE',

    # Extract DCO data to astropy table
    DCO = Table()
    for key in key_list:
        DCO[key] = dataFile[DCO_key][key][()]
    # Calculate the more massive and less massive component masses and chirp mass
    DCO['M_moreMassive'] = np.maximum(dataFile[DCO_key]['Mass(1)'][()], dataFile[DCO_key]['Mass(2)'][()])
    DCO['M_lessMassive'] = np.minimum(dataFile[DCO_key]['Mass(1)'][()], dataFile[DCO_key]['Mass(2)'][()])
    
    # Reduce the DCO table to only NSNS that merge in a Hubble time
    DCO_merger_calculated = DCO[RateFile[rate_key]['DCOmask'][()]] #RateFile[rate_key]['DCOmask'][()] reduces BSE_Double_Compact_Objects to the same shape as RateFile

    NSNS_bool = np.logical_and(DCO_merger_calculated['Stellar_Type(1)'] == 13, DCO_merger_calculated['Stellar_Type(2)'] == 13)
    
    print(f'You might not have always calculated all rates! sum NSNS_bool = {sum(NSNS_bool)}')

    merger_bool = DCO_merger_calculated['Merges_Hubble_Time'] == True


    NSNStable  = DCO_merger_calculated[np.logical_and(NSNS_bool, merger_bool)]
    rates_NSNS = RateFile[rate_key]['merger_rate'][np.logical_and(NSNS_bool, merger_bool)]

    # Extract redshift for the current frame
    redshifts = RateFile[rate_key]['redshifts'][()]
    # Get the redshift index closest to the redshift of interest
    redshift_i = np.argmin(np.abs(redshifts - redshift_interest))

    return NSNStable, rates_NSNS, redshifts, redshift_i

def samples(var, rate_key):
    NSNStable, rate, z, _ = read_DCO(var, rate_key)
    m1 = np.array(NSNStable['M_moreMassive'])
    m2 = np.array(NSNStable['M_lessMassive'])
    z+=0.001
    
    Mc = (m1*m2)**0.6/(m1+m2)**0.2
    logq = np.log(m2)-np.log(m1)
    print('logq>0:', np.where(logq>0))
    
    lenz = len(z)
    lenm = len(Mc)
    zs = np.repeat(z, lenm)
    Mcs = np.tile(Mc, lenz)
    logqs = np.tile(logq, lenz)
    rates = rate.flatten()
    
    weights = rates*(Planck18.differential_comoving_volume(zs).to(u.Gpc**3/u.sr).value/(1+zs))
    kernel = bounded_3d_kde.Bounded_3d_kde(np.vstack([Mcs, logqs, zs]).T, 
                                        low=[-np.inf,-np.inf,0.0], 
                                        high=[np.inf,0.0,np.inf],
                                        weights=weights)
    Mcinj_100000, logqinj_100000, zinj_100000 = kernel.resample(size=100000)
    return Mcinj_100000, logqinj_100000, zinj_100000

variant = ['N1e7_Fiducial_AllDCO_AIS', 'N1e7_CEa025_AllDCO_AIS', 'N1e7_CEa05_AllDCO_AIS', 'N1e7_CEa075_AllDCO_AIS', 'N1e7_CEa2_AllDCO_AIS', 'N1e7_CEa5_AllDCO_AIS', 'N1e7_FRYER2022_AllDCO_AIS', 'N1e7_MullerMandel_AllDCO_AIS', 'N1e7_zetaHG5_AllDCO_AIS', 'N1e7_zetaHG5d5_AllDCO_AIS', 'N1e7_zetaHG6_AllDCO_AIS', 'N1e7_beta025_AllDCO_AIS', 'N1e7_beta05_AllDCO_AIS', 'N1e7_beta075_AllDCO_AIS', 'N1e7_beta1_AllDCO_AIS', 'N1e7_ECSN10_AllDCO_AIS', 'N1e7_ECSN200_AllDCO_AIS']

with h5py.File("variant_pop.h5", "w") as file:
    file.create_dataset('variant', data=variant)
    for i in range(len(variant)):
        var = variant[i]
        print('rate_key=', var, i)

        if var=='N1e7_Fiducial_AllDCO_AIS':
            Mc, logq, z = samples(var, rate_key = 'Rates_mu00.025_muz-0.049_alpha-1.79_sigma01.129_sigmaz0.048_a0.017_b1.487_c4.442_d5.886')
        else:
            Mc, logq, z = samples(var, rate_key = 'Rates_mu00.025_muz-0.049_alpha-1.79_sigma01.129_sigmaz0.048_a0.017_b1.487_c4.442_d5.886_zBinned')
    
        file.create_dataset('Mc_'+var, data=Mc)
        file.create_dataset('logq_'+var, data=logq)
        file.create_dataset('z_'+var, data=z)
file.close()
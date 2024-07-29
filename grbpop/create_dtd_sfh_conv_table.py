import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
from tqdm import tqdm

def MD14_SFH(z,a,b,zp):
    """
    Smoothly broken power law event rate density, with functional form given
    in Eq. 9 in Salafia+2023. This is the same functional form as the Madau &
    Dickinson 2014 cosmic star formation history fitting formula.
    """
    return (1.+z)**a/(1.+((1.+z)/(1.+zp))**(b+a))

def lognormal(td, mu_td, sigma_td):
    """
    Lognormal distribution for the time delay. The unit for the time delay is Gyr.
    """
    return np.exp(-((np.log(td)-np.log(mu_td))**2)/(2*sigma_td**2))/(np.sqrt(2*np.pi)*sigma_td*td)


# Star formation history parameters (Madau & Fragos 2017)
a = 2.6
b = 3.6
zp = 2.2

# Setting the grid for the computation
at = np.linspace(0,3,10)
td_spacing = 0.01
tdmin_max = 3. # Max minimum time delay
z_max = 11. # Max redshift for the distribution

print('Computing grid of time delays and formation redshifts...')

# Grid of time delays
td_grid = np.arange(0,cosmo.lookback_time(1000).value,td_spacing)

# Grid of redshifts corresponding to the time delays, i.e. formation redshifts
zf = np.zeros_like(td_grid)
zf[1:] = z_at_value(cosmo.lookback_time, td_grid[1:]*u.Gyr)

# Grid of minimum time delays
tdmin = td_grid[td_grid<(tdmin_max + td_spacing/2)] # The td_spacing/2 is to avoid array boundary problems
tdmin = tdmin[1:] # Exclude tdmin = 0

z = zf[zf<=z_max] # Redshifts for the distribution
r_sgrb_pow = np.zeros([len(z),len(tdmin),len(at)])


print('Computing convolutions with power-law time delay distribution...')

for k in tqdm(range(len(tdmin))):
    td_index = int(tdmin[k]/td_spacing) # Array index corresponding to the minimum time delay
    for q in range(len(at)):
        dtd_norm = np.trapz(td_grid[td_index:]**(-at[q]), td_grid[td_index:]) # Delay time distribution normalization
        for i in range(len(z)):
            if i==0:
                td_int = td_grid[td_index:]
            else:
                td_int = td_grid[td_index:-i] # Indexing for convolution integral
            zf_index = i + td_index
            zf_int = zf[zf_index:] # Indexing for convolution integral
            dtd = td_int**(-at[q])/dtd_norm # Delay time distribution
            sfh = MD14_SFH(zf_int,a=a,b=b,zp=zp) # Star formation history
            r_sgrb_pow[i][k][q] = np.trapz(sfh*dtd/(1+zf_int), td_int) # Convolution integral



print('Computing convolutions with log-normal time delay distrbution...')

# Lognormal distribution parameters
mu_td = np.linspace(0.001,3,25)
sigma_td = np.linspace(0.001,3,25)
r_sgrb_log = np.zeros([len(z),len(mu_td),len(sigma_td)])

for i in tqdm(range(len(z))):
    for k in range(len(mu_td)):
        for q in range(len(sigma_td)):
            if i == 0:
                td_int = td_grid[1:]
            else:
                td_int = td_grid[1:-i] # Indexing for convolution integral
            zf_int = zf[(i+1):] # Indexing for convolution integral
            dtd = lognormal(td_int,mu_td=mu_td[k],sigma_td=sigma_td[q]) # Delay time distribution
            sfh = MD14_SFH(zf_int,a=a,b=b,zp=zp) # Star formation history
            r_sgrb_log[i][k][q] = np.trapz(sfh*dtd/(1+zf_int), td_int) # Convolution integral



print('Saving grids...')

np.save('dtd_sfh_conv_tables/z.npy',z)

np.save('dtd_sfh_conv_tables/at.npy',at)
np.save('dtd_sfh_conv_tables/tdmin.npy',tdmin)
np.save('dtd_sfh_conv_tables/r_sgrb_pow.npy',r_sgrb_pow)

np.save('dtd_sfh_conv_tables/mu_td.npy',mu_td)
np.save('dtd_sfh_conv_tables/sigma_td.npy',sigma_td)
np.save('dtd_sfh_conv_tables/r_sgrb_log.npy',r_sgrb_log)

print('Done')
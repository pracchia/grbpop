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


print(f'\nStar formation rate parameters: a = {a}, b = {b}, zp = {zp} \n')

# Setting the grid for the computation
at = np.linspace(0,5,20)
td_spacing = 0.01 # in Gyr
tdmin_max = 3.02 # Max minimum time delay, in Gyr
t_max = cosmo.lookback_time(100).to('Gyr').value
z_max = 11. # Max redshift for the distribution


print('Computing grid of time delays and formation redshifts...')

td_grid = np.arange(0, t_max, td_spacing)
neo_grid = np.zeros(len(td_grid)+1)
neo_grid[1:] = td_grid
neo_grid[1] = 0.005
td_grid = neo_grid
tdmin = td_grid[td_grid<=tdmin_max]

zf = np.zeros_like(td_grid)
zf[1:] = z_at_value(cosmo.lookback_time, td_grid[1:]*u.Gyr) # The grid of redshifts is built in a way that the redshifts are separated by a lookback time of td_spacing
z_grid = zf[zf<=z_max]

tdmin_index = range(1,len(tdmin),3)

r_sgrb_pow = np.zeros([len(z_grid),len(tdmin_index),len(at)])

H = cosmo.H(zf).to('1/Gyr').value # Grid of values for H(z)

print('Computing convolutions with power-law time delay distribution...')

for q, i in enumerate(tqdm(tdmin_index)):
    for j, a_t in enumerate(at):
        dtd_norm = np.trapezoid((td_grid[i:])**(-a_t)/H[i:]/(1+zf[i:]), zf[i:]) # Normalization factor for the time delay distribution, which depends on tdmin and a_t (we use the tdmin index since the redshift grid the time interval of tdmin)
        for k, z in enumerate(z_grid):
            z_int = zf[k:] # Redshift grid for the integral
            sfh = MD14_SFH(z_int,a,b,zp) # Star formation history
            tlb_f = td_grid[k+i:] # Formation lookback times, considering a minimum time delay given by the index "i"
            tlb_z = td_grid[k] # Lookback time at z
            dtd = np.zeros_like(z_int) # Create the grid for the time delay distribution, being 0 before tdmin...
            dtd[i:] = (tlb_f-tlb_z)**(-a_t) # ...and td^-a_t after (we use the tdmin index since the redshift grid the time interval of tdmin)
            r_sgrb_pow[k][q][j] = np.trapezoid(sfh*dtd/dtd_norm/(1+z_int)/H[k:],z_int)

tdmin_grid = tdmin[tdmin_index]
            
# # Setting the grid for the computation
# at = np.linspace(0,5,20)
# td_spacing = 0.01 # in Gyr
# tdmin_max = 3.02 # Max minimum time delay, in Gyr
# t_max = cosmo.lookback_time(100).to('Gyr').value
# z_max = 11. # Max redshift for the distribution


# print('Computing grid of time delays and formation redshifts...')

# td_grid = np.arange(0, t_max, td_spacing)
# tdmin = td_grid[td_grid<=tdmin_max]
# zf = np.zeros_like(td_grid)
# zf[1:] = z_at_value(cosmo.lookback_time, td_grid[1:]*u.Gyr) # The grid of redshifts is built in a way that the redshifts are separated by a lookback time of td_spacing
# z_grid = zf[zf<=z_max]

# modulus_tdgrid = 3.
# len_td_grid = int((tdmin_max-td_spacing)/td_spacing/modulus_tdgrid) + 1 # The step choosen for the tdmin grid is modulus_tdgrid because (3.00 - 0.01)/0.01 = 299 is divisible by modulus_tdgrid, otherwise it would take too much time... (maybe find a way to not hardcode this...)

# r_sgrb_pow = np.zeros([len(z_grid),len_td_grid,len(at)])

# H = cosmo.H(zf).to('1/Gyr').value # Grid of values for H(z)

# print('Computing convolutions with power-law time delay distribution...')

# tdmin_grid = np.zeros(len_td_grid)
# for i in tqdm(range(1,len(tdmin),int(modulus_tdgrid))):
#     q = int((i-1)/modulus_tdgrid) # tdmin index for the r_sgrb grid
#     tdmin_grid[q] = tdmin[i] # Grid of tdmin values for the interpolation
#     for j, a_t in enumerate(at):
#         dtd_norm = np.trapz((td_grid[i:])**(-a_t)/H[i:]/(1+zf[i:]), zf[i:]) # Normalization factor for the time delay distribution, which depends on tdmin and a_t (we use the tdmin index since the redshift grid the time interval of tdmin)
#         for k, z in enumerate(z_grid):
#             z_int = zf[k:] # Redshift grid for the integral
#             sfh = MD14_SFH(z_int,a,b,zp) # Star formation history
#             tlb_f = td_grid[k+i:] # Formation lookback times, considering a minimum time delay given by the index "i"
#             tlb_z = td_grid[k] # Lookback time at z
#             dtd = np.zeros_like(z_int) # Create the grid for the time delay distribution, being 0 before tdmin...
#             dtd[i:] = (tlb_f-tlb_z)**(-a_t) # ...and td^-a_t after (we use the tdmin index since the redshift grid the time interval of tdmin)
#             r_sgrb_pow[k][q][j] = np.trapz(sfh*dtd/dtd_norm/(1+z_int)/H[k:],z_int)



print('Computing convolutions with log-normal time delay distrbution...')

mu_td = np.linspace(0.001,5,30) # TEST 
# mu_td = np.linspace(0.01,5,30)
sigma_td = np.linspace(0.01,5,30)
r_sgrb_log = np.zeros([len(z_grid),len(mu_td),len(sigma_td)])

for i, mu in enumerate(tqdm(mu_td)):
    for j, sigma in enumerate(sigma_td):
        for k, z in enumerate(z_grid):
            z_int = zf[k:] # Redshift grid for the integral
            sfh = MD14_SFH(z_int,a,b,zp) # Star formation history
            tlb_f = td_grid[k:] # Formation lookback times, considering a minimum time delay given by the index "i"
            tlb_z = td_grid[k] # Lookback time at z
            dtd = np.nan_to_num(lognormal(tlb_f-tlb_z, mu, sigma)) # td = 0 ---> dtd = 0
            r_sgrb_log[k][i][j] = np.trapezoid(sfh*dtd/(1+z_int)/H[k:],z_int)


print('Saving grids...')

np.save('dtd_sfh_conv_tables/z.npy',z_grid)

np.save('dtd_sfh_conv_tables/at.npy',at)
np.save('dtd_sfh_conv_tables/tdmin.npy',tdmin_grid)
np.save('dtd_sfh_conv_tables/r_sgrb_pow.npy',r_sgrb_pow)

np.save('dtd_sfh_conv_tables/mu_td.npy',mu_td)
np.save('dtd_sfh_conv_tables/sigma_td.npy',sigma_td)
np.save('dtd_sfh_conv_tables/r_sgrb_log.npy',r_sgrb_log)

print('Done')
















### For the sake of clariy, the original start algorithm

## Single values of td and a_t
# td = 1
# a_t = at[3]
# H = cosmo.H(zf[td:]).to('1/Gyr').value
# dtd_norm = np.trapz((cosmo.lookback_time(zf[td:]).to('Gyr').value)**(-a_t)/(1+zf[td:])/H, zf[td:])
# for i, z in enumerate(z_grid):
#     z_int = zf[zf>=z]
#     sfh = MD14_SFH(z_int,a,b,zp) # Star formation history
#     dtd = np.zeros_like(z_int) # Create the grid for the time delay distribution, being 0 before tdmin...
#     if len(z_int) > td: # If with minimum time delay the lookback time goes over z = 100 we can neglect the star formation rate...
#         dtd[td:] = (cosmo.lookback_time(z_int[td:]).to('Gyr').value-cosmo.lookback_time(z).to('Gyr').value)**(-a_t) # ...and td^-a_t after (we use the tdmin index since the redshift grid the time interval of tdmin)
#     H = cosmo.H(z_int).to('1/Gyr').value # Factor to convert dP/dt in dP/dz
#     r_sgrb_pow[i] = np.trapz(sfh*dtd/dtd_norm/(1+z_int)/H, z_int)

## Original grid computing (time inefficient)
# r_sgrb = np.zeros([len(z_grid),int((tdmin_max-td_spacing)/td_spacing/13.),len(at)])
# ### The step is 13 because (3.00 - 0.01)/0.01 = 299 is divisible by 13, otherwise it would take too much time... (maybe find a way to not hardcode this...)
# for i in tqdm(range(1,len(tdmin),13)):
#     q = int((i-1)/13.) # tdmin index for the r_sgrb grid
#     for j, a_t in enumerate(at):
#         dtd_norm = np.trapz((cosmo.lookback_time(zf[i:]).to('Gyr').value)**(-a_t),zf[i:]) # Normalization factor for the time delay distribution, which depends on tdmin and a_t (we use the tdmin index since the redshift grid the time interval of tdmin)
#         for k, z in enumerate(z_grid):
#             z_int = zf[zf>=z] # Redshift grid for the integral
#             sfh = MD14_SFH(z_int,a,b,zp) # Star formation history
#             dtd = np.zeros_like(z_int) # Create the grid for the time delay distribution, being 0 before tdmin...
#             if len(z_int) > td: # If with minimum time delay the lookback time goes over z = 100 we can neglect the star formation rate...
#                 dtd[i:] = (cosmo.lookback_time(z_int[i:]).to('Gyr').value-cosmo.lookback_time(z_int[0]).to('Gyr').value)**(-a_t) # ...and td^-a_t after (we use the tdmin index since the redshift grid the time interval of tdmin)
#             H = cosmo.H(z_int).to('1/Gyr').value # Factor to convert dP/dt in dP/dz
#             r_sgrb[k][q][j] = np.trapz(sfh*dtd/dtd_norm/(1+z_int)/H,z_int)

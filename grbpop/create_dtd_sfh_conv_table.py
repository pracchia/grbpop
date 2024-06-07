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

a=2.6
b=3.6
zp=2.2
at = np.linspace(0,5,50)
td_spacing = 0.005
print('Computing grid of time delays and formation redshifts...')
td_grid = np.arange(0,cosmo.lookback_time(1000).value,td_spacing)
zf = np.zeros_like(td_grid)
zf[1:] = z_at_value(cosmo.lookback_time, td_grid[1:]*u.Gyr)
tdmin = td_grid[td_grid<=1]
tdmin = tdmin[1:]
z = zf[zf<=10]
r_sgrb_pow = np.zeros([len(z),len(tdmin),len(at)])
print('Computing convolutions...')
for k in tqdm(range(len(tdmin))):
    td_index = int(tdmin[k]/td_spacing)
    for q in range(len(at)):
        dtd_norm = np.trapz(td_grid[td_index:]**(-at[q]), td_grid[td_index:])
        for i in range(len(z)):
            zf_index = i + td_index + 1
            td_int = td_grid[td_index:-(i+1)]
            zf_int = zf[zf_index:]
            dtd = td_int**(-at[q])/dtd_norm
            sfh = MD14_SFH(zf_int,a=a,b=b,zp=zp)
            r_sgrb_pow[i][k][q] = np.trapz(sfh*dtd/(1+zf_int), td_int)


print('Saving grids...')

np.save('dtd_sfh_conv_tables/z.npy',z)

np.save('dtd_sfh_conv_tables/at.npy',at)
np.save('dtd_sfh_conv_tables/tdmin.npy',tdmin)
np.save('dtd_sfh_conv_tables/r_sgrb_pow.npy',r_sgrb_pow)

print('Done')
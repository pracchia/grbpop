import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from scipy.interpolate import RegularGridInterpolator

def MD14_SFH(z,a,b,zp):
    """
    Smoothly broken power law event rate density, with functional form given
    in Eq. 9 in Salafia+2023. This is the same functional form as the Madau &
    Dickinson 2014 cosmic star formation history fitting formula.
    """
    return (1.+z)**a/(1.+((1.+z)/(1.+zp))**(b+a))

def Ptd_pow(t, tdmin, alpha):
    ptd = np.zeros_like(t)
    ptd[t>tdmin] = t[t>tdmin]**(-alpha)
    if np.trapz(ptd,t) == 0:
        return 0
    else:
        return ptd/np.trapz(ptd,t)

z = np.logspace(-4,1,1000)
# For MD14_SFH (from Madau & Fragos 2017)
a=2.6
b=3.6
zp=2.2
# For power law
at = np.linspace(0,5,50)
tdmin = np.logspace(0,3,50) # in Myr
# For lognormal
td = np.logspace(1,4,50) # in Myr
sigma = np.logspace(1,4,50) # in Myr

r_sgrb_pow = np.zeros([len(z),len(tdmin),len(at)])
for i in range(len(z[0:-1])):
    # Compute quantities used for every minimum time delay and time delay distribution slope
    print('Computing redshift distributions for z = {0:.4f} ...           '.format(z[i]),end='\r')
    zf = z[z>z[i]] # Formation redshifts
    sfh = MD14_SFH(zf,a=a,b=b,zp=zp) # Star formation history for formation redshifts
    tlbz = cosmo.lookback_time(z[i]).to('Myr').value # Lookback time at redshift zeta in Myr
    tlbf = cosmo.lookback_time(zf).to('Myr').value # Formation lookback times in Myr
    dt_dr = 1/(1+zf)/cosmo.efunc(zf) # Derivative of lookback time with respect to redshift
    for j in range(len(tdmin)):
        for k in range(len(at)):
            r_sgrb_pow[i][j][k] = np.trapz(sfh*Ptd_pow(tlbf-tlbz, tdmin=tdmin[j], alpha=at[k])*dt_dr/(1+zf), zf)


np.save('dtd_sfh_conv_tables/z.npy',z)

np.save('dtd_sfh_conv_tables/at.npy',at)
np.save('dtd_sfh_conv_tables/tdmin.npy',tdmin)
np.save('dtd_sfh_conv_tables/r_sgrb_pow.npy',r_sgrb_pow)

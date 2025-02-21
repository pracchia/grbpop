import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
import pandas
import grbpop
from grbpop.pdet import pdet_GBM
from grbpop.globals import *
import pandas, os

#######
# # load SGRB data from GBM catalog to construct observer frame sample
# gbm = pandas.read_csv('grb_data/GBM_pflx_allinfo.csv')
# sgrb = gbm.loc[gbm['t90']<2]
# p50300 = sgrb['pflx_comp_phtfluxb'].values
# ep = sgrb['pflx_comp_epeak'].values
#######

# P64 flux cuts
p_gbm_lim = 2.37
p_swift_lim = 2.5
p_batse_lim = 1.5
ep_bias = 800.

# load SGRB data from GBM catalog to construct observer frame sample

# Fermi/GBM sample
gbm = pandas.read_csv('grb_data/FermiGBM_WP15.txt', sep='|')
gbm.columns = gbm.columns.str.strip()
gbm['t90'] = pandas.to_numeric(gbm['t90'], errors='coerce')
gbm['flux_batse_64'] = pandas.to_numeric(gbm['flux_batse_64'], errors='coerce')
gbm = gbm.drop(columns=[""], errors="ignore")
gbm = gbm.dropna(axis=1, how="all")
gbm = gbm.fillna(0)
sgrb_gbm = gbm.loc[gbm['t90']<2]
s_gbm = sgrb_gbm.loc[sgrb_gbm['flux_batse_64']>=p_gbm_lim]

p50300_gbm = s_gbm['flux_batse_64'].values
ep_gbm = np.zeros_like(p50300_gbm) + 490. # Ep_obs = 490 keV

print(f'Fermi/GBM sample: {len(p50300_gbm)} GRBs')

# CGRO/BATSE sample
batse = pandas.read_csv('grb_data/BATSE_WP15.txt', sep='|')
batse.columns = batse.columns.str.strip()
batse['t90'] = pandas.to_numeric(batse['t90'], errors='coerce')
batse['flux_64'] = pandas.to_numeric(batse['flux_64'], errors='coerce')
batse = batse.drop(columns=[""], errors="ignore")
batse = batse.dropna(axis=1, how="all")
batse = batse.fillna(0)
sgrb_batse = batse.loc[batse['t90']<2]
s_batse = sgrb_batse.loc[sgrb_batse['flux_64']>=p_batse_lim]

p50300_batse = s_batse['flux_64'].values
ep_batse = np.zeros_like(p50300_batse) + 490. # Ep_obs = 490 keV

print(f'CGRO/BATSE sample: {len(p50300_batse)} GRBs')

#######
# Lsamples = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_L_samples.npy')
# Epsamples = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_Ep_samples.npy')
# zobs = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_zobs.npy')
# Nsamples = Lsamples.shape[1]
#######

# load L, Ep and z posterior samples for restframe sample
bat = pandas.read_csv('grb_data/SwiftBAT_WP15.txt', sep='|')
bat.columns = bat.columns.str.strip()
bat = bat.drop(columns=[""], errors="ignore")
bat = bat.dropna(axis=1, how="all")
bat = bat.fillna(0)

L51_bat = bat['L51'].values
Lsamples = 1e51*L51_bat
zobs = bat['z'].values

# set alpha index for Band function
alpha = -0.5

def ptform(u):
    """Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest."""
    
    x = np.array(u)  # copy u

    # 'alpha_L':x[0], theta_pop['alpha_L']<0., theta_pop['alpha_L']>5.
    x[0] = u[0]*5.-1. # scale and shift to [0, 5]

    # 'beta_L':x[1], theta_pop['beta_L']<=0., theta_pop['beta_L']>6.
    x[1] = u[1]*6. # scale to [0, 6]
    
    # 'L_*':10.**x[2], theta_pop['L_*']<1e51, theta_pop['L_*']>1e53
    x[2] = u[2]*4. + 50. # scale and shift to [log10(1e51), log10(1e53)]

    # 'mu_td':x[3], theta_pop['mu_td']<0.01, theta_pop['mu_td']>5.
    x[3] = u[3]*(5.-0.01) + 0.01 # scale and shift to [0.01, 5]

    # 'sigma_td':x[4], theta_pop['sigma_td']<0.01, theta_pop['sigma_td']>5.
    x[4] = u[4]*(5.-0.01) + 0.01 # scale and shift to [0.01, 5]
    
    return x
    

def loglike(x):
    """
    log likelihood
    """

    # broken power law luminosity function
    theta_pop = {'rho_z':'DTD*SFH',
             'dtd':'lognorm',
             'alpha_L':x[0],
             'beta_L':x[1],
             'gamma_L':x[0],
             'L_*':10.**x[2],
             'L_**':6e49,
             'L_0':5e49,
             'tdmin':0.02,
             'mu_td':x[3],
             'sigma_td':x[4]
             }
    
    pi_Lz = lambda Lx,zx:Lx**-1 # L,z prior from spectral analysis
    pdet_gbm = lambda pf,ep: (pf>p_gbm_lim) # detection probability for flux-limited sample analysis
    pdet_batse = lambda pf,ep: (pf>p_batse_lim) # detection probability for flux-limited sample analysis
    
    # evaluate log likelihood
        
    ## Fermi/GBM sample
    logl_obsframe_fermi = grbpop.Ppop.obsframe_loglikelihood_lum2breaks_NO_EP(pf=p50300_gbm,epbias=ep_bias,alpha=alpha,specmodel='Band',inst='Fermi',theta_pop=theta_pop,res=100,pdet=pdet_gbm,pflim=p_gbm_lim,return_logalpha=False)
    
    ## CGRO/BATSE sample
    logl_obsframe_batse = grbpop.Ppop.obsframe_loglikelihood_lum2breaks_NO_EP(pf=p50300_batse,epbias=ep_bias,alpha=alpha,specmodel='Band',inst='Fermi',theta_pop=theta_pop,res=100,pdet=pdet_batse,pflim=p_batse_lim,return_logalpha=False)

    ## Swift/BAT sample
    logl_restframe = grbpop.Ppop.restframe_loglikelihood_lum2breaks_NO_EP(Lobs=Lsamples,zobs=zobs,epbias=ep_bias,alpha=alpha,inst='Swift',theta_pop=theta_pop,specmodel='Band',pdet=None,pflim=p_swift_lim,prior_Lz=pi_Lz,logalpha=None,res=100)

    # print(logl_obsframe_fermi, logl_obsframe_batse, logl_restframe)
    ## sum all contributions
    logl = logl_obsframe_fermi + logl_obsframe_batse + logl_restframe
    
    if np.isfinite(logl):
        return logl
    else:
        return -np.inf


if __name__=='__main__':
    from dynesty import DynamicNestedSampler
    from dynesty import pool as dypool
    
    nthreads = 8
    ndim = 5
    nlive = 100*ndim
    dlogz = 0.01
    N_effective_sample = 20000
    checkpoint_filename = 'nested_samplings/WP15_Dynesty_SGRB_fit_LOG.save'
    
    print('Starting dynamic nested sampling...')
    # initialize the sampler
    with dypool.Pool(nthreads, loglike=loglike, prior_transform=ptform) as pool:
        if os.path.exists(checkpoint_filename):
            dsampler = DynamicNestedSampler.restore(checkpoint_filename, pool=pool, sample='rslice')
            dsampler.run_nested(resume=True, n_effective=N_effective_sample, checkpoint_file=checkpoint_filename, dlogz_init=dlogz, nlive_init=nlive, nlive_batch=100)
        else:
            dsampler = DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool, sample='rslice')
            dsampler.run_nested(use_stop=True, n_effective=N_effective_sample, checkpoint_file=checkpoint_filename, dlogz_init=dlogz, nlive_init=nlive, nlive_batch=100)

    print('')
    
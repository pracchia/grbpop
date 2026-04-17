import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
import pandas
import grbpop
from grbpop.pdet import pdet_GBM
from grbpop.globals import *
import pandas, os


path = 'grb_data/mock_dataset/mock_sample.csv'
data = pandas.read_csv(path)

rest = data.loc[0:29] # Rest frame mocksample
obs = data.loc[30:] # Observer frame mocksample

p_gbm_lim = 1.5


# Obs frame
p50300 = obs['pf'].values
ep = obs['ep'].values

cut = p50300>p_gbm_lim

p50300 = p50300[cut]
ep = ep[cut]


# Rest frame
pf = rest['pf'].values
Lsamples = rest['L'].values
Epsamples = rest['Ep'].values
zobs = rest['z'].values

cut = pf>p_gbm_lim

Lsamples = Lsamples[cut]
Epsamples = Epsamples[cut]
zobs = zobs[cut]

Nsamples = len(Lsamples)


# set low-energy photon index to the median of the GBM sample
alpha=-0.4

# Duty cycles for the detectors
eta_GBM = 0.59

# Time intervals for Poisson distributions (in years)
T_GBM = 10.


def ptform(u):
    """Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest."""
    
    x = np.array(u)  # copy u

    # 'alpha_L':x[0], theta_pop['alpha_L']<-1., theta_pop['alpha_L']>5.
    x[0] = u[0]*6. - 1. # scale and shift to [-1, 5]

    # 'beta_L':x[1], theta_pop['beta_L']<=0., theta_pop['beta_L']>5.
    x[1] = u[1]*5. # scale to [0, 5]
    
    # 'gamma_L':x[2], theta_pop['gamma_L']<-5., theta_pop['gamma_L']>5.
    x[2] = u[2]*10. - 5. # scale to [-5, 5]
    
    # 'L_0':10.**x[5], theta_pop['L_0']<10**logLmin, theta_pop['L_0']>1e47
    x[5] = u[5]*(47.-logLmin) + logLmin # scale and shift to [logLmin, log10(1e47)]
   
    # 'L_**':10.**x[4], theta_pop['L_**']<L_0, theta_pop['L_**']>1e51
    x[4] = u[4]*(51.-x[5]) + x[5] # scale and shift to [log10(L_0), log10(1e51)]
    
    # 'L_*':10.**x[3], theta_pop['L_*']<1e51, theta_pop['L_*']>10**logLmax
    x[3] = u[3]*(logLmax-51.) + 51. # scale and shift to [log10(1e50), logLmax]

    # 'E_p*':10.**x[6], theta_pop['E_p*']<1e2, theta_pop['E_p*']>1e5
    x[6] = u[6]*3. + 2. # scale and shift to [log10(1e2), log10(1e5)]

    # 's_c':10.**x[7], theta_pop['s_c']<0.3, theta_pop['s_c']>3.
    x[7] = u[7]*np.log10(3/0.3) + np.log10(0.3) # scale and shift to [log10(0.3), log10(3)]

    # 'y':x[8], theta_pop['y']<-3., theta_pop['y']>3. 
    x[8] = u[8]*6. - 3. # scale and shift to [-3, 3]

    # LOG PRIOR VERSION
    # # 'tdmin':x[9], theta_pop['tdmin']<0.01, theta_pop['tdmin']>3.
    x[9] = u[9]*np.log10(3./0.005) + np.log10(0.005)
    # x[9] = u[9]*(3.-0.01) + 0.01 # scale and shift to [0.01, 3]

    # 'at':x[10], theta_pop['at']<0., theta_pop['at']>5.
    x[10] = u[10]*5. # scale [0, 5]
    
    # 'R0':10**x[11], theta_pop['R0']<1., theta_pop['R0']>1e5:
    # x[11] = u[11]*5. # scale and shift to [log10(1.), log10(1e5)]

    return x
    

def loglike(x):
    """
    log likelihood
    """

    # broken power law luminosity function
    theta_pop = {'rho_z':'DTD*SFH',
             'dtd':'pow',
             'alpha_L':x[0],
             'beta_L':x[1],
             'gamma_L':x[2],
             'L_*':10.**x[3],
             'L_**':10.**x[4],
             'L_0':10.**x[5],
             'E_p*':10.**x[6],
             's_c':10.**x[7],
             'y':x[8],
             'tdmin':10**x[9],
             'at':x[10],
             }
             # 'R0':10.**x[11]
    
    pi_EpLz = lambda Epx,Lx,zx:Lx**-1*(1.+zx)**-1 # Ep,L,z prior from spectral analysis
    pdet = lambda pf,ep: pf>p_gbm_lim # detection probability for flux-limited sample analysis
    
    # evaluate log likelihood
        
    ## observer frame sample
    logl_obsframe = grbpop.Ppop.obsframe_loglikelihood_lum_2breaks(p50300,ep,alpha=alpha,specmodel='Comp',inst='Fermi',theta_pop=theta_pop,res=100,pdet=pdet,pflim=p_gbm_lim,return_logalpha=None)

    ## restframe sample
    logl_restframe = grbpop.Ppop.restframe_loglikelihood_lum2breaks(Lsamples,Epsamples,zobs,alpha=alpha,inst='Fermi',theta_pop=theta_pop,specmodel='Comp',pdet=None,pflim=[p_gbm_lim],prior_EpLz=pi_EpLz,logalpha=None,res=100)

    ## Poissonian terms
    # log_poisson_obsframe = grbpop.Ppop.log_poissonian_observer_lum_2breaks(theta_pop,N_obs=len(p50300),eta=eta_GBM,T=T_GBM,logalpha=log_alpha_obsframe,alpha=alpha,specmodel='Comp',inst='Fermi',res=100,pdet=pdet,pflim=None)
    
    ## sum all contributions
    # logl = logl_obsframe + logl_restframe + log_poisson_obsframe
    logl = logl_obsframe + logl_restframe
    
    if np.isfinite(logl):
        return logl
    else:
        return -np.inf


if __name__=='__main__':
    from dynesty import DynamicNestedSampler
    from dynesty import pool as dypool
    
    nthreads = 8
    # ndim = 12
    ndim = 11
    nlive = 100*ndim
    nbatch = 20*ndim
    dlogz = 0.001
    N_effective_sample = 20000
    checkpoint_filename = f'nested_samplings/MOCKDATA_BIASES_{p_gbm_lim}_BPL_POW_dlogz{dlogz}_nlive{nlive}_nbatch{nbatch}_neff{N_effective_sample}.save'

    
    print('Starting dynamic nested sampling...')
    # initialize the sampler
    with dypool.Pool(nthreads, loglike=loglike, prior_transform=ptform) as pool:
        if os.path.exists(checkpoint_filename):
            dsampler = DynamicNestedSampler.restore(checkpoint_filename, pool=pool)
            dsampler.run_nested(resume=True, n_effective=N_effective_sample, checkpoint_file=checkpoint_filename, dlogz_init=dlogz, nlive_init=nlive, nlive_batch=nbatch)
        else:
            dsampler = DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool)
            dsampler.run_nested(use_stop=True, n_effective=N_effective_sample, checkpoint_file=checkpoint_filename, dlogz_init=dlogz, nlive_init=nlive, nlive_batch=nbatch)

    print('')
    
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import ks_2samp
import pandas
import grbpop
from grbpop.pdet import pdet_GBM
import emcee
from multiprocessing import Pool


# load SGRB data from GBM catalog to construct observer frame sample
gbm = pandas.read_csv('grb_data/GBM_pflx_allinfo.csv')
sgrb = gbm.loc[gbm['t90']<2]
p50300 = sgrb['pflx_comp_phtfluxb'].values
ep = sgrb['pflx_comp_epeak'].values

# impose a low flux cut to see the the effect of wrong treatment of selection effects
p_gbm_lim = 2.37
p_swift_lim = 2.5
clean = (p50300>p_gbm_lim) 

ep = ep[clean]
p50300 = p50300[clean]

# load L, Ep and z posterior samples for restframe sample (the last one is GW170817)
Lsamples = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_L_samples.npy')
Epsamples = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_Ep_samples.npy')
zobs = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_zobs.npy')
Nsamples = Lsamples.shape[1]


# construct GW170817 inclination posterior
gwsamples = np.genfromtxt('grb_data/high_spin_PhenomPNRT_posterior_samples.dat',names=True)

# costheta_jn to theta_view
thv17 = np.arccos(gwsamples['costheta_jn'])
thv17[thv17>(np.pi/2.)]=np.pi-thv17[thv17>(np.pi/2.)]

# resample thv to condition on NG4993's distance constraint from Cantiello et al. 2018
dL17s = gwsamples['luminosity_distance_Mpc']
dL17 = 40.7
dL17_err = (1.4**2+1.9**2)**0.5

w = np.exp(-0.5*((dL17-dL17s)/dL17_err)**2)

rng = np.random.default_rng()
thvs = rng.choice(thv17,Nsamples,p=w/np.sum(w))

# set low-energy photon index to the median of the GBM sample
alpha=-0.4

# Duty cycles for the detectors
eta_GBM = 0.59

# Time intervals for Poisson distributions (in years)
T_GBM = 10.
T_O3 = 11./12.


def logprior(theta_pop):
    """
    log prior 
    """
    if theta_pop['thc']<0.01 or theta_pop['thc']>(np.pi/2.)\
    or theta_pop['Lc*']<3e51 or theta_pop['Lc*']>1e55\
    or theta_pop['thw']<theta_pop['thc'] or theta_pop['thw']>np.pi/2.\
    or theta_pop['a_L']<0. or theta_pop['a_L']>6.\
    or theta_pop['b_L']<-3. or theta_pop['b_L']>6.\
    or theta_pop['a_Ep']<0. or theta_pop['a_Ep']>6.\
    or theta_pop['b_Ep']<-3. or theta_pop['b_Ep']>6.\
    or theta_pop['A']<1.5 or theta_pop['A']>5.\
    or theta_pop['a']<-1. or theta_pop['a']>5.\
    or theta_pop['b']<1. or theta_pop['b']>10.\
    or theta_pop['zp']<0.1 or theta_pop['zp']>3.\
    or theta_pop['R0']<1. or theta_pop['R0']>1e6:
        return -np.inf
    else:
        return np.log(theta_pop['thc']) + np.log(np.sin(theta_pop['thc'])) + np.log(theta_pop['thw']) + np.log(np.sin(theta_pop['thw'])) # "isotropic" prior on angles


def loglike(x):
    """
    log likelihood
    """
    
    # smooth double power law jet model
    theta_pop = {'jetmodel':'smooth double power law',
             'rho_z':'SBPL',
             'thc':10**x[0],
             'Lc*':10.**x[1],
             'a_L':x[2],
             'b_L':x[3],
             'Epc*':800.,
             'a_Ep':x[4],
             'b_Ep':x[5],
             'thw':10.**x[6],
             'A':x[7],
             's_c':0.3,
             'y':0.,
             'a':x[8],
             'b':x[9],
             'zp':x[10],
             'R0':10**x[11]
             }
    
    pi_EpLz = lambda Epx,Lx,zx:Lx**-1*(1.+zx)**-1 # Ep,L,z prior from spectral analysis
    pdet = lambda pf,ep: (pf>p_gbm_lim)*(ep<1e4)*(ep>50.) # detection probability for flux-limited sample analysis
    
    # evaluate log prior
    lpr = logprior(theta_pop)
    
    if not np.isfinite(lpr):
        return -np.inf
    
    # evaluate log likelihood

    ## observer frame sample
    logl_obsframe, log_alpha_obsframe = grbpop.Ppop.obsframe_loglikelihood(p50300,ep,alpha=alpha,specmodel='Comp',inst='Fermi',theta_pop=theta_pop,res=80,pdet=pdet,pflim=p_gbm_lim,return_logalpha=True)
    
    ## restframe sample
    logl_restframe = grbpop.Ppop.restframe_loglikelihood(Lsamples[:-1],Epsamples[:-1],zobs[:-1],alpha=alpha,inst='Fermi+Swift',theta_pop=theta_pop,specmodel='Comp',pdet=None,pflim=[p_gbm_lim,p_swift_lim],prior_EpLz=pi_EpLz,logalpha=None,res=60)

    ## Poissonian terms
    log_poisson_obsframe = grbpop.Ppop.log_poissonian_observer(theta_pop,N_obs=len(p50300),eta=eta_GBM,T=T_GBM,logalpha=log_alpha_obsframe,alpha=alpha,specmodel='Comp',inst='Fermi',res=80,pdet=pdet,pflim=None)
    
    ## sum all contributions
    logl = logl_obsframe + logl_restframe + log_poisson_obsframe
    
    if np.isfinite(logl + lpr):
        return logl + logprior(theta_pop)
    else:
        return -np.inf


if __name__=='__main__':
    nthreads = 8
    N_iter = 10000
    chain_filename = 'chains/SGRB_flux-limited-sample-analysis_Poisson_WRONGCUT_fixed_boundaries.h5' # full
    
    # initial guess vector

    #      log(thj)  log(Lj) a_L      b_L     a_Ep    b_Ep  log(thw)     A        a      b    zp   log(R0)  
    x0 = [-1.877,     51.55, 4.091, -2.318,    1.2,   2.069, -0.5058, 3.041,  3.431, 7.623,  0.9,   2.509]  # starting guess
    
    # as a cross check
    print('Log likelihood at starting guess: ',loglike(x0))
    
    # number of dimensions
    ndim = len(x0)
    
    # set number of walkers as 4 times the number of dimensions
    nwalkers = ndim*4
        
    # the initial positions of the walkers are distributed within a tiny hypersphere around the starting guess
    p0s = (np.array(x0).reshape([len(x0),1])*(1.+np.random.normal(loc=0.,scale=1e-3,size=[ndim,nwalkers]))).T
    
    # set up the backend
    backend = emcee.backends.HDFBackend(chain_filename)
    
    print('Starting MCMC...')
    # initialize the sampler
    with Pool(nthreads) as pool:
        sampler = emcee.EnsembleSampler(nwalkers,ndim,loglike,backend=backend,pool=pool)
        if sampler.iteration>0:
            sampler.run_mcmc(None,N_iter-sampler.iteration,progress=True,store=True)
        else:
            sampler.run_mcmc(p0s,N_iter,progress=True,store=True)
        
    print('')

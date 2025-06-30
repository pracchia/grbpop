import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from scipy import integrate, interpolate
import pandas, os
import grbpop
from grbpop.pdet import pdet_GBM


# load SGRB data from GBM catalog to construct observer frame sample
gbm = pandas.read_csv('grb_data/GBM_pflx_allinfo.csv')
sgrb = gbm.loc[gbm['t90']<2]
p50300 = sgrb['pflx_comp_phtfluxb'].values
ep = sgrb['pflx_comp_epeak'].values

# impose quality cuts
clean = (ep>50.) & (ep<1e4) & (p50300>1.) 

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


class log_iso_angle_prior:
    """ 
    Class defining an isotropic prior for log(angle) 
    """
    
    def __init__(self, low_bound, high_bound):
        """
        Initialize the bounds of the prior 
        """
        self.low = low_bound # Lower bound for the angle
        self.high = high_bound # Higher bound for the angle
        self.log_theta = np.linspace(np.log10(self.low), np.log10(self.high), 200) # Array of angle logarithms linearly spaced 
        if self.low<=0 or self.high>np.pi:
            print('WARNING! PRIOR NOT WELL DEFINED, MAY GIVE NEGATIVE VALUES OR NAN!')

    def pdf_nonorm(self, x):
        """ 
        Non normalized probability distribution function for a given array.
        Skipping the normalization step makes the CDF estimation an 'epsilon' quicker
        """
        Pdf = 10**x * np.sin(10**x)
        return Pdf
       
    def cdf(self, x):
        """ 
        Cumulative distribution function for given values 
        """
        Cdf = integrate.cumulative_trapezoid(self.pdf_nonorm(self.log_theta), self.log_theta, initial=0)
        Cdf /= Cdf[-1]
        interp_cdf = interpolate.interp1d(self.log_theta, Cdf, bounds_error=False, fill_value=0)
        return interp_cdf(x)

    def ppt(self, x):
        """ 
        Percent point function, a.k.a. the inverse cumulative distribution function, for given values 
        """
        interp_ppt = interpolate.interp1d(self.cdf(self.log_theta), self.log_theta, bounds_error=True)
        return interp_ppt(x)


thc_prior = log_iso_angle_prior(0.01,np.pi/2.)

def ptform(u):
    """Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest."""

    x = np.array(u)  # copy u

    # 'thc':10**x[0], if theta_pop['thc']<0.01, theta_pop['thc']>(np.pi/2.)
    x[0] = thc_prior.ppt(u[0]) # logprior = np.log(theta_pop['thc']) + np.log(np.sin(theta_pop['thc']))

    # 'Lc*':10.**x[1], theta_pop['Lc*']<3e51, theta_pop['Lc*']>1e55 
    x[1] = u[1]*np.log10(1e55/3e51) + np.log10(3e51) # scale and shift to [log10(3e51), log10(1e55)]
    
    # 'a_L':x[2], theta_pop['a_L']<0., theta_pop['a_L']>6.
    x[2] = u[2]*6. # scale to [0, 6] 
    
    # 'b_L':x[3], theta_pop['b_L']<-3., theta_pop['b_L']>6.
    x[3] = u[3]*9. - 3 # scale and shift to [-3, 6]

    # 'Epc*':10.**x[4], theta_pop['Epc*']<1e2, theta_pop['Epc*']>1e5 
    x[4] = u[4]*np.log10(1e5/1e2) + np.log10(1e2) # scale and shift to [log10(1e2), log10(1e5)] 
    
    # 'a_Ep':x[5], theta_pop['a_Ep']<0., theta_pop['a_Ep']>6.
    x[5] = u[5]*6. # scale to [0, 6]
    
    # 'b_Ep':x[6], theta_pop['b_Ep']<-3., theta_pop['b_Ep']>6.
    x[6] = u[6]*9. - 3 # scale and shift to [-3, 6]

    # 'thw':10.**x[7], theta_pop['thw']<theta_pop['thc'], theta_pop['thw']>np.pi/2.
    thw_prior = log_iso_angle_prior(10**x[0],np.pi/2.)
    x[7] = thw_prior.ppt(u[7]) # np.log(theta_pop['thw']) + np.log(np.sin(theta_pop['thw']))

    # 'A':x[8], theta_pop['A']<1.5, theta_pop['A']>5. 
    x[8] = u[8]*3.5 + 1.5 # scale and shift to [1.5, 5]

    # 's_c':10.**x[9], theta_pop['s_c']<0.3, theta_pop['s_c']>3.
    x[9] = u[9]*np.log10(3/0.3) + np.log10(0.3) # scale and shift to [log10(0.3), log10(3)]

    # 'y':x[10], theta_pop['y']<-3., theta_pop['y']>3. 
    x[10] = u[10]*6. - 3 # scale and shift to [-3, 3]

    # 'a':x[11], theta_pop['a']<-1., theta_pop['a']>5. 
    x[11] = u[11]*6. - 1 # scale and shift to [-1, 5]

    # 'b':x[12], theta_pop['b']<1., theta_pop['b']>10. 
    x[12] = u[12]*9. + 1 # scale and shift to [1, 10]

    # 'zp':x[13], theta_pop['zp']<0.1, theta_pop['zp']>3.
    x[13] = u[13]*(2.9) + 0.1 # scale and shift to [0.1, 3]

    # 'R0':10**x[14], theta_pop['R0']<1., theta_pop['R0']>1e6:
    x[14] = u[14]*6. # scale and shift to [log10(1.), log10(1e6)]

    return x


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
             'Epc*':10.**x[4],
             'a_Ep':x[5],
             'b_Ep':x[6],
             'thw':10.**x[7],
             'A':x[8],
             's_c':10.**x[9],
             'y':x[10],
             'a':x[11],
             'b':x[12],
             'zp':x[13],
             'R0':10**x[14]
             }
    
    pi_EpLz = lambda Epx,Lx,zx:Lx**-1*(1.+zx)**-1 # Ep,L,z prior from spectral analysis
    pdet = lambda pf,ep: pdet_GBM(pf,ep)*(pf>1.)*(ep<1e4)*(ep>50.) # detection probability for full sample analysis
    
    # evaluate log likelihood
    
    ## observer frame sample
    logl_obsframe, log_alpha_obsframe = grbpop.Ppop.obsframe_loglikelihood(p50300,ep,alpha=alpha,specmodel='Comp',inst='Fermi',theta_pop=theta_pop,res=80,pdet=pdet,pflim=None,return_logalpha=True)
    
    ## restframe sample
    logl_restframe = grbpop.Ppop.restframe_loglikelihood(Lsamples[:-1],Epsamples[:-1],zobs[:-1],alpha=alpha,inst='Fermi+Swift',theta_pop=theta_pop,specmodel='Comp',pdet='gbm',pflim=[None,3.5],prior_EpLz=pi_EpLz,logalpha=None,res=60)
    
    ## viewing angle sample
    log_alpha_GRB_GW = grbpop.Ppop.logalpha_GRB_GW(theta_pop,pdet_GW='O3')
    logl_GW170817 = grbpop.Ppop.known_theta_view_loglikelihood(Lsamples[-1],Epsamples[-1],thvs,theta_pop,prior_EpLz=pi_EpLz) + np.log(grbpop.Ppop.Pz(zobs[-1][0],theta_pop=theta_pop))-log_alpha_GRB_GW

    ## Poissonian terms
    log_poisson_obsframe = grbpop.Ppop.log_poissonian_observer(theta_pop,N_obs=len(p50300),eta=eta_GBM,T=T_GBM,logalpha=log_alpha_obsframe,alpha=alpha,specmodel='Comp',inst='Fermi',res=80,pdet=pdet,pflim=None)
    log_poisson_GW170817 = grbpop.Ppop.log_poissonian_GRB_GW(theta_pop,N_obs=1,eta=eta_GBM,T=T_O3,logalpha=log_alpha_GRB_GW,pdet_GW='O3')
    
    ## sum all contributions
    logl = logl_obsframe + logl_restframe + logl_GW170817 + log_poisson_obsframe + log_poisson_GW170817
    
    if np.isfinite(logl):
        return logl
    else:
        return -np.inf


if __name__=='__main__':
    from dynesty import DynamicNestedSampler
    from dynesty import pool as dypool
    
    nthreads = 8
    ndim = 15
    nlive = 500
    # N_effective_sample = 20000
    dlogz = 0.01
    # sample = 'rslice'
    sample = 'slice'
    bound = 'multi'
    # bound = 'balls'
    checkpoint_filename = 'nested_samplings/Dynesty_SGRB_full-sample-analysis_Poisson_nlive500_dlogz01_post100_slice_multi.save'
    
    print('Starting dynamic nested sampling...')
    # initialize the sampler
    with dypool.Pool(nthreads, loglike=loglike, prior_transform=ptform) as pool:
        if os.path.exists(checkpoint_filename):
            dsampler = DynamicNestedSampler.restore(checkpoint_filename, pool=pool)
            # dsampler.run_nested(resume=True, n_effective=N_effective_sample, checkpoint_file=checkpoint_filename, dlogz_init=dlogz, nlive_init=nlive, nlive_batch=100, sample='hslice', wt_kwargs={'pfrac': 1.0})
            dsampler.run_nested(resume=True, dlogz_init=dlogz, n_effective=None, checkpoint_file=checkpoint_filename, nlive_init=nlive, wt_kwargs={'pfrac': 1.0})
        else:
            dsampler = DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool, sample=sample, bound=bound)
            # dsampler.run_nested(use_stop=True, n_effective=N_effective_sample, checkpoint_file=checkpoint_filename, dlogz_init=dlogz, nlive_init=nlive, nlive_batch=100, sample='hslice', wt_kwargs={'pfrac': 1.0})
            dsampler.run_nested(use_stop=True, dlogz_init=dlogz, n_effective=None, checkpoint_file=checkpoint_filename, nlive_init=nlive, wt_kwargs={'pfrac': 1.0})

    print('')
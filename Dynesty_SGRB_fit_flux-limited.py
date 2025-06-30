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

# impose quality cuts and flux completeness cut
clean = (ep>50.) & (ep<1e4) & (p50300>3.5) 

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
             'zp':x[13]
             }
    
    pi_EpLz = lambda Epx,Lx,zx:Lx**-1*(1.+zx)**-1 # Ep,L,z prior from spectral analysis
    pdet = lambda pf,ep: (pf>3.5)*(ep<1e4)*(ep>50.) # detection probability for flux-limited sample analysis
    
    # evaluate log likelihood
    
    ## observer frame sample
    logl_obsframe = grbpop.Ppop.obsframe_loglikelihood(p50300,ep,alpha=alpha,specmodel='Comp',inst='Fermi',theta_pop=theta_pop,res=80,pdet=pdet,pflim=3.5,return_logalpha=False)
    
    ## restframe sample
    logl_restframe = grbpop.Ppop.restframe_loglikelihood(Lsamples[:-1],Epsamples[:-1],zobs[:-1],alpha=alpha,inst='Fermi+Swift',theta_pop=theta_pop,specmodel='Comp',pdet=None,pflim=[3.5,3.5],prior_EpLz=pi_EpLz,logalpha=None,res=60)
    
    ## viewing angle sample (this is actually a prior in the flux-limited sample analysis, see sec 2.5.3 in Salafia+2023 
    logl_GW170817 = grbpop.Ppop.known_theta_view_loglikelihood(Lsamples[-1],Epsamples[-1],thvs,theta_pop,prior_EpLz=pi_EpLz)
    
    ## sum all contributions
    logl = logl_obsframe + logl_restframe + logl_GW170817
    
    if np.isfinite(logl):
        return logl
    else:
        return -np.inf


if __name__=='__main__':
    from dynesty import DynamicNestedSampler
    from dynesty import pool as dypool
    
    nthreads = 8
    ndim = 14
    N_effective_sample = 20000
    # samples_filename = 'nested_samplings/Dynesty_SGRB_flux-limited-sample-analysis.h5'
    checkpoint_filename = 'nested_samplings/Dynesty_SGRB_flux-limited-sample-analysis.save'
    # test_u = np.zeros(ndim) + 0.5
    # test_x = ptform(test_u)
    
    # as a cross check
    # print('Test log likelihood for u[i] = 0.5: ',loglike(test_x))
    
    # set number of walkers as 4 times the number of dimensions
    # nwalkers = ndim*4
    
    print('Starting dynamic nested sampling...')
    # initialize the sampler
    with dypool.Pool(nthreads, loglike=loglike, prior_transform=ptform) as pool:
        if os.path.exists(checkpoint_filename):
            dsampler = DynamicNestedSampler.restore(checkpoint_filename, pool=pool)
            dsampler.run_nested(resume=True, n_effective=N_effective_sample, checkpoint_file=checkpoint_filename, dlogz_init=0.01, nlive_init=500, nlive_batch=100)
            # dsampler.run_nested(resume=True, use_stop=False, n_effective=N_effective_sample, checkpoint_file=checkpoint_filename)
        else:
            dsampler = DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool)
            dsampler.run_nested(use_stop=True, n_effective=N_effective_sample, checkpoint_file=checkpoint_filename, dlogz_init=0.01, nlive_init=500, nlive_batch=100)
            # dsampler.run_nested(use_stop=False, n_effective=N_effective_sample, checkpoint_file=checkpoint_filename)

    # import h5py
    # # Save samples to an HDF5 file
    # with h5py.File(samples_filename, 'w') as f:
    #     f.create_dataset('samples', data=dsampler.results.samples)  # Raw samples
    #     f.create_dataset('weights', data=dsampler.results.importance_weights)  # Importance weights
    #     f.create_dataset('logl', data=dsampler.results.logl)  # Log-likelihoods
    #     f.create_dataset('logwt', data=dsampler.results.logwt)  # Logarithmic weights
    #     f.create_dataset('logz', data=dsampler.results.logz)  # Log-evidence estimates
    #     f.create_dataset('logzerr', data=dsampler.results.logzerr)  # Log-evidence uncertainty

    print('')

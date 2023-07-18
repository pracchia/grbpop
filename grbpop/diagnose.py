import numpy as np
import emcee
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumtrapz
from . import Ppop
from . import pflux
from . import structjet
from .pdet import pdet_GBM
from .globals import *

def greedy2d(x,y,Pxy,smooth=0.5):
    dx = np.gradient(x)
    dy = np.gradient(y)
    p = Pxy*dx.reshape([len(x),1])*dy.reshape([1,len(y)])
    p = gaussian_filter(p,smooth)
    pr = p.ravel()
    s = np.argsort(pr)
    cr = np.zeros_like(pr)
    cr[s] = np.cumsum(pr[s])
    cr/=cr.max()
    c = cr.reshape(p.shape)
    return 1.-c

def luminosity_function(L,theta_pop=Ppop.default_theta_pop,res=100):
    """
    Return the luminosity function implied by a population model, evaluated at L (must be an array).
    """

    Ep = np.logspace(-1,6,res+1)
    
    Epg = Ep.reshape([len(Ep),1])
    Lg = L.reshape([1,len(L)])
    
    PEpL = Ppop.PEpL(L,Ep,theta_pop)
    
    return np.trapz(PEpL*Epg,np.log(Ep),axis=0)

def L_Ep_z_contours_and_cumulatives(theta_pop=Ppop.default_theta_pop,pflim=5.,inst='Fermi',alpha=-0.5,specmodel='Comp',res=50,smooth=0.5,pdet='gbm',pdetGW=None):
    L = np.logspace(logLmin,logLmax,res+1)
    Ep = np.logspace(logEpmin,logEpmax,res)
    z = np.logspace(logzmin,logzmax,res-1)
    
    zg = z.reshape([1,1,len(z)])
    Epg = Ep.reshape([len(Ep),1,1])
    Lg = L.reshape([1,len(L),1])
    
    if pdet=='gbm':
        pdet = pdet_GBM
    
    if pdet is None:
        if inst=='Fermi+Swift':
            pf_EpLz_fe = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst='Fermi')
            ep_EpLz = Epg/(1.+zg)
            pf_EpLz_sw = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst='Swift')
            Pdet = (pf_EpLz_fe>=pflim[0])*(pf_EpLz_sw>=pflim[1])
        else:
            pf_EpLz = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst=inst)
            Pdet = (pf_EpLz>pflim)
    else:
        if inst=='Fermi+Swift':
            pf_EpLz_fe = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst='Fermi')
            ep_EpLz = Epg/(1.+zg)
            pf_EpLz_sw = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst='Swift')
            Pdet = pdet(pf_EpLz_fe,ep_EpLz)*(pf_EpLz_sw>=pflim[1])
        elif inst=='Fermi+GW':
            pf_EpLz = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst='Fermi')
            ep_EpLz = Epg/(1.+zg)
            PdetGRB = pdet(pf_EpLz,ep_EpLz)
        else:
            pf_EpLz = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst=inst)
            ep_EpLz = Epg/(1.+zg)
            Pdet = pdet(pf_EpLz,ep_EpLz)

    Pz = Ppop.Pz(z,theta_pop)
    
    if inst=='Fermi+GW' :
        thv = np.logspace(logthvmin,np.log10(np.pi/2.),300)
        thvgGW,zgGW = np.broadcast_arrays(thv.reshape([1,len(thv)]),z.reshape([len(z),1]))
        PdetGW = pdetGW(zgGW.ravel(),thvgGW.ravel()).reshape([1,1,len(z),len(thv)])
        Pdet = np.trapz((thv*np.sin(thv)).reshape([1,1,1,300])*PdetGRB.reshape([len(Ep),len(L),len(z),1])*PdetGW,thv,axis=3)
                   
    PEpL = Ppop.PEpL(L,Ep,theta_pop)
    
    
    ppop = PEpL.reshape([len(Ep),len(L),1])*Pz
    
    PEpL_obs = np.trapz(zg*ppop*Pdet,np.log(z),axis=2)
    Pz_obs = np.trapz(np.trapz(Epg*Lg*ppop*Pdet,np.log(L),axis=1),np.log(Ep),axis=0)
    
    PEpL_obs_contours = greedy2d(np.log(Ep),np.log(L),Epg[:,:,0]*Lg[:,:,0]*PEpL_obs,smooth=smooth)
    
    PL_obs = np.trapz(Epg[:,:,0]*PEpL_obs,np.log(Ep),axis=0)
    PEp_obs = np.trapz(Lg[:,:,0]*PEpL_obs,np.log(L),axis=1)
    
    cum_L = cumtrapz(L*PL_obs,np.log(L),initial=0.)
    cum_L/=cum_L[-1]
    cum_Ep = cumtrapz(Ep*PEp_obs,np.log(Ep),initial=0.)
    cum_Ep/=cum_Ep[-1]
    cum_z = cumtrapz(z*Pz_obs,np.log(z),initial=0.)
    cum_z/=cum_z[-1]
    
    return L,Ep,z,PEpL_obs_contours,cum_L,cum_Ep,cum_z

def pf_ep_contours_and_cumulatives(theta_pop=Ppop.default_theta_pop,pflim=0.01,inst='Fermi',alpha=-0.5,specmodel='Comp',res=50,smooth=0.5,pdet='gbm'):
    
    L0 = np.logspace(logLmin,logLmax,res+1)
    Ep0 = np.logspace(logEpmin,logEpmax,res)
    
    PEpL = Ppop.PEpL(L0,Ep0,theta_pop)
    Itp_logPEpL = RegularGridInterpolator(points=(np.log10(Ep0),np.log10(L0)),values=np.log10(PEpL),bounds_error=False,fill_value=-np.inf)
    
    pf = np.logspace(np.log10(pflim),4.,res+1)
    ep = np.logspace(logEpmin-np.log10(1.+10**logzmax),logEpmax-np.log10(1.+10**logzmin),res)
    z = np.logspace(logzmin,logzmax,res-1)
    
    epg,pfg,zg = np.broadcast_arrays(ep.reshape([len(ep),1,1]),pf.reshape([1,len(pf),1]),z.reshape([1,1,len(z)]))
    
    Epg = epg*(1.+zg)
    Lg = pflux.L_from_phflux(zg,epg,pfg,alpha=alpha,model=specmodel,inst=inst)
    
    logEpLg = np.reshape((np.log10(Epg),np.log10(Lg)), (2, -1), order='C').T
    
    pepl = (10**Itp_logPEpL(logEpLg)).reshape([len(ep),len(pf),len(z)])
    
    Pz = Ppop.Pz(zg,theta_pop)
    
    Ppfep = np.trapz(np.nan_to_num(zg*pepl*Pz*(1.+zg)*Lg/pfg),np.log(z),axis=2)
    
    if pdet=='gbm':
        Ppfep *= pdet_GBM(pfg[:,:,0],epg[:,:,0])
    elif pdet is not None:
        Ppfep *= pdet(pfg[:,:,0],epg[:,:,0])
    
    Ppfep_contours = greedy2d(np.log(ep),np.log(pf),epg[:,:,0]*pfg[:,:,0]*Ppfep,smooth=smooth)
    
    Ppf = np.trapz(epg[:,:,0]*Ppfep,np.log(ep),axis=0)
    Pep = np.trapz(pfg[:,:,0]*Ppfep,np.log(pf),axis=1)
    
    cum_pf = cumtrapz(pf*Ppf,np.log(pf),initial=0.)
    cum_pf/=cum_pf[-1]
    cum_ep = cumtrapz(ep*Pep,np.log(ep),initial=0.)
    cum_ep/=cum_ep[-1]
    
    
    return pf,ep,Ppfep_contours,cum_pf,cum_ep

def chain_to_jet_structure(chain,theta_pop0=Ppop.default_theta_pop,chain_params=['thj','Lj','a_L','Epj','a_Ep']):

    Nsamples = chain.shape[0]
    
    thv0 = np.logspace(logthvmin,np.log10(np.pi/2.),100)
    Lthv = np.zeros([Nsamples,len(thv0)])
    Epthv = np.zeros([Nsamples,len(thv0)])

    for i in range(Nsamples):
        for j in range(len(chain_params)):
            theta_pop0[chain_params[j]]=chain[i,j]
        
        Lthv[i] = structjet.tildeL(thv0,theta_pop0)
        Epthv[i] = structjet.tildeEp(thv0,theta_pop0)
        
    return thv0,Lthv,Epthv


def core_L_Ep_contours_and_cumulatives(theta_pop=Ppop.default_theta_pop,res=100,smooth=0.5):
    L = np.logspace(logLmin,logLmax,res+1)
    Ep = np.logspace(logEpmin,logEpmax,res)
    
    Epg = Ep.reshape([len(Ep),1])
    Lg = L.reshape([1,len(L)])
       
    PEpL_core = Ppop.PEpLthv(Lg,Epg,0.,theta_pop)
    
    PEpL_core_contours = greedy2d(np.log(Ep),np.log(L),Epg[:,:]*Lg[:,:]*PEpL_core,smooth=smooth)
    
    PL_core = np.trapz(Epg[:,:]*PEpL_core,np.log(Ep),axis=0)
    PEp_core = np.trapz(Lg[:,:]*PEpL_core,np.log(L),axis=1)
    
    cum_L = cumtrapz(L*PL_core,np.log(L),initial=0.)
    cum_L/=cum_L[-1]
    cum_Ep = cumtrapz(Ep*PEp_core,np.log(Ep),initial=0.)
    cum_Ep/=cum_Ep[-1]
    
    return L,Ep,PEpL_core_contours,cum_L,cum_Ep

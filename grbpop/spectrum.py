import numpy as np

keV_to_erg = 1.60217662e-09

def dNdE_comp(E,Ep,alpha,beta):
    """ 
    GRB "comptonized" spectrum (power law with an exponential cut-off).
    
    Parameters:
    - E: photon energy
    - Ep: SED peak energy
    - alpha: power-law slope
    - beta: not used
    
    E and Ep must have the same units.
    
    Returns: dNdE
    - dNdE: (un-normalized) photon flux density at E
    """
    
    return E**alpha*np.exp(-(alpha+2.)*E/Ep)

def dNdE_Band(E,Ep,alpha,beta):
    """ 
    GRB "Band" spectrum (two power laws with an exponential transition).
    
    Parameters:
    - E: photon energy
    - Ep: SED peak energy
    - alpha: low-energy power-law slope
    - beta: high-energy power-law slope
    
    E and Ep must have the same units.
    
    Returns: dNdE
    - dNdE: (un-normalized) photon flux density at E
    """
    
    Ec = ((alpha-beta)*Ep/(alpha+2.))
    
    dnde = E**alpha*np.exp(-(alpha+2.)*E/Ep)
    dnde[E>Ec]=E[E>Ec]**beta*np.exp(beta-alpha)*Ec**(alpha-beta)
    
    return dnde



def photon_flux(Liso,Ep,alpha,z,dL,band=[10.,1000.],beta=-2.3,model='Comp',return_kcorr=False,Ebol_low=1.,Ebol_high=1e4):
    """ 
    GRB photon flux in a detector.
    
    Parameters:
    - Liso: isotropic equivalent luminosity [erg/s]
    - Ep: *rest frame* SED peak energy [keV]
    - alpha: low-energy photon index
    - z: redshift
    - dL: luminosity distance [cm]
    
    Keywords:
    - band: tuple containing the low and high ends of the detector band in keV
    - beta: the high-energy spectral slope, if applicable
    - model: the spectrum model (either 'Comp' or 'Band')
    - return_kcorr: if True, return also the k correction
    - Ebol_low: lower end of the "bolometric" luminosity band (keV)
    - Ebol_high: upper end of the "bolometric" luminosity band (keV)
    
    Returns: P (, k)
    - P: photon flux in ph cm-2 s-1
    - k: k correction (only if return_kcorr=True)
    """
    
    if model=='Band' or model=='band':
        dNdE = dNdE_Band
    else:
        dNdE = dNdE_comp
    
    E = np.logspace(np.log10(Ebol_low/(1.+z)),np.log10(Ebol_high/(1.+z)),1000)
    F = np.trapz(E**2*dNdE(E,Ep/(1.+z),alpha,beta),np.log(E))*keV_to_erg
    
    E = np.logspace(np.log10(band[0]),np.log10(band[1]),300)
    k = F/np.trapz(E*dNdE(E,Ep/(1.+z),alpha,beta),np.log(E))
    if k>0.:
        P = Liso/(4*np.pi*dL**2*k)
    else:
        P = 0.
    
    if return_kcorr:
        return P,k
    else:
        return P

def energy_flux(Liso,Ep,alpha,z,dL,band=[10.,1000.],beta=-2.3,model='Comp',Ebol_low=1.,Ebol_high=1e4):
    """ 
    GRB energy flux in a detector.
    
    Parameters:
    - Liso: isotropic equivalent luminosity [erg/s] (assumed to be in the rest-frame 1-10000 keV band)
    - Ep: *rest frame* SED peak energy [keV]
    - alpha: low-energy photon index
    - z: redshift
    - dL: luminosity distance [cm]
    
    Keywords:
    - band: tuple containing the low and high ends of the detector band in keV
    - beta: the high-energy spectral slope, if applicable
    - model: the spectrum model (either 'Comp' or 'Band')
    
    Returns: F
    - F: energy flux in erg cm-2 s-1
    """
    
    if model=='Band' or model=='band':
        dNdE = dNdE_Band
    else:
        dNdE = dNdE_comp
    
    E = np.logspace(np.log10(Ebol_low/(1.+z)),np.log10(Ebol_high/(1.+z)),1000)
    F = np.trapz(E**2*dNdE(E,Ep/(1.+z),alpha,beta),np.log(E))*keV_to_erg
    
    E = np.logspace(np.log10(band[0]),np.log10(band[1]),300)
    P = Liso/(4*np.pi*dL**2*F)*np.trapz(E*dNdE(E,Ep/(1.+z),alpha,beta),np.log(E))
    
    return P



def L_from_phflux(phflux,Epobs,alpha,z,dL,band=[15.,150.],beta=-2.3,model='Comp',Ebol_low=1.,Ebol_high=1e4):
    if model=='Band' or model=='band':
        dNdE = dNdE_Band
    else:
        dNdE = dNdE_comp
    
    EI = np.logspace(np.log10(band[0]),np.log10(band[1]),1000)
    E0 = np.logspace(np.log10(Ebol_low),np.log10(Ebol_high),1000)*(1.+z)
    pI = np.trapz(EI*dNdE(EI,Epobs,alpha,beta),np.log(EI))
    F = np.trapz(E0**2*dNdE(E0,Epobs,alpha,beta),np.log(E0))*keV_to_erg
    return 4*np.pi*dL**2*F*phflux/pI
    
def Eiso_from_fluence(flnc,Epobs,alpha,z,dL,band=[15.,150.],beta=-2.3,model='Comp',Ebol_low=1.,Ebol_high=1e4):
    if model=='Band' or model=='band':
        dNdE = dNdE_Band
    else:
        dNdE = dNdE_comp
    
    EI = np.logspace(np.log10(band[0]),np.log10(band[1]),1000)
    E0 = np.logspace(np.log10(Ebol_low),np.log10(Ebol_high),1000)
    fI = np.trapz(EI**2*dNdE(EI,Epobs,alpha,beta),np.log(EI))
    F = np.trapz(E0**2*dNdE(E0,Epobs,alpha,beta),np.log(E0))
    return 4*np.pi*dL**2*F*flnc/fI/(1.+z)



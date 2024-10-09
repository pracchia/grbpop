import numpy as np
from scipy.interpolate import RegularGridInterpolator
from astropy.cosmology import Planck15 as cosmo
import os
import pathlib
here = pathlib.Path(__file__).parent.resolve()

z0 = np.logspace(-4,1,1000)
dL0 = cosmo.luminosity_distance(z0).to('cm').value

z_grid = np.load(os.path.join(here,'pflux_tables/z.npy'))
Ep_grid = np.load(os.path.join(here,'pflux_tables/Ep.npy'))
alpha_grid = np.load(os.path.join(here,'pflux_tables/alpha.npy'))
pflux_L_Fermi_Comp = np.load(os.path.join(here,'pflux_tables/pflux_L_Fermi_Comp.npy'))
pflux_L_Swift_Comp = np.load(os.path.join(here,'pflux_tables/pflux_L_Swift_Comp.npy'))
kcorr_Fermi_Comp = np.load(os.path.join(here,'pflux_tables/kcorr_Fermi_Comp.npy'))
kcorr_Swift_Comp = np.load(os.path.join(here,'pflux_tables/kcorr_Swift_Comp.npy'))
pflux_L_Fermi_Band = np.load(os.path.join(here,'pflux_tables/pflux_L_Fermi_Band.npy'))
pflux_L_Swift_Band = np.load(os.path.join(here,'pflux_tables/pflux_L_Swift_Band.npy'))
kcorr_Fermi_Band = np.load(os.path.join(here,'pflux_tables/kcorr_Fermi_Band.npy'))
kcorr_Swift_Band = np.load(os.path.join(here,'pflux_tables/kcorr_Swift_Band.npy'))



Itp_Fermi_Comp = RegularGridInterpolator(points=(np.log10(z_grid),np.log10(Ep_grid),alpha_grid),values=np.nan_to_num(np.log10(pflux_L_Fermi_Comp)),bounds_error=False)
Itp_Swift_Comp = RegularGridInterpolator(points=(np.log10(z_grid),np.log10(Ep_grid),alpha_grid),values=np.nan_to_num(np.log10(pflux_L_Swift_Comp)),bounds_error=False)
Itp_k_Fermi_Comp = RegularGridInterpolator(points=(np.log10(z_grid),np.log10(Ep_grid),alpha_grid),values=np.nan_to_num(np.log10(kcorr_Fermi_Comp)),bounds_error=False)
Itp_k_Swift_Comp = RegularGridInterpolator(points=(np.log10(z_grid),np.log10(Ep_grid),alpha_grid),values=np.nan_to_num(np.log10(kcorr_Swift_Comp)),bounds_error=False)

Itp_Fermi_Band = RegularGridInterpolator(points=(np.log10(z_grid),np.log10(Ep_grid),alpha_grid),values=np.nan_to_num(np.log10(pflux_L_Fermi_Band)),bounds_error=False)
Itp_Swift_Band = RegularGridInterpolator(points=(np.log10(z_grid),np.log10(Ep_grid),alpha_grid),values=np.nan_to_num(np.log10(pflux_L_Swift_Band)),bounds_error=False)
Itp_k_Fermi_Band = RegularGridInterpolator(points=(np.log10(z_grid),np.log10(Ep_grid),alpha_grid),values=np.nan_to_num(np.log10(kcorr_Fermi_Band)),bounds_error=False)
Itp_k_Swift_Band = RegularGridInterpolator(points=(np.log10(z_grid),np.log10(Ep_grid),alpha_grid),values=np.nan_to_num(np.log10(kcorr_Swift_Band)),bounds_error=False)


def pflux_from_L(z,Ep,L,alpha=-0.5,model='Comp',inst='Fermi'):
    
    # check bounds
    zbounded = np.maximum(np.minimum(z,z_grid[-1]),z_grid[0])
    Epbounded = np.maximum(np.minimum(Ep,Ep_grid[-1]),Ep_grid[0])
    alphabounded = np.maximum(np.minimum(alpha,alpha_grid[-1]),alpha_grid[0])
    
    if not np.isscalar(z):
        zEp_mesh = np.broadcast_arrays(np.log10(zbounded),np.log10(Epbounded),alphabounded)
        
        zEp_list = np.reshape(zEp_mesh, (3, -1), order='C').T
    else:
        zEp_list = np.array([np.log10(z),np.log10(Ep),alpha])

    if inst=='Fermi':
        if model=='Comp':
            p = np.nan_to_num(10**Itp_Fermi_Comp(zEp_list))
        else:
            p = np.nan_to_num(10**Itp_Fermi_Band(zEp_list))
    elif inst=='Swift':
        if model=='Comp':
            p = np.nan_to_num(10**Itp_Swift_Comp(zEp_list))
        else:
            p = np.nan_to_num(10**Itp_Swift_Band(zEp_list))
            
    if not np.isscalar(z):
        return L*p.reshape(zEp_mesh[0].shape)
    else:
        return L*p
    

def Fermi(z,Ep,L,alpha=-0.5,model='Comp'):
    return pflux_from_L(z,Ep,L,alpha=alpha,model=model,inst='Fermi')

def Swift(z,Ep,L,alpha=-0.5,model='Comp'):
    return pflux_from_L(z,Ep,L,alpha=alpha,model=model,inst='Swift')


def L_from_phflux(z,ep,pf,alpha=-0.5,model='Comp',inst='Fermi'):
    
    Ep = ep*(1.+z)
    dL = np.interp(z,z0,dL0)
    
    if not np.isscalar(z):
        zEp_mesh = np.broadcast_arrays(np.log10(z),np.log10(Ep),alpha)
        
        zEp_list = np.reshape(zEp_mesh, (3, -1), order='C').T
    else:
        zEp_list = np.array([np.log10(z),np.log10(Ep),alpha])

    if inst=='Fermi':
        if model=='Comp':
            k = np.nan_to_num(10**Itp_k_Fermi_Comp(zEp_list))
        else:
            k = np.nan_to_num(10**Itp_k_Fermi_Band(zEp_list))
    elif inst=='Swift':
        if model=='Comp':
            k = np.nan_to_num(10**Itp_k_Swift_Comp(zEp_list))
        else:
            k = np.nan_to_num(10**Itp_k_Swift_Band(zEp_list))
    if not np.isscalar(z):
        return 4*np.pi*dL**2*pf*k.reshape(zEp_mesh[0].shape)
    else:
        return 4*np.pi*dL**2*pf*k


def L_from_phflux_biased_ep(z,epbias,pf,alpha=-0.5,model='Comp',inst='Fermi'):
    if not np.isscalar(z):
        Ep = np.zeros_like(z)+epbias
    else:
        Ep = epbias
    
    Ep = np.zeros_like(z)+epbias
    dL = np.interp(z,z0,dL0)
    
    if not np.isscalar(z):
        zEp_mesh = np.broadcast_arrays(np.log10(z),np.log10(Ep),alpha)
        
        zEp_list = np.reshape(zEp_mesh, (3, -1), order='C').T
    else:
        zEp_list = np.array([np.log10(z),np.log10(Ep),alpha])

    if inst=='Fermi':
        if model=='Comp':
            k = np.nan_to_num(10**Itp_k_Fermi_Comp(zEp_list))
        else:
            k = np.nan_to_num(10**Itp_k_Fermi_Band(zEp_list))
    elif inst=='Swift':
        if model=='Comp':
            k = np.nan_to_num(10**Itp_k_Swift_Comp(zEp_list))
        else:
            k = np.nan_to_num(10**Itp_k_Swift_Band(zEp_list))
    if not np.isscalar(z):
        return 4*np.pi*dL**2*pf*k.reshape(zEp_mesh[0].shape)
    else:
        return 4*np.pi*dL**2*pf*k

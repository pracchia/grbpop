import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from scipy.interpolate import RegularGridInterpolator
import spectrum

Ep = np.logspace(-1,6,60)
z = np.logspace(-4,1,100)
alphas = np.linspace(-1.9,1.,30)

Fermi = [50.,300.]
Swift = [15.,150.]

for model in ['Comp','Band']:
    
    if model=='Comp':
        dNdE = spectrum.dNdE_comp
    else:
        dNdE = spectrum.dNdE_Band
    
    pflux_L_Fermi = np.zeros([len(z),len(Ep),len(alphas)])
    pflux_L_Swift = np.zeros([len(z),len(Ep),len(alphas)])
    kcorr_Fermi = np.zeros([len(z),len(Ep),len(alphas)])
    kcorr_Swift = np.zeros([len(z),len(Ep),len(alphas)])
    
    EI_F = np.logspace(np.log10(50),np.log10(300),300)
    EI_S = np.logspace(np.log10(15.),np.log10(150.),200)
    
    dL = cosmo.luminosity_distance(z).to('cm').value
    
    print('')
    for i in range(len(z)):
        Er = np.logspace(-1,7,1000)*(1.+z[i])
        print('Computing fluxes for z = {0:.4f} ...           '.format(z[i]),end='\r')
        for j in range(len(Ep)):
            for k in range(len(alphas)):
                pflux_L_Fermi[i,j,k],kcorr_Fermi[i,j,k]=spectrum.photon_flux(1.,Ep[j],alphas[k],z[i],dL[i],band=Fermi,model=model,return_kcorr=True,Ebol_low=0.1,Ebol_high=1e7)
                pflux_L_Swift[i,j,k],kcorr_Swift[i,j,k]=spectrum.photon_flux(1.,Ep[j],alphas[k],z[i],dL[i],band=Swift,model=model,return_kcorr=True,Ebol_low=0.1,Ebol_high=1e7)
                
    print('..done!                        ')
    
    np.save('pflux_tables/alpha.npy',alphas)
    np.save('pflux_tables/Ep.npy',Ep)
    np.save('pflux_tables/z.npy',z)
    np.save('pflux_tables/dL.npy',dL)
    np.save('pflux_tables/pflux_L_Fermi_{}.npy'.format(model),pflux_L_Fermi)
    np.save('pflux_tables/pflux_L_Swift_{}.npy'.format(model),pflux_L_Swift)
    np.save('pflux_tables/kcorr_Fermi_{}.npy'.format(model),kcorr_Fermi)
    np.save('pflux_tables/kcorr_Swift_{}.npy'.format(model),kcorr_Swift)
    




for model in ['Comp','Band']:
    
    # check
    Itp_Fermi = RegularGridInterpolator(points=(np.log10(z),np.log10(Ep),alphas),values=np.log10(pflux_L_Fermi),bounds_error=False)
    Itp_Swift = RegularGridInterpolator(points=(np.log10(z),np.log10(Ep),alphas),values=np.log10(pflux_L_Swift),bounds_error=False)
    
    z = np.random.uniform(0.001,10.,100)
    dL = cosmo.luminosity_distance(z).to('cm').value
    Ep = 10**np.random.uniform(-1,6,100)
    alpha = np.random.uniform(-1.9,1,100)
    
    res_Swift = np.zeros_like(z)
    res_Fermi = np.zeros_like(z)
    
    pfFi = 10**Itp_Fermi(np.vstack([np.log10(z),np.log10(Ep),alpha]).T)
    pfSi = 10**Itp_Swift(np.vstack([np.log10(z),np.log10(Ep),alpha]).T)
    
    for i in range(len(z)):
        pfF = spectrum.photon_flux(1.,Ep[i],alpha[i],z[i],dL[i],band=Fermi,model=model,Ebol_low=0.1,Ebol_high=1e7)
        pfS = spectrum.photon_flux(1.,Ep[i],alpha[i],z[i],dL[i],band=Swift,model=model,Ebol_low=0.1,Ebol_high=1e7)
        
        res_Swift[i]=np.abs(pfSi[i]/pfS - 1.)
        res_Fermi[i]=np.abs(pfFi[i]/pfF - 1.)
    
    plt.plot(Ep,res_Swift,marker='*',ls='None',alpha=0.5,label='Swift/BAT band')
    plt.plot(Ep,res_Fermi,marker='s',ls='None',alpha=0.5,label='Fermi/GBM band')
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel(r'$E_p$ [keV]')
    plt.ylabel(r'| relative error |')
    
    plt.legend()
    
    plt.show()

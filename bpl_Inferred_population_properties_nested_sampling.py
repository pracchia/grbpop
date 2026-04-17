import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from astropy.io import ascii
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import grbpop, os, pathlib
from dynesty import DynamicNestedSampler
from inspect_fit_results import read_chain
here = pathlib.Path(__file__).parent.resolve()

# 2d greedy binning
def samples_to_mesh(x,y,bins=(30,31),smooth=0.7,weights=None):
    h,bx,by = np.histogram2d(x,y,bins=bins,range=[[x.min()*0.9, x.max()*1.1], [y.min()*0.9, y.max()*1.1]],weights=weights)
    h = gaussian_filter(h,sigma=smooth)
    h_r = h.ravel()
    c_r = np.zeros_like(h_r)
    sort_idx = np.argsort(h_r)
    c_r[sort_idx] = np.cumsum(h_r[sort_idx])
    c = c_r.reshape(h.shape)
    c/=c.max()
    bxm,bym = np.meshgrid((bx[1:]+bx[:-1])/2.,(by[1:]+by[:-1])/2.)
    return (1.-c).T,bxm,bym

plt.rcParams['font.family']='serif'
plt.rcParams['figure.figsize']=4,3.5
plt.rcParams['figure.autolayout']=False
plt.rcParams['mathtext.fontset']='dejavuserif'
plt.rcParams['font.size']=11

def psi_g16(z,p1,p2,p3):
    """
    Functional form for rate density evolution adopted in Ghirlanda+2016
    """
    return (1.+p1*z)/(1.+(z/p2)**p3)    

def psi_wp15(z):
    """
    Fitting formula for rate density evolution adopted in Wanderman & Piran 2015 
    """
    return 45./4.1*np.where(z<=0.9,np.exp((z-0.9)/0.39),np.exp(-(z-0.9)/0.26))


recompute = True
# recompute = False

# If the chain contains R0, set 'Poisson = True'. Otherwise, set 'Poisson = False' and compute R0.
Poisson = True
# Poisson = False

suffix = '' 

specmodel = 'Comp'
alpha = -0.4
inst = 'Fermi'
# pflim = 3.5
pflim = 2.37
res = 80
Robs = 212./0.59/10. # Fermi GBM SGRBs with p64>pflim, corrected for FoV and duty cycle

# thin = 10
thin = 1


#######################
### Nested sampling ###
#######################

folder = 'nested_samplings/'
# chain = 'SUB_nested_BPL_flux-limited_tighter_Poisson_dtdsfh_POW_dlogz0.001_nlive1200_nbatch240_neff20000.save'
# chain = 'SUB_nested_BPL_flux-limited_tighter_Poisson_dtdsfh_POW_WRONGCUT_dlogz0.001_nlive1200_nbatch240_neff20000.save'

chain = 'final_nested_BPL_flux-limited_tighter_Poisson_dtdsfh_POW_dlogz0.001_nlive1200_nbatch240_neff20000.save'

label_chain_1 = 'Pow'
dsampler1 = DynamicNestedSampler.restore(folder + chain)

folder2 = 'nested_samplings/'
# chain2 = None
# chain2 = 'SUB_nested_BPL_flux-limited_tighter_Poisson_dtdsfh_LOG_dlogz0.001_nlive1200_nbatch240_neff20000.save'
# chain2 = 'SUB_nested_BPL_flux-limited_tighter_Poisson_dtdsfh_LOG_WRONGCUT_dlogz0.001_nlive1200_nbatch240_neff20000.save'

chain2 = 'final_nested_BPL_flux-limited_tighter_Poisson_dtdsfh_LOG_WRONGCUT_dlogz0.001_nlive1200_nbatch240_neff20000.save'

label_chain_2 = 'Log'
dsampler2 = DynamicNestedSampler.restore(folder2 + chain2)

x = dsampler1.results.samples_equal()

if chain2 is not None:
    x2 = dsampler2.results.samples_equal()

#######################
#######################


############
### MCMC ###
############

# folder = 'chains/'
# chain = 'draft_MCMC_BPL_flux-limited_Poisson_dtdsfh_POW.h5'
# label_chain_1 = 'Pow'
    
# folder2 = 'chains/'
# # chain2 = None
# chain2 = 'draft_MCMC_BPL_flux-limited_Poisson_dtdsfh_LOG.h5'
# label_chain_2 = 'Log'

# x,ll = read_chain(folder+chain,burnin_fraction=0.5,thin=thin)

# if chain2 is not None:
#     x2,ll2 = read_chain(folder2+chain2,burnin_fraction=0.5,thin=thin)
    # x2,ll2 = read_chain(chain2,burnin_fraction=0.25,thin=thin)

############
############


# N = 10000
N = int(np.min([len(x),len(x2)])/2)
N2 = N

th = np.logspace(grbpop.globals.logthvmin,np.log10(np.pi/2.),300)
L = np.logspace(grbpop.globals.logLmin-2,grbpop.globals.logLmax,res+1)
Ep = np.logspace(grbpop.globals.logEpmin,grbpop.globals.logEpmax,res)
z = np.logspace(grbpop.globals.logzmin,grbpop.globals.logzmax,res-1)
dVdz = 4*np.pi*cosmo.differential_comoving_volume(z).to('Gpc3 sr-1').value

zg = z.reshape([1,1,len(z)])
Epg = Ep.reshape([len(Ep),1,1])
Lg = L.reshape([1,len(L),1])


pf_EpLz = grbpop.pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst=inst)
Pdet = pf_EpLz>=pflim

if recompute:
    z_grid = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/z.npy'))
    tdmin_grid = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/tdmin.npy'))
    at_grid = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/at.npy'))
    rhoz_grid_pow = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/r_sgrb_pow.npy'))
    Itp_rhoz_pow = RegularGridInterpolator(points=(np.log10(z_grid),tdmin_grid,at_grid),values=np.nan_to_num(rhoz_grid_pow),bounds_error=False)
    
    mu_td_grid = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/mu_td.npy'))
    sigma_td_grid = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/sigma_td.npy'))
    rhoz_grid_log = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/r_sgrb_log.npy'))
    Itp_rhoz_log = RegularGridInterpolator(points=(np.log10(z_grid),mu_td_grid,sigma_td_grid),values=np.nan_to_num(rhoz_grid_log),bounds_error=False)

    R0 = np.zeros(N)
    dR0_dlogL = np.zeros([N,len(L)])
    dN_dVdt = np.zeros([N,len(z)])
    tildeL = np.zeros([N,len(th)])
    tildeEp = np.zeros([N,len(th)])
    R0 = np.zeros(N)
    
    if chain2 is not None:
        R02 = np.zeros(N)
        dR0_dlogL2 = np.zeros([N,len(L)])
        dN_dVdt2 = np.zeros([N,len(z)])
        tildeL2 = np.zeros([N,len(th)])
        tildeEp2 = np.zeros([N,len(th)])
    
    print('')
    for i in range(N):
        print('Sample {0:d}/{1:d} ...   '.format(i,N),end='\r')
        
        if Poisson:
            # theta_pop = {'rho_z':'SBPL',
            #          'alpha_L':x[-thin*i,0],
            #          'beta_L':x[-thin*i,1],
            #          'gamma_L':x[-thin*i,2],
            #          'L_*':10.**x[-thin*i,3],
            #          'L_**':10.**x[-thin*i,4],
            #          'L_0':10.**x[-thin*i,5],
            #          'E_p*':10.**x[-thin*i,6],
            #          's_c':10.**x[-thin*i,7],
            #          'y':x[-thin*i,8],
            #          'a':x[-thin*i,9],
            #          'b':x[-thin*i,10],
            #          'zp':x[-thin*i,11],
            #          'R0':10**x[-thin*i,12]
            #          }
            theta_pop = {'rho_z':'DTD*SFH',
                     'dtd':'pow',
                     'alpha_L':x[-thin*i,0],
                     'beta_L':x[-thin*i,1],
                     'gamma_L':x[-thin*i,2],
                     'L_*':10.**x[-thin*i,3],
                     'L_**':10.**x[-thin*i,4],
                     'L_0':10.**x[-thin*i,5],
                     'E_p*':10.**x[-thin*i,6],
                     's_c':10.**x[-thin*i,7],
                     'y':x[-thin*i,8],
                     'tdmin':10**x[-thin*i,9],
                     'at':x[-thin*i,10],
                     'R0':10**x[-thin*i,11]
                     }
            # theta_pop = {'rho_z':'DTD*SFH',
            #          'dtd':'lognorm',
            #          'alpha_L':x[-thin*i,0],
            #          'beta_L':x[-thin*i,1],
            #          'gamma_L':x[-thin*i,2],
            #          'L_*':10.**x[-thin*i,3],
            #          'L_**':10.**x[-thin*i,4],
            #          'L_0':10.**x[-thin*i,5],
            #          'E_p*':10.**x[-thin*i,6],
            #          's_c':10.**x[-thin*i,7],
            #          'y':x[-thin*i,8],
            #          'mu_td':x[-thin*i,9],
            #          'sigma_td':x[-thin*i,10],
            #          'R0':10**x[-thin*i,11]
            #          }

            if chain2 is not None:
                # theta_pop2 = {'rho_z':'SBPL',
                #          'alpha_L':x2[-thin*i,0],
                #          'beta_L':x2[-thin*i,1],
                #          'gamma_L':x2[-thin*i,2],
                #          'L_*':10.**x2[-thin*i,3],
                #          'L_**':10.**x2[-thin*i,4],
                #          'L_0':10.**x2[-thin*i,5],
                #          'E_p*':10.**x2[-thin*i,6],
                #          's_c':10.**x2[-thin*i,7],
                #          'y':x2[-thin*i,8],
                #          'a':x2[-thin*i,9],
                #          'b':x2[-thin*i,10],
                #          'zp':x2[-thin*i,11],
                #          'R0':10**x2[-thin*i,12]
                #          }
                # theta_pop2 = {'rho_z':'DTD*SFH',
                #          'dtd':'pow',
                #          'alpha_L':x2[-thin*i,0],
                #          'beta_L':x2[-thin*i,1],
                #          'gamma_L':x2[-thin*i,2],
                #          'L_*':10.**x2[-thin*i,3],
                #          'L_**':10.**x2[-thin*i,4],
                #          'L_0':10.**x2[-thin*i,5],
                #          'E_p*':10.**x2[-thin*i,6],
                #          's_c':10.**x2[-thin*i,7],
                #          'y':x2[-thin*i,8],
                #          'tdmin':x2[-thin*i,9],
                #          'at':x2[-thin*i,10],
                #          'R0':10**x2[-thin*i,11]
                #          }
                theta_pop2 = {'rho_z':'DTD*SFH',
                         'dtd':'lognorm',
                         'alpha_L':x2[-thin*i,0],
                         'beta_L':x2[-thin*i,1],
                         'gamma_L':x2[-thin*i,2],
                         'L_*':10.**x2[-thin*i,3],
                         'L_**':10.**x2[-thin*i,4],
                         'L_0':10.**x2[-thin*i,5],
                         'E_p*':10.**x2[-thin*i,6],
                         's_c':10.**x2[-thin*i,7],
                         'y':x2[-thin*i,8],
                         'mu_td':10**x2[-thin*i,9],
                         'sigma_td':10**x2[-thin*i,10],
                         'R0':10**x2[-thin*i,11]
                         }
        
        else:
            # theta_pop = {'rho_z':'SBPL',
            #          'alpha_L':x[-thin*i,0],
            #          'beta_L':x[-thin*i,1],
            #          'gamma_L':x[-thin*i,2],
            #          'L_*':10.**x[-thin*i,3],
            #          'L_**':10.**x[-thin*i,4],
            #          'L_0':10.**x[-thin*i,5],
            #          'E_p*':10.**x[-thin*i,6],
            #          's_c':10.**x[-thin*i,7],
            #          'y':x[-thin*i,8],
            #          'a':x[-thin*i,9],
            #          'b':x[-thin*i,10],
            #          'zp':x[-thin*i,11]
            #          }
            theta_pop = {'rho_z':'DTD*SFH',
                     'dtd':'pow',
                     'alpha_L':x[-thin*i,0],
                     'beta_L':x[-thin*i,1],
                     'gamma_L':x[-thin*i,2],
                     'L_*':10.**x[-thin*i,3],
                     'L_**':10.**x[-thin*i,4],
                     'L_0':10.**x[-thin*i,5],
                     'E_p*':10.**x[-thin*i,6],
                     's_c':10.**x[-thin*i,7],
                     'y':x[-thin*i,8],
                     'tdmin':x[-thin*i,9],
                     'at':x[-thin*i,10]
                     }
            # theta_pop = {'rho_z':'DTD*SFH',
            #          'dtd':'lognorm',
            #          'alpha_L':x[-thin*i,0],
            #          'beta_L':x[-thin*i,1],
            #          'gamma_L':x[-thin*i,2],
            #          'L_*':10.**x[-thin*i,3],
            #          'L_**':10.**x[-thin*i,4],
            #          'L_0':10.**x[-thin*i,5],
            #          'E_p*':10.**x[-thin*i,6],
            #          's_c':10.**x[-thin*i,7],
            #          'y':x[-thin*i,8],
            #          'mu_td':x[-thin*i,9],
            #          'sigma_td':x[-thin*i,10]
            #          }
            
            if chain2 is not None:
                # theta_pop2 = {'rho_z':'SBPL',
                #          'alpha_L':x2[-thin*i,0],
                #          'beta_L':x2[-thin*i,1],
                #          'gamma_L':x2[-thin*i,2],
                #          'L_*':10.**x2[-thin*i,3],
                #          'L_**':10.**x2[-thin*i,4],
                #          'L_0':10.**x2[-thin*i,5],
                #          'E_p*':10.**x2[-thin*i,6],
                #          's_c':10.**x2[-thin*i,7],
                #          'y':x2[-thin*i,8],
                #          'a':x2[-thin*i,9],
                #          'b':x2[-thin*i,10],
                #          'zp':x2[-thin*i,11]
                #          }
                theta_pop2 = {'rho_z':'DTD*SFH',
                         'dtd':'pow',
                         'alpha_L':x2[-thin*i,0],
                         'beta_L':x2[-thin*i,1],
                         'gamma_L':x2[-thin*i,2],
                         'L_*':10.**x2[-thin*i,3],
                         'L_**':10.**x2[-thin*i,4],
                         'L_0':10.**x2[-thin*i,5],
                         'E_p*':10.**x2[-thin*i,6],
                         's_c':10.**x2[-thin*i,7],
                         'y':x2[-thin*i,8],
                         'tdmin':x2[-thin*i,9],
                         'at':x2[-thin*i,10]
                         }
                # theta_pop2 = {'rho_z':'DTD*SFH',
                #          'dtd':'lognorm',
                #          'alpha_L':x2[-thin*i,0],
                #          'beta_L':x2[-thin*i,1],
                #          'gamma_L':x2[-thin*i,2],
                #          'L_*':10.**x2[-thin*i,3],
                #          'L_**':10.**x2[-thin*i,4],
                #          'L_0':10.**x2[-thin*i,5],
                #          'E_p*':10.**x2[-thin*i,6],
                #          's_c':10.**x2[-thin*i,7],
                #          'y':x2[-thin*i,8],
                #          'mu_td':x2[-thin*i,9],
                #          'sigma_td':x2[-thin*i,10]
                #          }
        
        # PEpL = grbpop.Ppop.PEpL(L,Ep,theta_pop,grid=True)
        # PEpL/=np.trapezoid(np.trapezoid(PEpL*Epg[:,:,0]*Lg[:,:,0],np.log(L),axis=1),np.log(Ep))
        if (theta_pop['rho_z']=='SBPL'): 
            rhoz = grbpop.Ppop.MD14_SFH(z,theta_pop['a'],theta_pop['b'],theta_pop['zp'])
            rhoz/=rhoz[0]
        elif (theta_pop['rho_z']=='DTD*SFH' and theta_pop['dtd']=='pow'):
            rhoz = Itp_rhoz_pow((np.log10(z),theta_pop['tdmin'],theta_pop['at']))
            rhoz/=rhoz[0]
        elif (theta_pop['rho_z']=='DTD*SFH' and theta_pop['dtd']=='lognorm'):
            rhoz = Itp_rhoz_log((np.log10(z),theta_pop['mu_td'],theta_pop['sigma_td']))
            rhoz/=rhoz[0]

        psiz = rhoz/(1.+z)*dVdz
        if Poisson:
            R0[i] = theta_pop['R0']
        else:
            PEpL = grbpop.Ppop.PEpL(L,Ep,theta_pop,grid=True)
            PEpL/=np.trapezoid(np.trapezoid(PEpL*Epg[:,:,0]*Lg[:,:,0],np.log(L),axis=1),np.log(Ep))
            R0[i] = Robs/np.trapezoid(np.trapezoid(np.trapezoid(PEpL.reshape([len(Ep),len(L),1])*psiz.reshape([1,1,len(z)])*Epg*Lg*zg*Pdet,np.log(z),axis=2),np.log(L),axis=1),np.log(Ep))
        R0[i] = np.nan_to_num(R0[i])
        # dR0_dlogL[i] = L*R0[i]*grbpop.Ppop.lum_2breaks_pdf(L,theta_pop)
        dR0_dlogL[i] = L*R0[i]*grbpop.diagnose.luminosity_function_2breaks(L,theta_pop)
        dN_dVdt[i] = R0[i]*psiz*(1.+z)/dVdz

        if chain2 is not None:
            # PEpL2 = grbpop.Ppop.PEpL(L,Ep,theta_pop2,grid=True)
            # PEpL2/=np.trapezoid(np.trapezoid(PEpL2*Epg[:,:,0]*Lg[:,:,0],np.log(L),axis=1),np.log(Ep))
            if (theta_pop2['rho_z']=='SBPL'): 
                rhoz2 = grbpop.Ppop.MD14_SFH(z,theta_pop2['a'],theta_pop2['b'],theta_pop2['zp'])
                rhoz2/=rhoz2[0]
            elif (theta_pop2['rho_z']=='DTD*SFH' and theta_pop2['dtd']=='pow'):
                rhoz2 = Itp_rhoz_pow((np.log10(z),theta_pop2['tdmin'],theta_pop2['at']))
                rhoz2/=rhoz2[0]
            elif (theta_pop2['rho_z']=='DTD*SFH' and theta_pop2['dtd']=='lognorm'):
                rhoz2 = Itp_rhoz_log((np.log10(z),theta_pop2['mu_td'],theta_pop2['sigma_td']))
                rhoz2/=rhoz2[0]
            
            psiz2 = rhoz2/(1.+z)*dVdz
            if Poisson:
                R02[i] = theta_pop2 ['R0']
            else: 
                PEpL2 = grbpop.Ppop.PEpL(L,Ep,theta_pop2,grid=True)
                PEpL2/=np.trapezoid(np.trapezoid(PEpL2*Epg[:,:,0]*Lg[:,:,0],np.log(L),axis=1),np.log(Ep))
                R02[i] = Robs/np.trapezoid(np.trapezoid(np.trapezoid(PEpL2.reshape([len(Ep),len(L),1])*psiz2.reshape([1,1,len(z)])*Epg*Lg*zg*Pdet,np.log(z),axis=2),np.log(L),axis=1),np.log(Ep))
            R02[i] = np.nan_to_num(R02[i])
            # dR0_dlogL2[i] = L*R02[i]*grbpop.Ppop.lum_2breaks_pdf(L,theta_pop2)
            dR0_dlogL2[i] = L*R02[i]*grbpop.diagnose.luminosity_function_2breaks(L,theta_pop2)
            dN_dVdt2[i] = R02[i]*psiz2*(1.+z)/dVdz
    
    print('Done.                  ')
    
    np.save('cache/results_R0{0}.npy'.format(suffix),R0)
    np.save('cache/results_dR0_dlogL{0}.npy'.format(suffix),dR0_dlogL)
    np.save('cache/results_dN_dVdt{0}.npy'.format(suffix),dN_dVdt)
    np.save('cache/results_tildeL{0}.npy'.format(suffix),tildeL)
    np.save('cache/results_tildeEp{0}.npy'.format(suffix),tildeEp)

    if chain2 is not None:
        np.save('cache/results_R02{0}.npy'.format(suffix),R02)
        np.save('cache/results_dR0_dlogL2{0}.npy'.format(suffix),dR0_dlogL2)
        np.save('cache/results_dN_dVdt2{0}.npy'.format(suffix),dN_dVdt2)
        np.save('cache/results_tildeL2{0}.npy'.format(suffix),tildeL2)
        np.save('cache/results_tildeEp2{0}.npy'.format(suffix),tildeEp2)


else:
    R0 = np.load('cache/results_R0{0}.npy'.format(suffix))
    dR0_dlogL = np.load('cache/results_dR0_dlogL{0}.npy'.format(suffix))
    dN_dVdt = np.load('cache/results_dN_dVdt{0}.npy'.format(suffix))
    tildeL = np.load('cache/results_tildeL{0}.npy'.format(suffix))
    tildeEp = np.load('cache/results_tildeEp{0}.npy'.format(suffix))

    if chain2 is not None:
        R02 = np.load('cache/results_R02{0}.npy'.format(suffix))
        dR0_dlogL2 = np.load('cache/results_dR0_dlogL2{0}.npy'.format(suffix))
        dN_dVdt2 = np.load('cache/results_dN_dVdt2{0}.npy'.format(suffix))
        tildeL2 = np.load('cache/results_tildeL2{0}.npy'.format(suffix))
        tildeEp2 = np.load('cache/results_tildeEp2{0}.npy'.format(suffix))

L_min_mid = 5e49
R0_50 = np.nan_to_num(np.trapezoid(dR0_dlogL*(L>=L_min_mid),np.log(L),axis=1))

if chain2 is not None:
    R02_50 = np.nan_to_num(np.trapezoid(dR0_dlogL2*(L>=L_min_mid),np.log(L),axis=1))



R0_low,R0_med,R0_high = np.percentile(R0,[5.,50.,95.])
print('R_0=%.1f_{%.1f}^{%.1f} Gpc-3 yr-1'%(R0_med,R0_low-R0_med,R0_high-R0_med))

R0_50_low,R0_50_med,R0_50_high = np.percentile(R0_50,[5.,50.,95.])
print('R_0_50=%.1f_{%.1f}^{%.1f} Gpc-3 yr-1'%(R0_50_med,R0_50_low-R0_50_med,R0_50_high-R0_50_med))

if chain2 is not None:
    R02_low,R02_med,R02_high = np.percentile(R02,[5.,50.,95.])
    print('R_02=%.1f_{%.1f}^{%.1f} Gpc-3 yr-1'%(R02_med,R02_low-R02_med,R02_high-R02_med))
    
    R02_50_low,R02_50_med,R02_50_high = np.percentile(R02_50,[5.,50.,95.])
    print('R_02_50=%.1f_{%.1f}^{%.1f} Gpc-3 yr-1'%(R02_50_med,R02_50_low-R02_50_med,R02_50_high-R02_50_med))
    


############# Local rate ##############################################
figR0 = plt.figure('Local rate',figsize=(8.,3.5))

axR0lum = plt.subplot(122) 
axR0 = plt.subplot(121)

plt.title(r'$L_\mathrm{min}=10^{44}\,\mathrm{erg/s}$')


logR0_kde = gaussian_kde(np.log(R0))
logR0_50_kde = gaussian_kde(np.log(R0_50))

R00 = np.logspace(0.,5.,1000)
dP_dlogR0 = logR0_kde.pdf(np.log(R00))
dP_dlogR0_50 = logR0_50_kde.pdf(np.log(R00))

np.save('cache/curves_for_plots/dP_dlogR0/'+chain[:-5]+'.npy', (R00, dP_dlogR0))
np.save('cache/curves_for_plots/dP_dlogR0_50/'+chain[:-5]+'.npy', (R00, dP_dlogR0_50))

plt.plot(R00,dP_dlogR0,ls='-',color='r',lw=3,label=label_chain_1)

if chain2 is not None:
    logR02_kde = gaussian_kde(np.log(R02))
    logR02_50_kde = gaussian_kde(np.log(R02_50))
    
    dP_dlogR02 = logR02_kde.pdf(np.log(R00))
    dP_dlogR02_50 = logR02_50_kde.pdf(np.log(R00))
    
    np.save('cache/curves_for_plots/dP_dlogR0/'+chain2[:-5]+'.npy', (R00, dP_dlogR02))
    np.save('cache/curves_for_plots/dP_dlogR0_50/'+chain2[:-5]+'.npy', (R00, dP_dlogR02_50))
    
    plt.plot(R00,dP_dlogR02,ls='--',color='orange',lw=3,label=label_chain_2)

plt.semilogx()
plt.xlabel(r'$R_0\,\mathrm{[Gpc^{-3}\,yr^{-1}]}$')
plt.ylabel(r'$\mathrm{d}P/\mathrm{d}\ln(R_0/\mathrm{Gpc^{-3}\,yr^{-1}})$')

plt.ylim(0.,0.6)
plt.xlim(3,2e5)


plt.tick_params(which='both',direction='in',top=True,right=True)
plt.legend(frameon=False,markerfirst=False,loc='upper right',fontsize=9)


plt.fill_betweenx([0.,1.1],[10.,10.],[1700.,1700.],edgecolor='grey',facecolor='#EBEBEB',ls='--',zorder=-10)
plt.annotate(xy=((10*1700.)**0.5,0.45),text='BNS\n(GWTC-3)',ha='center',va='top',color='grey')

################ Luminosity function ##############################
plt.figure('Lum func')

dR0_dlogL_m = np.percentile(dR0_dlogL,50.,axis=0)
dR0_dlogL_u = np.percentile(dR0_dlogL,95.,axis=0)
dR0_dlogL_l = np.percentile(dR0_dlogL,5.,axis=0)

np.save('cache/curves_for_plots/dR_dlogL/'+chain[:-5]+'.npy',(L, dR0_dlogL_m, dR0_dlogL_u, dR0_dlogL_l))

plt.fill_between(L,dR0_dlogL_l,dR0_dlogL_u,edgecolor='r',facecolor='pink',alpha=0.5)
plt.plot(L,dR0_dlogL_m,'-r',lw=3,label=label_chain_1)

if chain2 is not None:
    dR0_dlogL2_m = np.percentile(dR0_dlogL2,50.,axis=0)
    dR0_dlogL2_u = np.percentile(dR0_dlogL2,95.,axis=0)
    dR0_dlogL2_l = np.percentile(dR0_dlogL2,5.,axis=0)
    
    np.save('cache/curves_for_plots/dR_dlogL/'+chain2[:-5]+'.npy',(L, dR0_dlogL2_m, dR0_dlogL2_u, dR0_dlogL2_l))
    
    plt.fill_between(L,dR0_dlogL2_l,dR0_dlogL2_u,edgecolor='orange',facecolor='#FFD8C0',alpha=0.5,zorder=-10,ls='--')
    plt.plot(L,dR0_dlogL2_m,ls='--',color='orange',lw=1.5,alpha=0.5,zorder=-10,label=label_chain_2)


# Plot the luminosity distribution of model (a) from Ghirlanda et al. 2016, for comparison

lgg16_corr = np.load('grb_data/l_gg16.npy')
dNdlogLgg16_corr = np.load('grb_data/dN_dlog10L_corr_gg16.npy')*np.log10(np.exp(1.))

R0_50_g16 = np.trapezoid(dNdlogLgg16_corr*(lgg16_corr>=L_min_mid),np.log(lgg16_corr),axis=1)

#dNdlogLgg16_corr/=np.log10(np.exp(1.))

## correle
plt.fill_between(lgg16_corr,np.percentile(dNdlogLgg16_corr,5.,axis=0),np.percentile(dNdlogLgg16_corr,90.,axis=0),facecolor='grey',edgecolor='k',zorder=-100,alpha=0.15)
plt.plot(lgg16_corr,np.percentile(dNdlogLgg16_corr,50.,axis=0),lw=1.5,color='grey',label='G16 (a)',zorder=-98)


# Plot the luminosity distribution from Wanderman and Piran 2015, for comparison

## these are the parameter values and their uncertainties as given in the paper
a1 = -0.94
da1l = 0.11
da1h = 0.13
a2 = -2.0
da2l = 1.
da2h = 0.7
logLb = np.log10(2e52)
dlogLbl = 0.4
dlogLbh = 1.3
logLmin = np.log10(5e49)
logL0 = np.linspace(logLmin,57.,100)

# we Monte Carlo over the parameters, assumed independent, to estimate the uncertainties
nt = 1000

dPdlogL = np.zeros([nt,len(logL0)])

for i in range(nt):
    a1i = np.random.normal(a1,da1h,1)[0]
    if a1i<a1:
        a1i = a1 - abs(np.random.normal(0.,da1l,1)[0]) 
    a2i = np.random.normal(a2,da2h,1)[0]
    if a2i<a2:
        a2i = a2 - abs(np.random.normal(0.,da2l,1)[0]) 
    logLbi = np.random.normal(logLb,dlogLbh,1)[0]
    if logLbi<logLb:
        logLbi = logLb - abs(np.random.normal(0.,dlogLbl,1)[0]) 
    dPdlogL[i] = 10**(a1i*(logL0-logLbi))
    dPdlogL[i][logL0>logLbi] = 10**(a2i*(logL0[logL0>logLbi]-logLbi))
    dPdlogL[i] *= np.random.normal(4.6,1.8,1)[0]/np.trapezoid(dPdlogL[i],logL0)*np.log10(np.exp(1.))


R0_50_wp15 = np.trapezoid(dPdlogL*(logL0>=50.),logL0/np.log10(np.exp(1.)),axis=1)

plt.fill_between(10**logL0,np.percentile(dPdlogL,5.,axis=0),np.percentile(dPdlogL,95.,axis=0),facecolor='#1E90FF',edgecolor='b',alpha=0.15)
plt.plot(10**logL0,np.percentile(dPdlogL,50.,axis=0),'-',color='b',lw=1.5,label='W15')

plt.tick_params(which='both',direction='in',top=True,right=True)

plt.ylabel(r'$R_0\phi(L)$ [Gpc$^{-3}$ yr$^{-1}$]')
plt.xlabel(r'$L\,[\mathrm{erg\,s^{-1}}]$')

plt.xlim(1e44,5e54)
plt.ylim(1e-4,1e4)

plt.loglog()

plt.legend(loc='upper right',frameon=False,fontsize=9)

plt.savefig('figures/Luminosity_function{0}.pdf'.format(suffix),bbox_inches='tight')

################## local rate R0 of luminous SGRBs ##############################################
plt.sca(axR0lum)

plt.title(r'$L_\mathrm{min}=10^{50}\,\mathrm{erg/s}$')


logR0_50_g16_kde = gaussian_kde(np.log(R0_50_g16[R0_50_g16>0.]))
logR0_50_wp15_kde = gaussian_kde(np.log(R0_50_wp15[R0_50_wp15>0.]))

plt.plot(R00,dP_dlogR0_50,ls='-',color='r',lw=3,label=label_chain_1)

if chain2 is not None:
    plt.plot(R00,dP_dlogR02_50,ls='--',color='orange',lw=3,label=label_chain_2)

plt.plot(R00,logR0_50_g16_kde.pdf(np.log(R00)),ls='-',color='grey',lw=2,label=r'G16 (a) ',zorder=-1)
plt.plot(R00,logR0_50_wp15_kde.pdf(np.log(R00)),ls='-',color='blue',lw=2,label=r'W15',zorder=-2)


plt.legend(frameon=False,markerfirst=False,loc='upper right',fontsize=9)

plt.semilogx()
plt.xlabel(r'$R_0\,\mathrm{[Gpc^{-3}\,yr^{-1}]}$')

plt.ylim(0.,1.2)
plt.xlim(0.07,50)

plt.tick_params(which='both',direction='in',top=True,right=True,labelleft=False,labelright=True)

#plt.annotate(xy=(30.,0.4),text=r'$L_\mathrm{min}=10^{50}\,\mathrm{erg/s}$',ha='center',va='center')

figR0.savefig('figures/Local_rate_density{0}.pdf'.format(suffix),bbox_inches='tight')

#################### redshift evolution ####################
plt.figure('Redshift rate evolution')
plt.title(r'$L_\mathrm{min}=10^{50}\,\mathrm{erg/s}$')

dN_dVdt *= (R0_50/R0).reshape([N,1])

dN_dVdt_m = np.percentile(dN_dVdt,50.,axis=0)
dN_dVdt_u = np.percentile(dN_dVdt,95.,axis=0)
dN_dVdt_l = np.percentile(dN_dVdt,5.,axis=0)

np.save('cache/curves_for_plots/dN_dVdt/'+chain[:-5]+'.npy',(z, dN_dVdt_m, dN_dVdt_u, dN_dVdt_l))

plt.fill_between(z,dN_dVdt_l,dN_dVdt_u,edgecolor='r',facecolor='pink',alpha=0.5)
plt.plot(z,dN_dVdt_m,'-r',lw=3,label=label_chain_1,zorder=3)

if chain2 is not None:
    dN_dVdt2 *= (R02_50/R02).reshape([N2,1])

    dN_dVdt2_m = np.percentile(dN_dVdt2,50.,axis=0)
    dN_dVdt2_u = np.percentile(dN_dVdt2,95.,axis=0)
    dN_dVdt2_l = np.percentile(dN_dVdt2,5.,axis=0)
    
    np.save('cache/curves_for_plots/dN_dVdt/'+chain2[:-5]+'.npy',(z, dN_dVdt2_m, dN_dVdt2_u, dN_dVdt2_l))
    
    plt.fill_between(z,dN_dVdt2_l,dN_dVdt2_u,edgecolor='orange',facecolor='#FFD8C0',alpha=0.5,ls='--')
    plt.plot(z,dN_dVdt2_m,ls='--',color='orange',lw=1.5,label=label_chain_2,zorder=10)

#plt.plot(1.+z,dN_dVdt.T[:,:100],color='grey',lw=0.5,alpha=0.5)

# ## G16
# p1,p2,p3 = np.loadtxt('grb_data/density_lmin_11SW.clean.txt',usecols=(9,10,11),unpack=True)
# R0_g16 = np.trapezoid(dNdlogLgg16_corr,np.log(lgg16_corr),axis=1)
# dN_dVdt_g16 = np.zeros([N,len(z)])
# for i in range(N):
#     dN_dVdt_g16[i] = R0_50_g16[i]*psi_g16(z,p1[i],p2[i],p3[i])

# plt.fill_between(z,np.percentile(dN_dVdt_g16,16.,axis=0),np.percentile(dN_dVdt_g16,84.,axis=0),edgecolor='k',facecolor='grey',alpha=0.2)
# plt.plot(z,np.percentile(dN_dVdt_g16,50.,axis=0),'-',lw=1.5,color='grey',label='G16')

########## OUT OF BOUND FOR N>1000
# R0_wp15 = np.random.normal(4.1,2.,N)
# dN_dVdt_wp15 = np.zeros([N,len(z)])
# for i in range(N):
#     dN_dVdt_wp15[i] = R0_50_wp15[i]*psi_wp15(z)


# plt.fill_between(z,np.percentile(dN_dVdt_wp15,16.,axis=0),np.percentile(dN_dVdt_wp15,84.,axis=0),edgecolor='b',facecolor='#1E90FF',alpha=0.2)
# plt.plot(z,np.percentile(dN_dVdt_wp15,50.,axis=0),'-',lw=1.5,color='b',label='W15')


plt.tick_params(which='both',direction='in',top=True,right=True)

plt.xlabel(r'$z$')
plt.ylabel(r'$\dot \rho(z,L>10^{50}\,\mathrm{erg\,s^{-1}})$ [Gpc$^{-3}$ yr$^{-1}$]')

plt.semilogy()

plt.xlim(0.,6.)
plt.ylim(1e-1,3e3)

plt.legend(frameon=False,loc='upper right',markerfirst=False,fontsize=9)

plt.savefig('figures/Rate_density_evolution{0}.pdf'.format(suffix),bbox_inches='tight')

print('\n Mission Passed')
print(' Respect +')

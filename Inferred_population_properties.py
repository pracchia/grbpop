import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from astropy.io import ascii
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import grbpop, os, pathlib
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

chain = 'chains/SGRB_full_Poisson_dtdsfh_lognorm.h5'
# chain = 'chains/SGRB_full_Poisson_dtdsfh_pow.h5'
# chain = 'chains/SGRB_full-sample-analysis_dtdsfh_log.h5'
# chain = 'chains/SGRB_full-sample-analysis_dtdsfh.h5'
# chain = 'chains/SGRB_flux-limited-sample-analysis_dtdsfh_log.h5'
# chain = 'chains/SGRB_flux-limited-sample-analysis_dtdsfh.h5'
# chain = 'chains/SGRB_full_sample_analysis_Poisson.h5'
# chain = 'chains/SGRB_GBM_plim_final_newsample.h5'


chain2 = 'chains/SGRB_flux-limited-sample-analysis_Poisson_WRONGCUT_dtdsfh_log_tight_boundaries.h5'
# chain2 = 'chains/SGRB_flux-limited-sample-analysis_Poisson_WRONGCUT_dtdsfh_pow_tight_boundaries.h5'
# chain2 = 'chains/SGRB_flux-limited-sample-analysis_dtdsfh_log.h5'
# chain2 = 'chains/SGRB_flux-limited-sample-analysis_dtdsfh.h5'
# chain2 = 'chains/SGRB_flux-limited-sample-analysis_WRONGCUT_dtdsfh_log_tight_boundaries.h5'
# chain2 = 'chains/SGRB_flux-limited-sample-analysis_WRONGCUT_dtdsfh_pow_tight_boundaries.h5'
# chain2 = 'chains/SGRB_flux-limited-sample-analysis_Poisson_WRONGCUT_tight_boundaries.h5'
# chain2 = 'chains/SGRB_flux-limited-sample-analysis_WRONGCUT_tight_boundaries.h5'
# chain2 = 'chains/SGRB_flux-limited-sample-analysis.h5'
specmodel = 'Comp'
alpha = -0.4
inst = 'Fermi'
pflim = 3.5
N = 10
res = 80
Robs = 212./0.59/10. # Fermi GBM SGRBs with p64>pflim, corrected for FoV and duty cycle

thin = 3

x,ll = read_chain(chain,burnin_fraction=0.5,thin=thin)

if chain2 is not None:
    x2,ll2 = read_chain(chain2,burnin_fraction=0.5,thin=thin)
    # x2,ll2 = read_chain(chain2,burnin_fraction=0.25,thin=thin)
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
            # theta_pop = {'jetmodel':'smooth double power law',
            #  'rho_z':'SBPL',
            #  'thc':10**x[i,0],
            #  'Lc*':10**x[i,1],
            #  'a_L':x[i,2],
            #  'b_L':x[i,3],
            #  'Epc*':10**x[i,4],
            #  'a_Ep':x[i,5],
            #  'b_Ep':x[i,6],
            #  'thw':10**x[i,7],
            #  'A':x[i,8],
            #  's_c':10**x[i,9],
            #  'y':x[i,10],
            #  'a':x[i,11],
            #  'b':x[i,12],
            #  'zp':x[i,13],
            #  'R0':10**x[i,14]
            #  }
            theta_pop = {'jetmodel':'smooth double power law',
             'rho_z':'DTD*SFH',
             'dtd':'lognorm',
             'thc':10**x[i,0],
             'Lc*':10**x[i,1],
             'a_L':x[i,2],
             'b_L':x[i,3],
             'Epc*':10**x[i,4],
             'a_Ep':x[i,5],
             'b_Ep':x[i,6],
             'thw':10**x[i,7],
             'A':x[i,8],
             's_c':10**x[i,9],
             'y':x[i,10],
             'mu_td':x[i,11],
             'sigma_td':x[i,12],
             'R0':10**x[i,13]
             }
             # 'dtd':'lognorm',
             # 'mu_td':x[i,11],
             # 'sigma_td':x[i,12]
             # 'dtd':'pow',
             # 'tdmin':x[i,11],
             # 'at':x[i,12]

            if chain2 is not None:
                # theta_pop2 = {'jetmodel':'smooth double power law',
                #  'rho_z':'SBPL',
                #  'thc':10**x2[i,0],
                #  'Lc*':10**x2[i,1],
                #  'a_L':x2[i,2],
                #  'b_L':x2[i,3],
                #  'Epc*':10**x2[i,4],
                #  'a_Ep':x2[i,5],
                #  'b_Ep':x2[i,6],
                #  'thw':10**x2[i,7],
                #  'A':x2[i,8],
                #  's_c':10**x2[i,9],
                #  'y':x2[i,10],
                #  'a':x2[i,11],
                #  'b':x2[i,12],
                #  'zp':x2[i,13],
                #  'R0':10**x2[i,14]
                # }
                theta_pop2 = {'jetmodel':'smooth double power law',
                 'rho_z':'DTD*SFH',
                 'dtd':'lognorm',
                 'thc':10**x2[i,0],
                 'Lc*':10**x2[i,1],
                 'a_L':x2[i,2],
                 'b_L':x2[i,3],
                 'Epc*':10**x2[i,4],
                 'a_Ep':x2[i,5],
                 'b_Ep':x2[i,6],
                 'thw':10**x2[i,7],
                 'A':x2[i,8],
                 's_c':10**x2[i,9],
                 'y':x2[i,10],
                 'mu_td':x2[i,11],
                 'sigma_td':x2[i,12],
                 'R0':10**x2[i,13]
                }
                 # 'dtd':'lognorm',
                 # 'mu_td':x2[i,11],
                 # 'sigma_td':x2[i,12]
                 # 'dtd':'pow',
                 # 'tdmin':x2[i,11],
                 # 'at':x2[i,12]
        
        else:
            # theta_pop = {'jetmodel':'smooth double power law',
            #      'rho_z':'SBPL',
            #      'thc':10**x[i,0],
            #      'Lc*':10**x[i,1],
            #      'a_L':x[i,2],
            #      'b_L':x[i,3],
            #      'Epc*':10**x[i,4],
            #      'a_Ep':x[i,5],
            #      'b_Ep':x[i,6],
            #      'thw':10**x[i,7],
            #      'A':x[i,8],
            #      's_c':10**x[i,9],
            #      'y':x[i,10],
            #      'a':x[i,11],
            #      'b':x[i,12],
            #      'zp':x[i,13]
            #      }
            theta_pop = {'jetmodel':'smooth double power law',
                 'rho_z':'DTD*SFH',
                 'dtd':'lognorm',
                 'thc':10**x[i,0],
                 'Lc*':10**x[i,1],
                 'a_L':x[i,2],
                 'b_L':x[i,3],
                 'Epc*':10**x[i,4],
                 'a_Ep':x[i,5],
                 'b_Ep':x[i,6],
                 'thw':10**x[i,7],
                 'A':x[i,8],
                 's_c':10**x[i,9],
                 'y':x[i,10],
                 'mu_td':x[i,11],
                 'sigma_td':x[i,12]
                 }
                 # 'dtd':'lognorm',
                 # 'mu_td':x[i,11],
                 # 'sigma_td':x[i,12]
                 # 'dtd':'pow',
                 # 'tdmin':x[i,11],
                 # 'at':x[i,12]
            
            if chain2 is not None:
                # theta_pop2 = {'jetmodel':'smooth double power law',
                #  'rho_z':'SBPL',
                #  'thc':10**x2[i,0],
                #  'Lc*':10**x2[i,1],
                #  'a_L':x2[i,2],
                #  'b_L':x2[i,3],
                #  'Epc*':10**x2[i,4],
                #  'a_Ep':x2[i,5],
                #  'b_Ep':x2[i,6],
                #  'thw':10**x2[i,7],
                #  'A':x2[i,8],
                #  's_c':10**x2[i,9],
                #  'y':x2[i,10],
                #  'a':x2[i,11],
                #  'b':x2[i,12],
                #  'zp':x2[i,13]
                # }
                theta_pop2 = {'jetmodel':'smooth double power law',
                 'rho_z':'DTD*SFH',
                 'dtd':'lognorm',
                 'thc':10**x2[i,0],
                 'Lc*':10**x2[i,1],
                 'a_L':x2[i,2],
                 'b_L':x2[i,3],
                 'Epc*':10**x2[i,4],
                 'a_Ep':x2[i,5],
                 'b_Ep':x2[i,6],
                 'thw':10**x2[i,7],
                 'A':x2[i,8],
                 's_c':10**x2[i,9],
                 'y':x2[i,10],
                 'mu_td':x2[i,11],
                 'sigma_td':x2[i,12]
                }
                 # 'dtd':'lognorm',
                 # 'mu_td':x2[i,11],
                 # 'sigma_td':x2[i,12]
                 # 'dtd':'pow',
                 # 'tdmin':x2[i,11],
                 # 'at':x2[i,12]
        
        PEpL = grbpop.Ppop.PEpL(L,Ep,theta_pop,grid=True)
        PEpL/=np.trapz(np.trapz(PEpL*Epg[:,:,0]*Lg[:,:,0],np.log(L),axis=1),np.log(Ep))
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
            R0[i] = Robs/np.trapz(np.trapz(np.trapz(PEpL.reshape([len(Ep),len(L),1])*psiz.reshape([1,1,len(z)])*Epg*Lg*zg*Pdet,np.log(z),axis=2),np.log(L),axis=1),np.log(Ep))
        R0[i] = np.nan_to_num(R0[i])
        dR0_dlogL[i] = L*R0[i]*grbpop.diagnose.luminosity_function(L,theta_pop)
        dN_dVdt[i] = R0[i]*psiz*(1.+z)/dVdz
        tildeL[i] = theta_pop['Lc*']*grbpop.structjet.ell(th,theta_pop)
        tildeEp[i] = theta_pop['Epc*']*grbpop.structjet.eta(th,theta_pop)

        if chain2 is not None:
            PEpL2 = grbpop.Ppop.PEpL(L,Ep,theta_pop2,grid=True)
            PEpL2/=np.trapz(np.trapz(PEpL2*Epg[:,:,0]*Lg[:,:,0],np.log(L),axis=1),np.log(Ep))
            if (theta_pop2['rho_z']=='SBPL'): 
                rhoz2 = grbpop.Ppop.MD14_SFH(z,theta_pop2['a'],theta_pop2['b'],theta_pop2['zp'])
                rhoz2/=rhoz2[0]
            elif (theta_pop2['rho_z']=='DTD*SFH' and theta_pop['dtd']=='pow'):
                rhoz2 = Itp_rhoz_pow((np.log10(z),theta_pop2['tdmin'],theta_pop2['at']))
                rhoz2/=rhoz2[0]
            elif (theta_pop2['rho_z']=='DTD*SFH' and theta_pop['dtd']=='lognorm'):
                rhoz2 = Itp_rhoz_log((np.log10(z),theta_pop2['mu_td'],theta_pop2['sigma_td']))
                rhoz2/=rhoz2[0]
            
            psiz2 = rhoz2/(1.+z)*dVdz
            if Poisson:
                R02[i] = theta_pop2 ['R0']
            else: 
                R02[i] = Robs/np.trapz(np.trapz(np.trapz(PEpL2.reshape([len(Ep),len(L),1])*psiz2.reshape([1,1,len(z)])*Epg*Lg*zg*Pdet,np.log(z),axis=2),np.log(L),axis=1),np.log(Ep))
            R02[i] = np.nan_to_num(R02[i])
            dR0_dlogL2[i] = L*R02[i]*grbpop.diagnose.luminosity_function(L,theta_pop2)
            dN_dVdt2[i] = R02[i]*psiz2*(1.+z)/dVdz
            tildeL2[i] = theta_pop2['Lc*']*grbpop.structjet.ell(th,theta_pop2)
            tildeEp2[i] = theta_pop2['Epc*']*grbpop.structjet.eta(th,theta_pop2)
    
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


R0_50 = np.nan_to_num(np.trapz(dR0_dlogL*(L>=1e50),np.log(L),axis=1))

if chain2 is not None:
    R02_50 = np.nan_to_num(np.trapz(dR0_dlogL2*(L>=1e50),np.log(L),axis=1))



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

R00 = np.logspace(-2.,5.4,1000)
dP_dlogR0 = logR0_kde.pdf(np.log(R00))
dP_dlogR0_50 = logR0_50_kde.pdf(np.log(R00))

plt.plot(R00,dP_dlogR0,ls='-',color='r',lw=3,label='Completeness')
# plt.plot(R00,dP_dlogR0,ls='-',color='r',lw=3,label='Full')

if chain2 is not None:
    logR02_kde = gaussian_kde(np.log(R02))
    logR02_50_kde = gaussian_kde(np.log(R02_50))
    
    dP_dlogR02 = logR02_kde.pdf(np.log(R00))
    dP_dlogR02_50 = logR02_50_kde.pdf(np.log(R00))
    
    plt.plot(R00,dP_dlogR02,ls='--',color='orange',lw=3,label='Incorrect selection effects')
    # plt.plot(R00,dP_dlogR02,ls='--',color='orange',lw=3,label='Flux-limited')

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

plt.fill_between(L,np.percentile(dR0_dlogL,5.,axis=0),np.percentile(dR0_dlogL,95.,axis=0),edgecolor='r',facecolor='pink',alpha=0.5)
plt.plot(L,np.percentile(dR0_dlogL,50.,axis=0),'-r',lw=3,label='Completeness')
# plt.plot(L,np.percentile(dR0_dlogL,50.,axis=0),'-r',lw=3,label='Full')

if chain2 is not None:
    plt.fill_between(L,np.percentile(dR0_dlogL2,5.,axis=0),np.percentile(dR0_dlogL2,90.,axis=0),edgecolor='orange',facecolor='#FFD8C0',alpha=0.5,zorder=-10,ls='--')
    plt.plot(L,np.percentile(dR0_dlogL2,50.,axis=0),ls='--',color='orange',lw=1.5,alpha=0.5,zorder=-10,label='Incorrect selection effects')
    # plt.plot(L,np.percentile(dR0_dlogL2,50.,axis=0),ls='--',color='orange',lw=1.5,alpha=0.5,zorder=-10,label='Flux-limited')


# Plot the luminosity distribution of model (a) from Ghirlanda et al. 2016, for comparison

lgg16_corr = np.load('grb_data/l_gg16.npy')
dNdlogLgg16_corr = np.load('grb_data/dN_dlog10L_corr_gg16.npy')*np.log10(np.exp(1.))

R0_50_g16 = np.trapz(dNdlogLgg16_corr*(lgg16_corr>=1e50),np.log(lgg16_corr),axis=1)

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
    dPdlogL[i] *= np.random.normal(4.6,1.8,1)[0]/np.trapz(dPdlogL[i],logL0)*np.log10(np.exp(1.))


R0_50_wp15 = np.trapz(dPdlogL*(logL0>=50.),logL0/np.log10(np.exp(1.)),axis=1)

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

plt.plot(R00,dP_dlogR0_50,ls='-',color='r',lw=3,label='Completeness')
# plt.plot(R00,dP_dlogR0_50,ls='-',color='r',lw=3,label='Full')

if chain2 is not None:
    plt.plot(R00,dP_dlogR02_50,ls='--',color='orange',lw=3,label='Incorrect selection effects')
    # plt.plot(R00,dP_dlogR02_50,ls='--',color='orange',lw=3,label='Flux-limited')

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

plt.fill_between(z,np.percentile(dN_dVdt,5.,axis=0),np.percentile(dN_dVdt,95.,axis=0),edgecolor='r',facecolor='pink',alpha=0.5)
plt.plot(z,np.percentile(dN_dVdt,50.,axis=0),'-r',lw=3,label='Completeness',zorder=3)
# plt.plot(z,np.percentile(dN_dVdt,50.,axis=0),'-r',lw=3,label='Full',zorder=3)

if chain2 is not None:
    dN_dVdt2 *= (R02_50/R02).reshape([N2,1])
    
    plt.fill_between(z,np.percentile(dN_dVdt2,16.,axis=0),np.percentile(dN_dVdt2,84.,axis=0),edgecolor='orange',facecolor='#FFD8C0',alpha=0.5,ls='--')
    plt.plot(z,np.percentile(dN_dVdt2,50.,axis=0),ls='--',color='orange',lw=1.5,label='Incorrect selection effects',zorder=10)
    # plt.plot(z,np.percentile(dN_dVdt2,50.,axis=0),ls='--',color='orange',lw=1.5,label='Flux-limited',zorder=10)

#plt.plot(1.+z,dN_dVdt.T[:,:100],color='grey',lw=0.5,alpha=0.5)

## G16
p1,p2,p3 = np.loadtxt('grb_data/density_lmin_11SW.clean.txt',usecols=(9,10,11),unpack=True)
R0_g16 = np.trapz(dNdlogLgg16_corr,np.log(lgg16_corr),axis=1)
dN_dVdt_g16 = np.zeros([N,len(z)])
for i in range(N):
    dN_dVdt_g16[i] = R0_50_g16[i]*psi_g16(z,p1[i],p2[i],p3[i])

plt.fill_between(z,np.percentile(dN_dVdt_g16,16.,axis=0),np.percentile(dN_dVdt_g16,84.,axis=0),edgecolor='k',facecolor='grey',alpha=0.2)
plt.plot(z,np.percentile(dN_dVdt_g16,50.,axis=0),'-',lw=1.5,color='grey',label='G16')

R0_wp15 = np.random.normal(4.1,2.,N)
dN_dVdt_wp15 = np.zeros([N,len(z)])
for i in range(N):
    dN_dVdt_wp15[i] = R0_50_wp15[i]*psi_wp15(z)


plt.fill_between(z,np.percentile(dN_dVdt_wp15,16.,axis=0),np.percentile(dN_dVdt_wp15,84.,axis=0),edgecolor='b',facecolor='#1E90FF',alpha=0.2)
plt.plot(z,np.percentile(dN_dVdt_wp15,50.,axis=0),'-',lw=1.5,color='b',label='W15')


plt.tick_params(which='both',direction='in',top=True,right=True)

plt.xlabel(r'$z$')
plt.ylabel(r'$\dot \rho(z,L>10^{50}\,\mathrm{erg\,s^{-1}})$ [Gpc$^{-3}$ yr$^{-1}$]')

plt.semilogy()

plt.xlim(0.,6.)
plt.ylim(1e-1,3e3)

plt.legend(frameon=False,loc='upper right',markerfirst=False,fontsize=9)

plt.savefig('figures/Rate_density_evolution{0}.pdf'.format(suffix),bbox_inches='tight')


########################### jet structure #######################
fig,(ax0,ax1,ax2) = plt.subplots(3,1,gridspec_kw={'height_ratios':[0.2,1,1]},figsize=(4,7),label='Jet structure',tight_layout=False)
plt.subplots_adjust(hspace=0.02,top=0.95,left=0.14,right=0.96)


### transition angles ###
plt.sca(ax0)

thc = 10**x[:,0]
thw = 10**x[:,7]

plogthc = gaussian_kde(np.log(thc/np.pi*180.))
plogthw = gaussian_kde(np.log(thw/np.pi*180.))

thv0 = np.logspace(np.log10(0.6),np.log10(90.),100)

pthc = plogthc.pdf(np.log(thv0))
pthw = plogthw.pdf(np.log(thv0))

plt.fill_between(thv0,pthc/pthc.max(),facecolor='grey',edgecolor='k',alpha=0.4)
plt.fill_between(thv0,pthw/pthw.max(),facecolor='green',edgecolor='grey',alpha=0.4)

plt.annotate(xy=(0.7,1.),text=r'$\theta_\mathrm{c}$',ha='center',va='top',color='grey')
plt.annotate(xy=(30.,1.),text=r'$\theta_\mathrm{w}$',ha='center',va='top',color='green')

plt.tick_params(which='both',direction='in',labelbottom=False,labeltop=True,top=True,left=False,right=False,labelleft=False)
plt.xlim(0.6,90.)
plt.ylim(0.,1.1)
plt.semilogx()

if chain2 is not None:
    thc2 = 10**x2[:,0]
    thw2 = 10**x2[:,7]
    
    plogthc = gaussian_kde(np.log(thc2/np.pi*180))
    plogthw = gaussian_kde(np.log(thw2/np.pi*180.))
    
    pthc = plogthc.pdf(np.log(thv0))
    pthw = plogthw.pdf(np.log(thv0))
    
    plt.fill_between(thv0,pthc/pthc.max(),facecolor='None',edgecolor='k',alpha=0.4,ls='--')
    plt.fill_between(thv0,pthw/pthw.max(),facecolor='None',edgecolor='green',alpha=0.4,ls='--')



# Core luminosity slope and Epeak dispersion
A = x[:,8]
sigma_c_dex = 10**x[:,9]/np.log(10.)
y = x[:,10]

if chain2 is not None:
    A2 = x2[:,8]
    sigma_c_dex2 = 10**x2[:,9]/np.log(10.)
    y2 = x2[:,10]
    

# load GW170817 posterior samples
gwsamples = np.genfromtxt('grb_data/high_spin_PhenomPNRT_posterior_samples.dat',names=True)

# NGC4993 distance from Cantiello et al. 2018
dL17 = 40.7
dL17_err = (1.4**2+1.9**2)**0.5

#GRB 170817A Liso and Ep
Lsamples = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_L_samples.npy')[-1]
Epsamples = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_Ep_samples.npy')[-1]

# costheta_jn to theta_view
thv = np.arccos(gwsamples['costheta_jn'])
thv[thv>(np.pi/2.)]=np.pi-thv[thv>(np.pi/2.)]


# dL
dL17s = gwsamples['luminosity_distance_Mpc']
w = np.exp(-0.5*((dL17-dL17s)/dL17_err)**2)

rng = np.random.default_rng()
thvs = rng.choice(thv,len(Lsamples),p=w/np.sum(w))

d2L,dthvL,dlogL = samples_to_mesh(thvs,np.log10(Lsamples),bins=(60,51),smooth=1.)
d2Ep,dthvEp,dlogEp = samples_to_mesh(thvs,np.log10(Epsamples),bins=(40,31),smooth=1.)
dEp_dL,dlogLL,dlogEpEp = samples_to_mesh(np.log10(Lsamples),np.log10(Epsamples),bins=(51,31),smooth=1.)


##### ell ####
plt.sca(ax1)

# trends
plt.plot(th/np.pi*180.,np.where((th>0.05)&(th<0.25),1.,np.nan)*3e49*(th/0.25)**-4.7,ls=':',color='grey')
plt.annotate(xy=(8,4e50),text=r'$\theta_\mathrm{v}^{-4.7}$',ha='left',va='bottom',color='grey')


# struct
plt.fill_between(th/np.pi*180.,np.percentile(tildeL,16.,axis=0),np.percentile(tildeL,85.,axis=0),edgecolor='pink',facecolor='#FFB7E1',alpha=0.5,label='90% credible region')
plt.plot(th/np.pi*180.,np.percentile(tildeL,50.,axis=0),'-r',label=r'Completeness')
# plt.plot(th/np.pi*180.,np.percentile(tildeL,50.,axis=0),'-r',label=r'Full')

if chain2 is not None:
    plt.fill_between(th/np.pi*180.,np.percentile(tildeL2,16.,axis=0),np.percentile(tildeL2,85.,axis=0),edgecolor='#FFDFB7',facecolor='#FFD8C0',alpha=0.5,ls='--',zorder=-2)
    plt.plot(th/np.pi*180.,np.percentile(tildeL2,50.,axis=0),ls='--',color='orange',zorder=-1,label='Incorrect selection effects')
    # plt.plot(th/np.pi*180.,np.percentile(tildeL2,50.,axis=0),ls='--',color='orange',zorder=-1,label='Flux-limited')
    

plt.contour(dthvL/np.pi*180.,10**dlogL,d2L,levels=[0.5,0.9],colors=['#11FF00','#11FF00'],linestyles=['-','-'],zorder=4,alpha=1.)
plt.ylim([1e45,1e53])

plt.annotate(xy=(10,1e46),text='GRB170817A',color='#11FF00',ha='center',va='center')

plt.ylabel('$L_\mathrm{c}^\star\ell\,[\mathrm{erg/s}]$')
plt.xlim(0.6,90.)
plt.loglog()

plt.legend(loc='upper right',frameon=False,markerfirst=False,fontsize=9)
plt.tick_params(which='both',direction='in',top=True,right=True,labelbottom=False)


# inset with posterior on A
inset_sigmaL = plt.axes([0.18,0.55,0.2,0.12])

Akde = gaussian_kde(A)
A0 = np.linspace(2.,4.,100)

inset_sigmaL.fill_between(A0,Akde.pdf(A0),edgecolor='r',facecolor='#FFB7E1',alpha=0.5)

if chain2 is not None:
    Akde = gaussian_kde(A2)
    
    inset_sigmaL.fill_between(A0,Akde.pdf(A0),edgecolor='#FFCA87',facecolor='#FFD8C0',zorder=-1,ls='--')

plt.xlim([2.,4.])
plt.xticks([2.,3.,4.])
plt.xlabel(r'$A$')


plt.tick_params(which='both',direction='in',top=True,right=False,left=False,labelleft=False)



################## eta ##########
plt.sca(ax2)

# trends
plt.plot(th/np.pi*180.,np.where((th>0.05)&(th<0.4),1.,np.nan)*0.7e3*(th/0.4)**-2,ls=':',color='grey')
plt.annotate(xy=(10.,4e3),text=r'$\theta_\mathrm{v}^{-2}$',ha='left',va='bottom',color='grey')

plt.fill_between(th/np.pi*180.,np.percentile(tildeEp,16.,axis=0),np.percentile(tildeEp,84.,axis=0),edgecolor='#54ABFF',facecolor='#54ABFF',alpha=0.5)
plt.plot(th/np.pi*180.,np.percentile(tildeEp,50.,axis=0),'-b',label=r'Completeness')
# plt.plot(th/np.pi*180.,np.percentile(tildeEp,50.,axis=0),'-b',label=r'Full')

if chain2 is not None:
    plt.fill_between(th/np.pi*180.,np.percentile(tildeEp2,16.,axis=0),np.percentile(tildeEp2,84.,axis=0),edgecolor='#26E9E9',facecolor='#B6FFFF',alpha=0.5,zorder=-2,ls='--')
    plt.plot(th/np.pi*180.,np.percentile(tildeEp2,50.,axis=0),ls='--',color='cyan',label=r'Incorrect selection effects',zorder=-1)
    # plt.plot(th/np.pi*180.,np.percentile(tildeEp2,50.,axis=0),ls='--',color='cyan',label=r'Flux-limited',zorder=-1)

plt.contour(dthvEp/np.pi*180.,10**dlogEp,d2Ep,levels=[0.5,0.9],colors=['#11FF00','#11FF00'],linestyles=['-','-'],zorder=-1,alpha=0.8)

plt.ylim([1e0,1e5])
plt.xlim(0.6,90.)
plt.loglog()

plt.ylabel('$E_\mathrm{p,c}^\star\,\eta\,[\mathrm{keV}]$')

if chain2 is not None:
    plt.legend(loc='upper right',frameon=False,markerfirst=False,fontsize=9)

plt.tick_params(which='both',direction='in',top=True,right=True)

plt.xlabel(r'$\theta_\mathrm{v}\,[\mathrm{deg}]$')


# inset with dispersion
inset_sigmaEp = plt.axes([0.18,0.18,0.2,0.12])

sEpkde = gaussian_kde(sigma_c_dex)
sEp0 = np.linspace(0.1,0.7,100)

inset_sigmaEp.fill_between(sEp0,sEpkde.pdf(sEp0),edgecolor='b',facecolor='#54ABFF',alpha=0.5)

if chain2 is not None:
    sEpkde = gaussian_kde(sigma_c_dex2)
    
    inset_sigmaEp.fill_between(sEp0,sEpkde.pdf(sEp0),edgecolor='#26E9E9',facecolor='#B6FFFF',zorder=-1,ls='--')
    

plt.xlim([0.2,0.7])
plt.xlabel(r'$\sigma_\mathrm{c}$ [dex]')
plt.xticks([0.2,0.4,0.6])
plt.tick_params(which='both',direction='in',top=True,right=False,left=False,labelleft=False)

plt.savefig('figures/Average_structure{0}.pdf'.format(suffix),bbox_inches='tight')


#################### Yonetoku ##################################

plt.figure('Yonetoku')

h,binL,binEp = np.histogram2d(tildeL.ravel(),tildeEp.ravel(),bins=(L,Ep))

Y = np.zeros(h.shape)

for i in range(len(binL)-1):
    h[i] = gaussian_filter(h[i],0.8)
    s = np.argsort(h[i])
    c = np.zeros(h.shape[1])
    c[s] = np.cumsum(h[i,s])
    if np.any(c>0.):
        c/=c.max()
    Y[i,c>0.05]=1.2

Y = gaussian_filter(Y,0.5)

plt.contourf((L[1:]*L[:-1])**0.5,(Ep[1:]*Ep[:-1])**0.5,Y.T,levels=[1.,2.],colors=['cyan'],alpha=0.4,linestyles=['-'])


plt.fill_between([0.],[0.],[0.],edgecolor='teal',facecolor='cyan',label='90% credible region',alpha=0.4,ls='-')
plt.plot(np.percentile(tildeL,50.,axis=0),np.percentile(tildeEp,50.,axis=0),color='teal',label='Completeness')
# plt.plot(np.percentile(tildeL,50.,axis=0),np.percentile(tildeEp,50.,axis=0),color='teal',label='Full')

for i in range(min(100,N)):
    plt.plot(tildeL[i],tildeEp[i],color='grey',lw=0.2,alpha=0.1)

if chain2 is not None:
    h2,binL,binEp = np.histogram2d(tildeL2.ravel(),tildeEp2.ravel(),bins=(L,Ep))

    Y2 = np.zeros(h2.shape)

    for i in range(len(binL)-1):
        h2[i] = gaussian_filter(h2[i],0.8)
        s = np.argsort(h2[i])
        c = np.zeros(h2.shape[1])
        c[s] = np.cumsum(h2[i,s])
        if np.any(c>0.):
            c/=c.max()
        Y2[i,c>0.05]=1.2
    
    Y2 = gaussian_filter(Y2,0.5)
    
    plt.contourf((L[1:]*L[:-1])**0.5,(Ep[1:]*Ep[:-1])**0.5,Y2.T,levels=[1.,1.2],colors=['#FDFF00'],alpha=0.4,linestyles=['-'],zorder=-2)

    
    plt.plot(np.percentile(tildeL2,50.,axis=0),np.percentile(tildeEp2,50.,axis=0),color='#79D900',ls='--',label='Incorrect selection effects',zorder=10)
    # plt.plot(np.percentile(tildeL2,50.,axis=0),np.percentile(tildeEp2,50.,axis=0),color='#79D900',ls='--',label='Flux-limited',zorder=10)


# show obs GRBs
Lsamples = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_L_samples.npy')
Epsamples = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_Ep_samples.npy')

SGRBs = {'Liso':np.median(Lsamples[:-1],axis=1),
         'Liso_err_low':np.median(Lsamples[:-1],axis=1)-np.percentile(Lsamples[:-1],5.,axis=1),
         'Liso_err_high':-np.median(Lsamples[:-1],axis=1)+np.percentile(Lsamples[:-1],95.,axis=1),
         'Ep':np.median(Epsamples[:-1],axis=1),
         'Ep_err_low':np.median(Epsamples[:-1],axis=1)-np.percentile(Epsamples[:-1],5.,axis=1),
         'Ep_err_high':-np.median(Epsamples[:-1],axis=1)+np.percentile(Epsamples[:-1],95.,axis=1),
         }


plt.errorbar(SGRBs['Liso'],SGRBs['Ep'],xerr=[SGRBs['Liso_err_low'],SGRBs['Liso_err_high']],yerr=[SGRBs['Ep_err_low'],SGRBs['Ep_err_high']],ls='None',color='r',lw=0.8,label='Fermi/GBM short GRBs',zorder=3,capsize=0.)


# GRB 170817A
Liso_GRB170817A = np.median(Lsamples[-1])   #1.6e47 #https://iopscience.iop.org/article/10.3847/2041-8213/aa920c/pdf
dLiso_GRB170817A_low = -np.percentile(Lsamples[-1],5.)+np.median(Lsamples[-1])
dLiso_GRB170817A_high = np.percentile(Lsamples[-1],95.)-np.median(Lsamples[-1])
Ep_GRB170817A = np.mean(Epsamples[-1])  #185.
dEp_GRB170817A_low = -np.percentile(Epsamples[-1],5.)+np.median(Epsamples[-1]) #62.
dEp_GRB170817A_high = np.percentile(Epsamples[-1],95.)-np.median(Epsamples[-1]) #62.
thv_GRB170817A = 15.
dthv_GRB170817A_low = 1.
dthv_GRB170817A_high = 5.5

plt.errorbar([Liso_GRB170817A],[Ep_GRB170817A],yerr=[[dEp_GRB170817A_low],[dEp_GRB170817A_high]],xerr=[[dLiso_GRB170817A_low],[dLiso_GRB170817A_high]],marker='o',markersize=1,color='b',zorder=3,capsize=0.,ls='None',elinewidth=0.5,label='GRB 170817A')

plt.loglog()

plt.xlim([1e44,1e54])
plt.ylim([3e0,3e4])

plt.xlabel(r'$L$ [erg/s]')
plt.ylabel(r'$E_\mathrm{p}$ [keV]')

plt.tick_params(which='both',direction='in',top=True,right=True)
plt.legend(frameon=False,markerfirst=True,loc='upper left',fontsize=9)


inset = plt.axes([0.7,0.22,0.16,0.16])

inset.set_xlabel(r'y',fontsize=8)

ykde = gaussian_kde(x[:,10])
y0 = np.linspace(-1,1,100)

inset.fill_between(y0,ykde.pdf(y0),edgecolor='teal',facecolor='cyan',alpha=0.4)

if chain2 is not None:
    ykde = gaussian_kde(x2[:,10])
    
    inset.fill_between(y0,ykde.pdf(y0),edgecolor='#79D900',facecolor='#FDFF00',ls='--',alpha=0.4)
        

plt.xlim(-1,1)
plt.ylim(0.,3.3)
plt.tick_params(which='both',direction='in',top=True,left=False,labelleft=False,labelsize=8)

plt.savefig('figures/Yonetoku_plane{0}.pdf'.format(suffix),bbox_inches='tight')

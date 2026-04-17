import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from astropy.io import ascii
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import grbpop, os, pathlib
from dynesty import DynamicNestedSampler
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


recompute = True
# recompute = False

suffix = '' 

folder = 'nested_samplings/'
# chain = 'wp15_new_Dynesty_Poisson_SGRB_fit_POW_reduced.001.save'
chain = 'wp15_new_Dynesty_Poisson_SGRB_fit_POW_extended.001.save'
label_chain_1 = 'Pow'
dsampler1 = DynamicNestedSampler.restore(folder + chain)

folder2 = 'nested_samplings/'
# chain2 = None
# chain2 = 'wp15_new_Dynesty_Poisson_SGRB_fit_LOG_reduced.001.save'
chain2 = 'wp15_new_Dynesty_Poisson_SGRB_fit_LOG_extended.001.save'
label_chain_2 = 'Log'
dsampler2 = DynamicNestedSampler.restore(folder2 + chain2)


specmodel = 'Band'
alpha = -0.5
inst = 'Fermi'
pflim = 2.37
# pflim = 1.5
res = 80
# Robs = 146./3.65 # Fermi GBM SGRBs with p64>pflim with numbers considered by WP15
# Robs = 414./4.44 # CGRO BATSE SGRBs with p64>pflim with numbers considered by WP15
# Robs = (414. + 146.)/(4.44 + 3.65) # Fermi GBM SGRBs with p64>pflim up to date considered for WP15, corrected for FoV and duty cycle
# Robs = 212./0.59/10. # Fermi GBM SGRBs with p64>pflim, corrected for FoV and duty cycle
Robs = 146./0.59/4.75 # Fermi GBM SGRBs with p64>pflim up to date considered for WP15, corrected for FoV and duty cycle
# Robs = 146./4.75 # Fermi GBM SGRBs with p64>pflim up to date considered for WP15, corrected for FoV and duty cycle
# Robs = (414. + 146.)/(4.44 + 3.65) # Fermi GBM SGRBs with p64>pflim up to date considered for WP15, corrected for FoV and duty cycle
# Robs = (341. + 146.)/(4.44 + 3.65) # Fermi GBM SGRBs with p64>pflim up to date considered for WP15, corrected for FoV and duty cycle
# Robs = (146.)/(3.65) # Fermi GBM SGRBs with p64>pflim up to date considered for WP15, corrected for FoV and duty cycle
# Robs = (341.)/(4.44) # Fermi GBM SGRBs with p64>pflim up to date considered for WP15, corrected for FoV and duty cycle

x = dsampler1.results.samples_equal()

if chain2 is not None:
    x2 = dsampler2.results.samples_equal()

thin = 1
# thin = 1

# N = 10000
N = int(np.min([len(x),len(x2)])/2)
N2 = N

th = np.logspace(grbpop.globals.logthvmin,np.log10(np.pi/2.),300)
L = np.logspace(grbpop.globals.logLmin-2,grbpop.globals.logLmax,res+1)
# Ep = np.logspace(grbpop.globals.logEpmin,grbpop.globals.logEpmax,res)
Ep = 800. # keV
z = np.logspace(grbpop.globals.logzmin,grbpop.globals.logzmax,res-1)
dVdz = 4*np.pi*cosmo.differential_comoving_volume(z).to('Gpc3 sr-1').value

# zg = z.reshape([1,1,len(z)])
# Epg = Ep.reshape([len(Ep),1,1])
# Lg = L.reshape([1,len(L),1])
zg = z.reshape([1,len(z)])
Lg = L.reshape([len(L),1])


pf_EpLz = grbpop.pflux.pflux_from_L(zg,Ep,Lg,alpha=alpha,model=specmodel,inst=inst)
Pdet = pf_EpLz>=pflim

if recompute:
    z_grid = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/z.npy'))
    tdmin_grid = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/tdmin.npy'))
    at_grid = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/at.npy'))
    rhoz_grid_pow = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/r_sgrb_pow.npy'))
    Itp_rhoz_pow = RegularGridInterpolator(points=(z_grid,tdmin_grid,at_grid),values=np.nan_to_num(rhoz_grid_pow),bounds_error=False)
    
    mu_td_grid = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/mu_td.npy'))
    sigma_td_grid = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/sigma_td.npy'))
    rhoz_grid_log = np.load(os.path.join(here,'grbpop/dtd_sfh_conv_tables/r_sgrb_log.npy'))
    Itp_rhoz_log = RegularGridInterpolator(points=(z_grid,mu_td_grid,sigma_td_grid),values=np.nan_to_num(rhoz_grid_log),bounds_error=False)

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
        
        theta_pop = {'rho_z':'DTD*SFH',
                 'dtd':'pow',
                 'alpha_L':x[i,0],
                 'beta_L':x[i,1],
                 'gamma_L':x[i,0],
                 'L_*':10.**x[i,2],
                 'L_**':6e49,
                 'L_0':5e49,
                 'tdmin':0.02,
                 'at':x[i,3],
                 'R0':10**x[i,4]
                 }
        # theta_pop = {'rho_z':'DTD*SFH',
        #          'alpha_L':x[i,0],
        #          'beta_L':x[i,1],
        #          'gamma_L':x[i,0],
        #          'L_*':10.**x[i,2],
        #          'L_**':6e49,
        #          'L_0':5e49,
        #          'mu_td':x[i,3],
        #          'sigma_td':x[i,4]
        #          }
        
        if chain2 is not None:
            # theta_pop2 = {'rho_z':'DTD*SFH',
            #          'dtd':'pow',
            #          'alpha_L':x2[i,0],
            #          'beta_L':x2[i,1],
            #          'gamma_L':x2[i,0],
            #          'L_*':10.**x2[i,2],
            #          'L_**':6e49,
            #          'L_0':5e49,
            #          'tdmin':0.02,
            #          'at':x2[i,3]
            #          }
            theta_pop2 = {'rho_z':'DTD*SFH',
                     'dtd':'lognorm',
                     'alpha_L':x2[i,0],
                     'beta_L':x2[i,1],
                     'gamma_L':x2[i,0],
                     'L_*':10.**x2[i,2],
                     'L_**':6e49,
                     'L_0':5e49,
                     'mu_td':x2[i,3],
                     'sigma_td':x2[i,4],
                     'R0':10**x2[i,5]
                     }
        
        if (theta_pop['rho_z']=='SBPL'): 
            rhoz = grbpop.Ppop.MD14_SFH(z,theta_pop['a'],theta_pop['b'],theta_pop['zp'])
            rhoz /= grbpop.Ppop.MD14_SFH(0,theta_pop['a'],theta_pop['b'],theta_pop['zp'])
        elif (theta_pop['rho_z']=='DTD*SFH' and theta_pop['dtd']=='pow'):
            rhoz = Itp_rhoz_pow((z,theta_pop['tdmin'],theta_pop['at']))
            rhoz /= Itp_rhoz_pow((0,theta_pop['tdmin'],theta_pop['at']))
        elif (theta_pop['rho_z']=='DTD*SFH' and theta_pop['dtd']=='lognorm'):
            rhoz = Itp_rhoz_log((z,theta_pop['mu_td'],theta_pop['sigma_td']))
            rhoz /= Itp_rhoz_log((0,theta_pop['mu_td'],theta_pop['sigma_td']))

        psiz = rhoz/(1.+z)*dVdz

        # PEpL = grbpop.Ppop.PEpL(L,Ep,theta_pop,grid=True)
        # PL = grbpop.Ppop.PL(L,theta_pop,grid=True)
        # PL = grbpop.Ppop.lum_2breaks_pdf(L, theta_pop)
        # PEpL/=np.trapezoid(np.trapezoid(PEpL*Epg[:,:,0]*Lg[:,:,0],np.log(L),axis=1),np.log(Ep))
        # PL/=np.trapezoid(PL*Lg[:,0],np.log(L),axis=0)
        # R0[i] = Robs/np.trapezoid(np.trapezoid(np.trapezoid(PEpL.reshape([len(Ep),len(L),1])*psiz.reshape([1,1,len(z)])*Epg*Lg*zg*Pdet,np.log(z),axis=2),np.log(L),axis=1),np.log(Ep))
        # R0[i] = Robs/np.trapezoid(np.trapezoid(PL.reshape([len(L),1])*psiz.reshape([1,len(z)])*Lg*zg*Pdet,np.log(z),axis=1),np.log(L),axis=0)
        R0[i] = theta_pop['R0']
        R0[i] = np.nan_to_num(R0[i])
        dR0_dlogL[i] = L*R0[i]*grbpop.Ppop.lum_2breaks_pdf(L,theta_pop)
        dN_dVdt[i] = R0[i]*psiz*(1.+z)/dVdz

        if chain2 is not None:
            if (theta_pop2['rho_z']=='SBPL'): 
                rhoz2 = grbpop.Ppop.MD14_SFH(z,theta_pop2['a'],theta_pop2['b'],theta_pop2['zp'])
                rhoz2 /= grbpop.Ppop.MD14_SFH(0,theta_pop2['a'],theta_pop2['b'],theta_pop2['zp'])
            elif (theta_pop2['rho_z']=='DTD*SFH' and theta_pop2['dtd']=='pow'):
                rhoz2 = Itp_rhoz_pow((z,theta_pop2['tdmin'],theta_pop2['at']))
                rhoz2 /= Itp_rhoz_pow((0,theta_pop2['tdmin'],theta_pop2['at']))
            elif (theta_pop2['rho_z']=='DTD*SFH' and theta_pop2['dtd']=='lognorm'):
                rhoz2 = Itp_rhoz_log((z,theta_pop2['mu_td'],theta_pop2['sigma_td']))
                rhoz2 /= Itp_rhoz_log((0,theta_pop2['mu_td'],theta_pop2['sigma_td']))
            
            psiz2 = rhoz2/(1.+z)*dVdz
            # PEpL2 = grbpop.Ppop.PEpL(L,Ep,theta_pop2,grid=True)
            # PL2 = grbpop.Ppop.PL(L,theta_pop2,grid=True)
            # PL2 = grbpop.Ppop.lum_2breaks_pdf(L, theta_pop2)
            # PEpL2/=np.trapezoid(np.trapezoid(PEpL2*Epg[:,:,0]*Lg[:,:,0],np.log(L),axis=1),np.log(Ep))
            # PL2/=np.trapezoid(PL2*Lg[:,0],np.log(L),axis=0)
            # R02[i] = Robs/np.trapezoid(np.trapezoid(np.trapezoid(PEpL2.reshape([len(Ep),len(L),1])*psiz2.reshape([1,1,len(z)])*Epg*Lg*zg*Pdet,np.log(z),axis=2),np.log(L),axis=1),np.log(Ep))
            # R02[i] = Robs/np.trapezoid(np.trapezoid(PL2.reshape([len(L),1])*psiz2.reshape([1,len(z)])*Lg*zg*Pdet,np.log(z),axis=1),np.log(L),axis=0)
            R02[i] = theta_pop2['R0']
            R02[i] = np.nan_to_num(R02[i])
            dR0_dlogL2[i] = L*R02[i]*grbpop.Ppop.lum_2breaks_pdf(L,theta_pop2)
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

# L_min_mid = 5e49
# R0_50 = np.nan_to_num(np.trapezoid(dR0_dlogL*(L>=L_min_mid),np.log(L),axis=1))

# if chain2 is not None:
#     R02_50 = np.nan_to_num(np.trapezoid(dR0_dlogL2*(L>=L_min_mid),np.log(L),axis=1))



# R0_low,R0_med,R0_high = np.percentile(R0,[5.,50.,95.])
R0_low,R0_med,R0_high = np.percentile(R0,[16.,50.,84.])
print('R_0=%.1f_{%.1f}^{%.1f} Gpc-3 yr-1'%(R0_med,R0_low-R0_med,R0_high-R0_med))

# # R0_50_low,R0_50_med,R0_50_high = np.percentile(R0_50,[5.,50.,95.])
# R0_50_low,R0_50_med,R0_50_high = np.percentile(R0_50,[16.,50.,84.])
# print('R_0_50=%.1f_{%.1f}^{%.1f} Gpc-3 yr-1'%(R0_50_med,R0_50_low-R0_50_med,R0_50_high-R0_50_med))

if chain2 is not None:
    # R02_low,R02_med,R02_high = np.percentile(R02,[5.,50.,95.])
    R02_low,R02_med,R02_high = np.percentile(R02,[16.,50.,84.])
    print('R_02=%.1f_{%.1f}^{%.1f} Gpc-3 yr-1'%(R02_med,R02_low-R02_med,R02_high-R02_med))
    
    # # R02_50_low,R02_50_med,R02_50_high = np.percentile(R02_50,[5.,50.,95.])
    # R02_50_low,R02_50_med,R02_50_high = np.percentile(R02_50,[16.,50.,84.])
    # print('R_02_50=%.1f_{%.1f}^{%.1f} Gpc-3 yr-1'%(R02_50_med,R02_50_low-R02_50_med,R02_50_high-R02_50_med))
    


############# Local rate ##############################################
logR0_kde = gaussian_kde(np.log(R0))
# logR0_50_kde = gaussian_kde(np.log(R0_50))

R00 = np.logspace(-2.,5.4,1000)
dP_dlogR0 = logR0_kde.pdf(np.log(R00))
# dP_dlogR0_50 = logR0_50_kde.pdf(np.log(R00))

np.save('cache/curves_for_plots/dP_dlogR0/'+chain[:-5]+'.npy', (R00, dP_dlogR0))
# np.save('cache/curves_for_plots/dP_dlogR0_50/'+chain[:-5]+'.npy', (R00, dP_dlogR0_50))

if chain2 is not None:
    logR02_kde = gaussian_kde(np.log(R02))
    # logR02_50_kde = gaussian_kde(np.log(R02_50))
    
    dP_dlogR02 = logR02_kde.pdf(np.log(R00))
    # dP_dlogR02_50 = logR02_50_kde.pdf(np.log(R00))
    
    np.save('cache/curves_for_plots/dP_dlogR0/'+chain2[:-5]+'.npy', (R00, dP_dlogR02))
    # np.save('cache/curves_for_plots/dP_dlogR0_50/'+chain2[:-5]+'.npy', (R00, dP_dlogR02_50))
    

################ Luminosity function ##############################

dR0_dlogL_m = np.percentile(dR0_dlogL,50.,axis=0)
dR0_dlogL_u = np.percentile(dR0_dlogL,95.,axis=0)
dR0_dlogL_l = np.percentile(dR0_dlogL,5.,axis=0)

np.save('cache/curves_for_plots/dR_dlogL/'+chain[:-5]+'.npy',(L, dR0_dlogL_m, dR0_dlogL_u, dR0_dlogL_l))


if chain2 is not None:
    dR0_dlogL2_m = np.percentile(dR0_dlogL2,50.,axis=0)
    dR0_dlogL2_u = np.percentile(dR0_dlogL2,95.,axis=0)
    dR0_dlogL2_l = np.percentile(dR0_dlogL2,5.,axis=0)
    
    np.save('cache/curves_for_plots/dR_dlogL/'+chain2[:-5]+'.npy',(L, dR0_dlogL2_m, dR0_dlogL2_u, dR0_dlogL2_l))


#################### redshift evolution ####################

# dN_dVdt *= (R0_50/R0).reshape([N,1])

dN_dVdt_m = np.percentile(dN_dVdt,50.,axis=0)
dN_dVdt_u = np.percentile(dN_dVdt,95.,axis=0)
dN_dVdt_l = np.percentile(dN_dVdt,5.,axis=0)

np.save('cache/curves_for_plots/dN_dVdt/'+chain[:-5]+'.npy',(z, dN_dVdt_m, dN_dVdt_u, dN_dVdt_l))

if chain2 is not None:
    # dN_dVdt2 *= (R02_50/R02).reshape([N2,1])

    dN_dVdt2_m = np.percentile(dN_dVdt2,50.,axis=0)
    dN_dVdt2_u = np.percentile(dN_dVdt2,95.,axis=0)
    dN_dVdt2_l = np.percentile(dN_dVdt2,5.,axis=0)
    
    np.save('cache/curves_for_plots/dN_dVdt/'+chain2[:-5]+'.npy',(z, dN_dVdt2_m, dN_dVdt2_u, dN_dVdt2_l))


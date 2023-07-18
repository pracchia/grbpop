import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from astropy.cosmology import Planck15 as cosmo
from astropy.io import ascii
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz
from scipy.stats import kstest
from scipy.special import erf
import pandas
import grbpop
from inspect_fit_results import read_chain


cmap = plt.cm.Blues_r
boundaries = np.array([0.,0.5,0.9,0.99,0.999,0.9999,1.])
norm = mplcolors.BoundaryNorm(boundaries=boundaries,ncolors=256)

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=8,6
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'

def ltrapz(x,y,axis=-1,reshape=None):
    if reshape is None:
        return np.trapz(y*x,np.log(y),axis=axis)
    else:
        return np.trapz(y.reshape(reshape)*x,np.log(y),axis=axis)

def lcumtrapz(x,y,axis=-1,initial=0.):
    return cumtrapz(y*x,np.log(y),axis=axis,initial=initial)


x2 = [-1.285, 51.66, 4.891, 1.905, 3.653, 1.549, 1.51, 0.05123, 2.911, -0.05058, -0.01845, 3.82, 5.462, 2.346] # DBPL flux-limited
x = [-1.325, 51.58, 4.364, -0.2971, 4.352, 1.868, -1.76, 0.1811, 3.245, -0.0001222, -0.3173, 4.849, 9.02, 2.4] #DBPL full


theta_pop2 = {'jetmodel':'smooth double power law',
             'thc':10**x2[0],
             'Lc*':10**x2[1],
             'a_L':x2[2],
             'b_L':x2[3],
             'Epc*':10**x2[4],
             'a_Ep':x2[5],
             'b_Ep':x2[6],
             'thw':10**x2[7],
             'A':x2[8],
             's_c':10**x2[9],
             'y':x2[10],
             'a':x2[11],
             'b':x2[12],
             'zp':x2[13]
             }

theta_pop = {'jetmodel':'smooth double power law',
             'thc':10**x[0],
             'Lc*':10**x[1],
             'a_L':x[2],
             'b_L':x[3],
             'Epc*':10**x[4],
             'a_Ep':x[5],
             'b_Ep':x[6],
             'thw':10**x[7],
             'A':x[8],
             's_c':10**x[9],
             'y':x[10],
             'a':x[11],
             'b':x[12],
             'zp':x[13]
             }

plim_Swift = 3.5
plim_Fermi = 0.01
inst_restframe = 'Fermi+Swift'
specmodel = 'Comp'
alpha = -0.4
pdet = lambda pf,epp: (pf>3.5)
    

L = np.logspace(44,54.5,51)
Ep = np.logspace(0,5,50)
z = np.logspace(-3,1,49)

PEpL = grbpop.PEpL(L,Ep,theta_pop)

PEp = ltrapz(PEpL,L,axis=1,reshape=[1,len(L)])
PL = ltrapz(PEpL,Ep,axis=0,reshape=[len(Ep),1])
Pz = grbpop.Ppop.Pz(z,theta_pop)


############################################### Rest-frame constraints #######################################

plt.figure('Rest-frame constraints')

L,Ep,z,PEpL_contours,cum_L,cum_Ep,cum_z = grbpop.diagnose.L_Ep_z_contours_and_cumulatives(theta_pop,pflim=[plim_Fermi,plim_Swift],pdet='gbm',inst=inst_restframe,alpha=alpha,specmodel=specmodel)

L2,Ep2,z2,PEpL_contours2,cum_L2,cum_Ep2,cum_z2 = grbpop.diagnose.L_Ep_z_contours_and_cumulatives(theta_pop2,pflim=[plim_Fermi,plim_Swift],pdet=pdet,inst=inst_restframe,alpha=alpha,specmodel=specmodel)

LGW,EpGW,zGW,PEpL_contoursGW,cum_LGW,cum_EpGW,cum_zGW = grbpop.diagnose.L_Ep_z_contours_and_cumulatives(theta_pop,pflim=None,pdet='gbm',inst='Fermi+GW',alpha=alpha,specmodel=specmodel,pdetGW=grbpop.pdet.pdet_GW_O3)


plt.subplot(223)
plt.annotate(xy=(0.85,0.05),xycoords='axes fraction',text='(a.0)')


plt.contourf(L,Ep,PEpL_contours,levels=[0.,0.5,0.9,0.99,0.999],cmap=cmap,norm=norm)
plt.contour(L2,Ep2,PEpL_contours2,levels=[0.5,0.9,0.99,0.999],colors=['grey'],linestyles=['--'],alpha=0.5)
plt.contour(LGW,EpGW,PEpL_contoursGW,levels=[0.5,0.9,0.99,0.999],cmap=plt.cm.Oranges,norm=norm,alpha=0.5)

plt.xlabel('$L\,[\mathrm{erg/s}]$')
plt.ylabel('$E_\mathrm{p}\,[\mathrm{keV}]$')
plt.loglog()

plt.ylim(6e0,5e4)
plt.xlim(1e46,1e55)
plt.tick_params(which='both',direction='in',top=True,right=True)

# show obs GRBs
Lsamples = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_L_samples.npy')
Epsamples = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_Ep_samples.npy')
zsamples = np.load('grb_data/GBM_BAT_SGRB_spec_or_photoz_zobs.npy')

SGRBs = {'Liso':np.median(Lsamples[:-1],axis=1),
         'Liso_err_low':np.median(Lsamples[:-1],axis=1)-np.percentile(Lsamples[:-1],5.,axis=1),
         'Liso_err_high':-np.median(Lsamples[:-1],axis=1)+np.percentile(Lsamples[:-1],95.,axis=1),
         'Ep':np.median(Epsamples[:-1],axis=1),
         'Ep_err_low':np.median(Epsamples[:-1],axis=1)-np.percentile(Epsamples[:-1],5.,axis=1),
         'Ep_err_high':-np.median(Epsamples[:-1],axis=1)+np.percentile(Epsamples[:-1],95.,axis=1),
         'z':np.median(zsamples[:-1],axis=1)}

print('KS prob L (SGRB): ',kstest(SGRBs['Liso'],lambda x: np.interp(x,L,cum_L)).pvalue,kstest(SGRBs['Liso'],lambda x: np.interp(x,L2,cum_L2)).pvalue)
print('KS prob Ep (SGRB): ',kstest(SGRBs['Ep'],lambda x: np.interp(x,Ep,cum_Ep)).pvalue,kstest(SGRBs['Ep'],lambda x: np.interp(x,Ep2,cum_Ep2)).pvalue)
print('KS prob z (SGRB): ',kstest(SGRBs['z'],lambda x: np.interp(x,z,cum_z)).pvalue,kstest(SGRBs['z'],lambda x: np.interp(x,z2,cum_z2)).pvalue)


plt.errorbar(SGRBs['Liso'],SGRBs['Ep'],xerr=(SGRBs['Liso_err_low'],SGRBs['Liso_err_high']),yerr=(SGRBs['Ep_err_low'],SGRBs['Ep_err_high']),ls='None',color='r',lw=0.8,label='GBM+BAT sample',zorder=3,capsize=0.)


# GRB 170817A
Liso_GRB170817A = np.mean(Lsamples[-1])   #1.6e47 #https://iopscience.iop.org/article/10.3847/2041-8213/aa920c/pdf
dLiso_GRB170817A_low = -np.percentile(Lsamples[-1],5.)+np.mean(Lsamples[-1])
dLiso_GRB170817A_high = np.percentile(Lsamples[-1],95.)-np.mean(Lsamples[-1])
Ep_GRB170817A = np.mean(Epsamples[-1])  #185.
dEp_GRB170817A_low = -np.percentile(Epsamples[-1],5.)+np.mean(Epsamples[-1]) #62.
dEp_GRB170817A_high = np.percentile(Epsamples[-1],95.)-np.mean(Epsamples[-1]) #62.
thv_GRB170817A = 15.
dthv_GRB170817A_low = 1.
dthv_GRB170817A_high = 5.5

Lsamples = Lsamples[:-1]
Epsamples = Epsamples[:-1]

plt.errorbar([Liso_GRB170817A],[Ep_GRB170817A],yerr=[[dEp_GRB170817A_low],[dEp_GRB170817A_high]],xerr=[[dLiso_GRB170817A_low],[dLiso_GRB170817A_high]],marker='o',markersize=1,color='orange',zorder=3,capsize=0.,ls='None',elinewidth=0.5,label='GRB 170817A')

plt.tick_params(which='both',direction='in',top=True,right=True)

plt.legend(frameon=False,loc='upper left')

# ----- Luminosity subplot ---
NN = 10

cum_L = np.zeros([NN,len(cum_L)])
cum_L2 = np.zeros([NN,len(cum_L2)])
cum_Ep = np.zeros([NN,len(cum_Ep)])
cum_Ep2 = np.zeros([NN,len(cum_Ep2)])
cum_z = np.zeros([NN,len(cum_z)])
cum_z2 = np.zeros([NN,len(cum_z2)])

chain2 = 'chains/SGRB_flux-limited-sample-analysis.h5'
chain = 'chains/SGRB_full-sample-analysis.h5'

thin = 10

x,ll = read_chain(chain,burnin_fraction=0.5,thin=thin)
x2,ll2 = read_chain(chain2,burnin_fraction=0.5,thin=thin)

for i in range(NN):
    theta_pop_i = {'jetmodel':'smooth double power law',
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
             'a':x[i,11],
             'b':x[i,12],
             'zp':x[i,13]
             }
             
    theta_pop2_i = {'jetmodel':'smooth double power law',
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
             'a':x2[i,11],
             'b':x2[i,12],
             'zp':x2[i,13]
             }


    L,Ep,z,PEpL_contours,cum_L[i],cum_Ep[i],cum_z[i] = grbpop.diagnose.L_Ep_z_contours_and_cumulatives(theta_pop_i,pflim=[plim_Fermi,plim_Swift],pdet='gbm',inst=inst_restframe,alpha=alpha,specmodel=specmodel)
    L2,Ep2,z2,PEpL_contours2,cum_L2[i],cum_Ep2[i],cum_z2[i] = grbpop.diagnose.L_Ep_z_contours_and_cumulatives(theta_pop2_i,pflim=[plim_Fermi,plim_Swift],pdet=pdet,inst=inst_restframe,alpha=alpha,specmodel=specmodel)

plt.subplot(221)
plt.annotate(xy=(0.85,0.05),xycoords='axes fraction',text='(a.1)')

plt.fill_between(L,np.percentile(cum_L,5.,axis=0),np.percentile(cum_L,95.,axis=0),facecolor=cmap(0.67),alpha=0.9)
plt.plot(L,np.percentile(cum_L,50,axis=0),ls='-',c='b',lw=2,label='Full')

plt.plot(L2,np.percentile(cum_L2,5.,axis=0),ls=':',c='grey',lw=1,alpha=0.9)
plt.plot(L2,np.percentile(cum_L2,95.,axis=0),ls=':',c='grey',lw=1,alpha=0.9)
plt.plot(L2,np.percentile(cum_L2,50.,axis=0),ls='--',c='grey',lw=2,label='Flux-limited',alpha=0.8)

sgrb_cum_L = np.zeros(len(L)) 
sgrb_cum_L_low = np.zeros(len(L))
sgrb_cum_L_high = np.zeros(len(L))

sgrb_cum_L_all = np.arange(0,Lsamples[:-1].shape[0]+1)/Lsamples[:-1].shape[0]
Lcum_all = np.sort(Lsamples[:-1],axis=0)
Lcum_05,Lcum_50,Lcum_95 = np.percentile(Lcum_all,[5.,50.,95.],axis=1)

Lcum_05 = np.concatenate([[Lcum_50[0]],Lcum_05])
Lcum_50 = np.concatenate([[Lcum_50[0]],Lcum_50])
Lcum_95 = np.concatenate([[Lcum_50[0]],Lcum_95])

plt.fill_betweenx(sgrb_cum_L_all,Lcum_05,Lcum_95,facecolor='pink',alpha=0.5,step='pre')
plt.step(Lcum_50,sgrb_cum_L_all,color='r',lw=3,where='post',label='Observed')

plt.xlim(1e46,1e55)
plt.ylim([0.,1.])
plt.semilogx()
plt.tick_params(which='both',direction='in',top=True,right=True,labelbottom=False,labeltop=True)

plt.gca().xaxis.set_label_position('top')
plt.xlabel('$L\,[\mathrm{erg/s}]$')
plt.ylabel('Cumulative fraction')


plt.legend(loc='upper left',frameon=False,markerfirst=True)

## ---- Epeak subplot ---------
plt.subplot(224)
plt.annotate(xy=(0.85,0.05),xycoords='axes fraction',text='(a.2)')

plt.fill_betweenx(Ep,np.percentile(cum_Ep,5.,axis=0),np.percentile(cum_Ep,95.,axis=0),facecolor=cmap(0.67),alpha=0.9)
plt.plot(np.percentile(cum_Ep,50.,axis=0),Ep,ls='-',c='b',lw=2)

plt.plot(np.percentile(cum_Ep2,95.,axis=0),Ep2,ls=':',c='grey',lw=1,alpha=0.9)
plt.plot(np.percentile(cum_Ep2,5.,axis=0),Ep2,ls=':',c='grey',lw=1,alpha=0.9)
plt.plot(np.percentile(cum_Ep2,50.,axis=0),Ep2,ls='--',c='grey',lw=2,alpha=0.8)

sgrb_cum_Ep_all = np.arange(0,Epsamples[:-1].shape[0]+1)/Epsamples[:-1].shape[0]
Epcum_all = np.sort(Epsamples[:-1],axis=0)
Epcum_05,Epcum_50,Epcum_95 = np.percentile(Epcum_all,[5.,50.,95.],axis=1)

Epcum_05 = np.concatenate([[Epcum_50[0]],Epcum_05])
Epcum_50 = np.concatenate([[Epcum_50[0]],Epcum_50])
Epcum_95 = np.concatenate([[Epcum_50[0]],Epcum_95])

plt.fill_between(sgrb_cum_Ep_all,Epcum_05,Epcum_95,facecolor='pink',alpha=0.5,step='pre')
plt.step(sgrb_cum_Ep_all,Epcum_50,color='r',lw=3,where='pre')
plt.ylim([6e0,5e4])
plt.xlim([0.,1.])
plt.semilogy()
plt.tick_params(which='both',direction='in',top=True,right=True,labelleft=False,labelright=True)

plt.gca().yaxis.set_label_position('right')
plt.ylabel('$E_\mathrm{p}\,[\mathrm{keV}]$')
plt.xlabel('Cumulative fraction')

# redshift subplot
plt.subplot(222)
plt.annotate(xy=(0.85,0.05),xycoords='axes fraction',text='(b)')


plt.fill_between(z,np.percentile(cum_z,5.,axis=0),np.percentile(cum_z,95.,axis=0),facecolor=cmap(0.67),alpha=0.9)
plt.plot(z,np.percentile(cum_z,50.,axis=0),ls='-',color='b',lw=2)

plt.plot(z2,np.percentile(cum_z2,5.,axis=0),ls=':',color='grey',lw=1,alpha=0.9)
plt.plot(z2,np.percentile(cum_z2,95.,axis=0),ls=':',color='grey',lw=1,alpha=0.9)
plt.plot(z2,np.percentile(cum_z2,50.,axis=0),ls='--',color='grey',lw=2,alpha=0.8)

sgrb_cum_z_all = np.arange(0,zsamples[:-1].shape[0]+1)/zsamples[:-1].shape[0]
zcum_all = np.sort(zsamples[:-1],axis=0)
zcum_05,zcum_50,zcum_95 = np.percentile(zcum_all,[5.,50.,95.],axis=1)

zcum_05 = np.concatenate([[zcum_50[0]],zcum_05])
zcum_50 = np.concatenate([[zcum_50[0]],zcum_50])
zcum_95 = np.concatenate([[zcum_50[0]],zcum_95])

plt.fill_betweenx(sgrb_cum_z_all,zcum_05,zcum_95,facecolor='pink',alpha=0.5,step='pre')
plt.step(zcum_50,sgrb_cum_z_all,color='r',lw=3,where='post')

plt.xlim(0.,2.5)
plt.ylim(0.,1.)

plt.title('Redshift')
plt.xlabel(r'$z$')
plt.ylabel('Cumulative fraction')

plt.gca().yaxis.set_label_position('right')

plt.tick_params(which='both',direction='in',top=True,right=True,labelleft=False,labelright=True)

plt.savefig('figures/restframe_constraints.pdf')


####################################### Observer-frame constraints #################################################


plt.figure('Observer-frame constraints')

plt.subplot(223)
plt.annotate(xy=(0.05,0.9),xycoords='axes fraction',text='(a.0)')

pf0,ep0,Ppfep_contours,cum_pf,cum_ep = grbpop.diagnose.pf_ep_contours_and_cumulatives(theta_pop,pflim=0.01,pdet='gbm',inst='Fermi',alpha=alpha,specmodel=specmodel)
pf02,ep02,Ppfep_contours2,cum_pf2,cum_ep2 = grbpop.diagnose.pf_ep_contours_and_cumulatives(theta_pop2,pflim=3.5,pdet=None,inst='Fermi',alpha=alpha,specmodel=specmodel)

cb = plt.contourf(pf0,ep0,Ppfep_contours,levels=[0.,0.5,0.9,0.99,0.999],cmap=cmap,norm=norm)
cb2 = plt.contour(pf02,ep02,Ppfep_contours2,levels=[0.5,0.9,0.99,0.999],colors=['grey'],alpha=0.5,linestyles=['--'],zorder=10)

plt.xlabel('$p_{[50-300]}\,[\mathrm{cm^{-2}\,s^{-1}}]$')
plt.ylabel('$E_\mathrm{p,obs}\,[\mathrm{keV}]$')
plt.loglog()

plt.ylim(0.3e1,1e6)
plt.xlim(1e-1,3e3)

gbm = pandas.read_csv('grb_data/GBM_pflx_allinfo.csv')
sgrb = gbm.loc[gbm['t90']<2.]
p50300 = sgrb['pflx_comp_phtfluxb'].values
p50300_err = sgrb['pflx_comp_phtfluxb_error'].values
ep = sgrb['pflx_comp_epeak'].values
ep_neg_err = sgrb['pflx_comp_epeak_neg_err'].values
ep_pos_err = sgrb['pflx_comp_epeak_pos_err'].values

alpha = sgrb['pflx_comp_index'].values
clean = (ep>0) & (ep<1e6) & (p50300>0)

ep = ep[clean]
ep_neg_err = ep_neg_err[clean]
ep_pos_err = ep_pos_err[clean]
alpha = alpha[clean]
pf = p50300[clean]
pf_err = p50300_err[clean]

noshit = (ep>50.)&(ep<1e4)&(pf>1.)
shit = ~noshit

precise = ((pf_err/pf)<0.5)&((ep_pos_err/ep)<0.5)&((ep_neg_err/ep)<0.5)

# plot good ones (no shit)
plt.errorbar(pf[noshit],ep[noshit],xerr=np.where(precise[noshit],0.,pf_err[noshit]),yerr=(np.where(precise[noshit],0.,ep_neg_err[noshit]),np.where(precise[noshit],0.,ep_pos_err[noshit])),ls='None',color='grey',lw=0.1,capsize=0,label='Short',zorder=3,marker='+',markersize=1.5,mfc='r',mec='r')

# plot shit
plt.errorbar(pf[shit],ep[shit],xerr=np.where(precise[shit],0.,pf_err[shit]),yerr=(np.where(precise[shit],0.,ep_neg_err[shit]),np.where(precise[shit],0.,ep_pos_err[shit])),ls='None',color='grey',lw=0.1,capsize=0,zorder=3,marker='+',markersize=1.5,mfc='grey',mec='grey')

plt.tick_params(which='both',direction='in',top=True,right=True)

# remove shit before constructing the marginals
ep = ep[noshit]
ep_pos_err = ep_pos_err[noshit]
ep_neg_err = ep_neg_err[noshit]
pf = pf[noshit]
pf_err = pf_err[noshit]

logerr_pos = np.sqrt(np.log(1.+ep_pos_err**2/ep**2))
logerr_neg = np.sqrt(np.log(1.+ep_neg_err**2/ep**2))

logep_pos_err_samples = np.abs(np.random.normal(0.,logerr_pos.reshape([len(ep),1]),[len(ep),1000]))
logep_neg_err_samples = -np.abs(np.random.normal(0.,logerr_neg.reshape([len(ep),1]),[len(ep),1000]))

ep_samples = ep.reshape([len(ep),1])*np.exp(logep_pos_err_samples)
ep_samples[:,::2] = ep.reshape([len(ep),1])*np.exp(logep_neg_err_samples[:,::2])

logpferr = np.sqrt(np.log(1.+pf_err**2/pf**2))

pf_samples = pf.reshape([len(pf),1])*np.exp(np.random.normal(0.,logpferr.reshape([len(pf),1]),[len(pf),1000]))

plt.subplot(221)
plt.annotate(xy=(0.05,0.9),xycoords='axes fraction',text='(a.1)')

frac_fluxlimited = len(pf[pf>3.5])/len(pf)

cum_pf2 = (1.-frac_fluxlimited) + frac_fluxlimited*cum_pf2

plt.plot(pf0,cum_pf,ls='-',c='b',lw=0.8,label='Full',zorder=9)
plt.plot(pf02,cum_pf2,ls='--',c='grey',lw=1.5,alpha=0.8,zorder=10,label='Flux-limited')

sgrb_cum_pf = np.arange(1,len(pf)+0.5,1)/len(pf)

plt.fill_betweenx(sgrb_cum_pf,np.percentile(np.sort(pf_samples,axis=0),5.,axis=1),np.percentile(np.sort(pf_samples,axis=0),95.,axis=1),color='pink',alpha=0.5,zorder=-1,step='pre')
plt.step(np.sort(pf),sgrb_cum_pf,color='r',lw=1,where='pre',label='Observed')
plt.xlim(1e-1,3e3)
plt.ylim([0.,1.])
plt.semilogx()

plt.tick_params(which='both',direction='in',top=True,right=True,labelbottom=False,labeltop=True)

plt.gca().xaxis.set_label_position('top')
plt.xlabel('$p_{[50-300]}\,[\mathrm{cm^{-2}\,s^{-1}}]$')
plt.ylabel('Cumulative fraction')

plt.legend(loc='lower right',frameon=False,markerfirst=False)

plt.subplot(224)
plt.annotate(xy=(0.05,0.9),xycoords='axes fraction',text='(a.2)')

plt.plot(cum_ep,ep0,ls='-',c='b',lw=0.8,zorder=9)
plt.plot(cum_ep2,ep02,ls='--',c='grey',lw=1.5,alpha=0.8,zorder=10)

sgrb_cum_ep = np.linspace(1.,len(ep),len(ep))/len(ep)

plt.fill_between(sgrb_cum_ep,np.percentile(np.sort(ep_samples,axis=0),5.,axis=1),np.percentile(np.sort(ep_samples,axis=0),95.,axis=1),color='pink',alpha=0.5,zorder=-1,step='pre')
plt.step(sgrb_cum_ep,np.sort(ep),color='r',lw=1,where='pre')
plt.ylim([0.3e1,1e6])
plt.xlim([0.,1.])
plt.semilogy()
plt.tick_params(which='both',direction='in',top=True,right=True,labelleft=False,labelright=True)

plt.ylabel('$E_\mathrm{p,obs}\,[\mathrm{keV}]$')
plt.xlabel('Cumulative fraction')

plt.gca().yaxis.set_label_position('right')

############# logN-logS ####################
plt.subplot(222)
plt.annotate(xy=(0.85,0.9),xycoords='axes fraction',text='(b)')

c = np.arange(len(pf),0.5,-1)
c_low = c-c**0.5
c_high = c+c**0.5

# plot resulting fit
plt.loglog()
plt.fill_between(np.sort(pf),c_low/c.max(),c_high/c.max(),edgecolor='None',facecolor='#FFC0CB',step='pre')
plt.step(np.sort(pf),c/c.max(),'-r')
plt.plot(pf0,1.-cum_pf,ls='-',color='b',lw=0.8)
plt.plot(pf02,1.-cum_pf2,ls='--',color='grey',alpha=0.8,zorder=10)
plt.xlim(5e-1,2e2)
plt.ylim(1e-3,3e0)
plt.xlabel(r'$p_{[50-300]}$ [cm$^{-2}$ s$^{-1}$]')
plt.ylabel(r'$N_\mathrm{obs}(>p_{[50-300]})/N_\mathrm{obs}$')

plt.tick_params(which='both',direction='in',top=True,right=True)

plt.savefig('figures/obsframe_constraints.pdf')

plt.show()

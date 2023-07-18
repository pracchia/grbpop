import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
from sklearn.decomposition import PCA

def credible(x_i,level=0.68,bins=50):
    y,x = np.histogram(x_i,bins=bins)
    
    s = np.argsort(y)
    c = np.zeros_like(y)
    c[s] = np.cumsum(y[s])
    c = c/c.max()
    c0 = np.min(x[:-1][c>(1.-level)])
    c1 = np.max(x[1:][c>(1.-level)])
    return c0,c1

def read_chain(chain_filename,burnin_fraction=0.1,thin=1,bounds=None):
    # open chain file
    reader = emcee.backends.HDFBackend(chain_filename,read_only=True)
    
    # extract chain and loglike
    discard = int(burnin_fraction*reader.iteration)
    flatchain = reader.get_chain(flat=True,discard=discard,thin=thin)
    flatll = reader.get_log_prob(flat=True,discard=discard,thin=thin)
    
    # impose bounds if given
    if bounds is not None:
        for i in range(flatchain.shape[-1]):
            if i in bounds.keys():
                flatll = flatll[(flatchain[:,i]>bounds[i][0])&(flatchain[:,i]<bounds[i][1])]
                flatchain = flatchain[(flatchain[:,i]>bounds[i][0])&(flatchain[:,i]<bounds[i][1])]
                
    
    return flatchain,flatll
    
def Delta_AIC(chain_filename1,chain_filename2):
    
    # open chain file
    reader = emcee.backends.HDFBackend(chain_filename1,read_only=True)
    
    # extract chain and loglike
    discard = int(burnin_fraction*reader.iteration)
    flatchain = reader.get_chain(flat=True,discard=discard,thin=thin)
    k1 = flatchain.shape[-1]
    flatll1 = reader.get_log_prob(flat=True,discard=discard,thin=thin)
    
    # open chain file
    reader = emcee.backends.HDFBackend(chain_filename2,read_only=True)
    
    # extract chain and loglike
    discard = int(burnin_fraction*reader.iteration)
    flatchain = reader.get_chain(flat=True,discard=discard,thin=thin)
    k2 = flatchain.shape[-1]
    flatll2 = reader.get_log_prob(flat=True,discard=discard,thin=thin)
    
    return 2*(k2-k1+flatll2.max()-flatll1.max())

def corner_plot(chain_filename,burnin_fraction=0.1,savefig=None,show=False,verbose=True,credible_level=0.9,limit_level=0.95,labels=None,truths=None,bestfit_method='pca',bounds=None,transform=None,thin=1,return_chain=False,hide=None,return_figure=False,truth_color='r',cred_int_color='k',annotate_cred_int=True,greedy=True,**kwargs):
    """
    Make a corner plot of the posterior probability density distribution.
    
    Parameters:
    - chain_filename:  the name of the emcee .h5 file that contains the 
                       chain
    - burnin_fraction: the fraction of the posterior samples to be 
                       discarded as burnin
    - savefig:         if set to a string, this is the filename to which
                       the output figure is saved
    - show:            if True, show the corner plot in an interactive window.
    - verbose:         if True, print some information, including the
                       parameter confidence bounds
    - credible_level:  the level at which credible bounds are given. 
                       Default: 0.9 (90%)
    - limit_level:     the ccredible level at which upper or lower limits 
                       must be given. Default: 0.95 (2 sigma)
    - labels:          Parameter names to be used as labels in the plot. Optional.
    - truths:          The "true" parameter values.
    - bestfit_method:  'pca' approximates the maximum a posteriori by doing
                       a principal component analysis decomposition and taking
                       the mean in the decomposed variable space; 'median'
                       gives the median of the marginalised posterior; 
                       'best sample' just takes the sample with the best
                       probability.
    - bounds:          dictionary in the form {i:(l,h), ...}, where i is the
                       index of the parameter and l and h are the lower and
                       upper bounds. If given, the chain is cut off to exclude
                       parameter values outside of the bounds.
    - transform:       dictionary in the form {i:f, ...}, where i is the
                       index of the parameter and f is a function that is
                       applied to all the samples to transform the parameter
                       to a different representation (e.g. to transform from
                       log to linear).
    - hide     :       list containing indices of parameters that must not
                       be shown in the corner plot.
    - thin:            factor by which the chain must be thinned. Default: 1
    - return_chain:    if True, return flat chain and flat log probability
    - return_figure:   if True, return figure
    - cred_int_color:  the color of the vertical lines that show the credible ranges
                       in the 1D plots. Default: 'black' 
    - annotate_cred_int: whether to annotate the credible intervals on top of the 
                       diagonal plots. Default: True 
    - **kwargs:        arguments passed to corner.corner
    
    """
    
    # open chain file
    reader = emcee.backends.HDFBackend(chain_filename,read_only=True)
    mean_autocorr_time = np.mean(reader.get_autocorr_time(tol=0))
    # get autocorr time & other info
    if verbose:
        print('Number of iterations recorded in chain file: ',reader.iteration)
        print('Mean autocorrelation time: ',mean_autocorr_time)
    
    if thin=='autocorr':
        thin = int(mean_autocorr_time)//5
    
    # extract chain and loglike
    discard = int(burnin_fraction*reader.iteration)
    flatchain = reader.get_chain(flat=True,discard=discard,thin=thin)
    flatll = reader.get_log_prob(flat=True,discard=discard,thin=thin)
    ndim = flatchain.shape[-1]
    
    
    
    # apply transforms if given
    if transform is not None:
        for i in range(flatchain.shape[-1]):
            if i in transform.keys():
                flatchain[:,i] = transform[i](flatchain[:,i])
                
    # impose bounds if given
    if bounds is not None:
        for i in range(flatchain.shape[-1]):
            if i in bounds.keys():
                flatll = flatll[(flatchain[:,i]>bounds[i][0])&(flatchain[:,i]<bounds[i][1])]
                flatchain = flatchain[(flatchain[:,i]>bounds[i][0])&(flatchain[:,i]<bounds[i][1])]
    
    if hide is not None:
        h = np.array([(i not in hide) for i in np.arange(ndim)],dtype=bool)
        flatchain = flatchain[:,h]
        ndim -= len(hide)
    
    
    # find maximum a posteriori
    if bestfit_method=='median':
        maxap = np.median(flatchain,axis=0)
    elif bestfit_method=='pca':
        # find best fit using PCA
        pca = PCA()
        pca.fit(flatchain)
        xt = pca.transform(flatchain)
        maxap = pca.inverse_transform(np.mean(xt,axis=0))
    elif bestfit_method=='best sample':
        maxap = flatchain[np.argmax(flatll)]
    else:
        maxap = np.mean(flatchain,axis=0)

    
    if labels is None:
        labels = ['' for i in range(ndim)]
    
    if truths is None:
        figure = corner.corner(flatchain,labels=labels,truths=maxap,truth_color=truth_color,**kwargs)
    else:
        figure = corner.corner(flatchain,labels=labels,truths=truths,truth_color=truth_color,**kwargs)
    
    axes = np.array(figure.axes).reshape((ndim, ndim))
    
    # show maximum a posteriori & credible intervals on the diagonal
    lower_bounds = np.zeros(ndim)
    upper_bounds = np.zeros(ndim)
    
    for i in range(ndim):
        lolim = False
        uplim = False
        lev = credible_level
        if greedy:
            c0,c1 = credible(flatchain[:,i],level=lev)
        else:
            c0,c1 = np.percentile(flatchain[:,i],[50.-100.*lev/2.,50.+100.*lev/2.])
        
        if c0==flatchain[:,i].min():
            lev = limit_level
            uplim = True
            c0,c1 = credible(flatchain[:,i],level=lev)
        elif c1==flatchain[:,i].max():
            lev = limit_level
            lolim = True
            c0,c1 = credible(flatchain[:,i],level=lev)
            
        Dh = c1-maxap[i]
        Dl = maxap[i]-c0
        if lolim:
            print('{0} > {1:.3f} ({2:.0%})'.format(labels[i%len(labels)],c0,lev))
        elif uplim:
            print('{0} < {1:.3f} ({2:.0%})'.format(labels[i%len(labels)],c1,lev))
        else:
            print('{0} = {1:.3f} (+{2:.3f}, -{3:.3f}) ({4:.0%})'.format(labels[i%len(labels)],maxap[i],Dh,Dl,lev))
        ax = axes[i, i]
        ax.axvline(c0, color=cred_int_color,linestyle='--')
        ax.axvline(c1, color=cred_int_color,linestyle='--')
        
        if annotate_cred_int:
            tit = r'$' + '{0:.2f}'.format(maxap[i]) + r'^{' + '+{0:.2f}'.format(Dh) + r'}_{' + '-{0:.2f}'.format(Dl)
            if lev!=credible_level:
                tit = tit + r'}$' + ' ({0:.0f}%)'.format(lev*100.)
            else:
                tit = tit + r'}$'
            ax.set_title(tit)
        
        lower_bounds[i]=c0
        upper_bounds[i]=c1
        
    
    if savefig is not None:
        plt.savefig(savefig)
    
    if show:
        plt.show()
    
    if return_chain and return_figure:
        return figure,maxap,lower_bounds,upper_bounds,flatchain,flatll
    elif return_chain:
        return maxap,lower_bounds,upper_bounds,flatchain,flatll
    elif return_figure:
        return figure,maxap,lower_bounds,upper_bounds
    else:
        return maxap,lower_bounds,upper_bounds



def chain_plot(chain_filename,burnin_fraction=0.1,labels=None,truths=None):
    """
    Make a plot of the chains and logprobability.
    
    Parameters:
    - chain_filename:  the name of the emcee .h5 file that contains the 
                       chain
    - burnin_fraction: the fraction of the posterior samples to be 
                       discarded as burnin
    - labels:          
    - truths:          
    
    """
    
    # open chain file
    reader = emcee.backends.HDFBackend(chain_filename,read_only=True)
    
    # extract chain and loglike
    chain = reader.get_chain(flat=False)
    ll = reader.get_log_prob(flat=False)
    ndim = chain.shape[-1]
    
    if labels is None:
        labels = ['' for i in range(ndim)]
    
    for i in range(ndim):
        plt.subplot(ndim+1,1,i+1)
        for j in range(chain.shape[1]):
            plt.plot(np.arange(chain.shape[0]),chain[:,j,i],alpha=0.5)
        plt.ylabel(labels[i])
        plt.tick_params(which='both',direction='in',top=True,right=True,labelbottom=False)
        plt.xlim([0,chain.shape[0]])
        
        if truths is not None:
            plt.axhline(y=truths[i],ls='--',color='k',zorder=100)
    
    plt.subplot(ndim+1,1,ndim+1)
    plt.plot(np.arange(chain.shape[0]),ll,alpha=0.5)
    plt.ylabel('logprob')
    plt.xlabel('step')
    plt.xlim([0,chain.shape[0]])
    plt.tick_params(which='both',direction='in',top=True,right=True,labelbottom=True)
    
    
    plt.subplots_adjust(hspace=0.075)
    
    plt.show()


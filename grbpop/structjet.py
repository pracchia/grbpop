import numpy as np

def ell(th,theta_pop):
    if theta_pop['jetmodel']=='smooth double power law':
        return (1.+(th/theta_pop['thc'])**4.)**(-theta_pop['a_L']/4.)*(1.+(th/theta_pop['thw'])**4)**(-(theta_pop['b_L']-theta_pop['a_L'])/4.)
    elif theta_pop['jetmodel']=='power law':
        return np.minimum(1.,(th/theta_pop['thc'])**(-theta_pop['a_L']))
    elif theta_pop['jetmodel']=='gaussian':
        return np.exp(-0.5*(th/theta_pop['thc'])**2)
    else:
        return np.zeros_like(th)    
    
def eta(th,theta_pop):
    
    if 'thc_Ep' in theta_pop.keys():
        thc = theta_pop['thc_Ep']
    else:
        thc = theta_pop['thc']
    
    if theta_pop['jetmodel']=='smooth double power law':
        if 'thw_Ep' in theta_pop.keys():
            thw = theta_pop['thw_Ep']
        else:
            thw = theta_pop['thw']
    
        return (1.+(th/thc)**4)**(-theta_pop['a_Ep']/4.)*(1.+(th/thw)**4)**(-(theta_pop['b_Ep']-theta_pop['a_Ep'])/4.)
    
    elif theta_pop['jetmodel']=='power law':
        return (1.+(th/thc)**2)**(-theta_pop['a_Ep']/2)
        
    elif theta_pop['jetmodel']=='gaussian':
        return np.exp(-0.5*(th/thc)**2)
    else:
        return 1e-10*np.ones_like(th)

import numpy as np
import matplotlib.pyplot as plt


## creating mock data
N = 150
# parameters of the model
a = 2.0
b = 1.0
x = np.random.uniform(low=0.0,high= 10, size = N)
y = a*x+b
## errors are defined to be a random percentage of y
s = np.random.uniform(low=0.01,high= 0.2, size = N) * y
data = np.array(([np.array([x[i],y[i],s[i]]) for i in range(N)]))

def lnhood(pars):
    '''
    Loglikelihood function for a linear model 
    
    Args:
        pars: 
            vector of free parameters: array
    Returns:
            likelihood value: scalar
    '''
    a = pars[0]
    b = pars[1]
    x = data[:,0]
    y = data[:,1]
    s = data[:,2]
    logl = -0.5*np.sum(((a*x + b-y)/s)**2)
    return logl

def mh_sampler(logL, propdist,init_smpl,nsteps):
    '''
    Metropolis-Hastings MCMC sampler. 
    .....
    Returns: a tuple (chain,lhood,a)
            chain: np.ndarray of shape (nsteps, nparams)
            lnL_vals: np.array of length nsteps
            a: integer
    '''
    #checks if input is in proper form
    # assert len(init_smpl) == len(parlims), "init_sample len and parlims len mismatch"
    assert isinstance(nsteps,int) and nsteps > 0, "Nsteps should be positive integer"
    pdim = len(init_smpl)
    #Initialization:
    x_0 = init_smpl
    chain = np.zeros((nsteps,pdim))
    lnlvals = np.zeros((nsteps,))
    count = 0
    for i in range(nsteps):
        x_prop = propdist(x_0)
        loga = logL(x_prop) - logL(x_0)
        a = np.exp(loga)
        u = np.random.uniform(low = 0.0, high = 1.0)
        if u <= a:
            count += 1
            x_1 = x_prop
        else:
            x_1 = x_0
        chain[i] = x_1
        x_0 = x_1
    a = count/nsteps
    return chain, lnlvals, a

def main():
    '''
    Main function, used to test the sampler, 
    '''
    #definition of sampler parameters
    nsteps = 2500
    s = np.array([0.05,0.03])
    mu = np.array([1.0,1.0])
    # a Gaussian proposed distribution arround point x_0 with mean, mu, and s the error  
    propdist = lambda x_0: np.random.normal(mu*x_0,s)
    init_smpl = np.array([0.1,0.3])
    chain, lnlvals, a = mh_sampler(lnhood,propdist,init_smpl,nsteps)
    
    print('acc_ratio = ', a)
    print('a := ', np.median(chain[:,0][-100:]))
    print('b := ', np.median(chain[:,1][-100:]))
    bf = chain[np.argmin(lnlvals)]
    #### Ploting the chains ###
    ns = np.arange(nsteps)
    plt.plot(ns,chain[:,0])
    plt.plot(ns,chain[:,1])
    plt.show()


if __name__ == '__main__':
    main()
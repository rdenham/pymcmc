### More detailed examples are available in
### when installing the pymcmc package
### see the directory $PREFIX/pymcmc/examples/
### where PREFIX is where the package was
### installed (eg /usr/local/lib/python2.6/dist-packages/)
### More detailed examples are available in
### when installing the pymcmc package
### see the directory $PREFIX/pymcmc/examples/
### where PREFIX is where the package was
### installed (eg /usr/local/lib/python2.6/dist-packages/)

### Empirical illustrations ###
### Example 3: First order
###            autoregressive regression


from numpy import random, ones, zeros, dot, hstack, eye, log
from scipy import sparse
from pysparse import spmatrix
from pymcmc.mcmc import MCMC, SliceSampler, RWMH, OBMC, MH, CFsampler
from pymcmc.regtools import BayesRegression 

def simdata(nobs, kreg):
    """function simulates data from a first order autoregressive regression"""
    xmat = hstack((ones((nobs, 1)), random.randn(nobs, kreg - 1)))
    beta = random.randn(kreg)
    sig = 0.2
    rho = 0.90
    yvec = zeros(nobs)
    eps = zeros(nobs)
    eps[0] = sig**2/(1.-rho**2)
    for i in xrange(nobs - 1):
        eps[i + 1] = rho * eps[i] + sig * random.randn(1)
    yvec = dot(xmat, beta) + eps
    return yvec, xmat

def calcweighted(store):
    """re - weights yvec and xmat, for use in weighted least squares regression"""
    nobs = store['yvec'].shape[0]
    store['Upper'].put(-store['rho'], range(0, nobs - 1), range(1, nobs))
    store['Upper'].matvec(store['yvec'], store['yvectil'])
    for i in xrange(store['xmat'].shape[1]):
        store['Upper'].matvec(store['xmat'][:, i], store['xmattil'][:, i])

def WLS(store):
    """computes weighted least square regression"""
    calcweighted(store)
    store['regsampler'].update_yvec(store['yvectil'])
    store['regsampler'].update_xmat(store['xmattil'])
    return store['regsampler'].sample()


def loglike(store):
    """
    calculates log - likelihood for the the first
    order autoregressive regression model
    """
    nobs = store['yvec'].shape[0]
    calcweighted(store)
    store['regsampler'].update_yvec(store['yvectil'])
    store['regsampler'].update_xmat(store['xmattil'])
    return store['regsampler'].loglike(store['sigma'], store['beta'])

def prior_rho(store):
    """
    evaulates the log of the prior distribution
    for rho. the beta distribution is used
    """
    if store['rho'] > 0. and store['rho'] < 1.0:
        alpha = 1.0
        beta = 1.0
        return (alpha - 1.) * log(store['rho']) + (beta - 1.) * \
               log(1.-store['rho'])
    else:
        return -1E256

def post_rho(store):
    """
    evaulates the log of the posterior distrbution for rho
    """
    return loglike(store) + prior_rho(store)


# Main program
random.seed(12345)
nobs = 1000
kreg = 3
yvec, xmat = simdata(nobs, kreg)

# we use a g - prior for the regression coefficients.
priorreg = ('g_prior', zeros(kreg), 1000.0)
regs = BayesRegression(yvec, xmat, prior = priorreg)

"""A dictionary is set up. The contents of the dictionary will be
available for use for by the functions that make up the MCMC sampler.
Note that we pass in storage space as well as the class intance used
to sample the regression from."""
data ={'yvec':yvec, 'xmat':xmat, 'regsampler':regs}
U = spmatrix.ll_mat(nobs, nobs, 2 * nobs - 1)
U.put(1.0, range(0, nobs), range(0, nobs))
data['yvectil'] = zeros(nobs)
data['xmattil'] = zeros((nobs, kreg))
data['Upper'] = U

# Use Bayesian regression to initialise MCMC sampler
bayesreg = BayesRegression(yvec, xmat)
sig, beta = bayesreg.posterior_mean()

simsigbeta = CFsampler(WLS, [sig, beta], ['sigma', 'beta'])

rho = 0.9
simrho = SliceSampler([post_rho], 0.1, 5, rho, 'rho')
blocks = [simrho, simsigbeta]

loglikeinfo = (loglike, kreg + 2, 'yvec')
ms = MCMC(10000, 2000, data, blocks, loglike = loglikeinfo)
ms.sampler()

ms.output()
ms.plot('rho')

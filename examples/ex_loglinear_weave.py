# bayesian MCMC estimation of the log - linear model

import os
from numpy import random, loadtxt, hstack, ones, dot, exp, zeros, outer, diag
from numpy import linalg
from pymcmc.mcmc import MCMC, RWMH, OBMC
from pymcmc.regtools import BayesRegression
from scipy.optimize.minpack import leastsq
from scipy import weave

## get the path for the data,
## If this was installed using setup.py
## it will be in the data directory of the
## module
import pymcmc
datadir = os.path.join(os.path.dirname(pymcmc.__file__),
                       'data')


def minfunc(beta, yvec, xmat ):
    """function used by nonlinear least squares routine"""
    return yvec - exp(dot(xmat, beta))


def prior(store):
    """function evaluates the prior pdf for beta"""
    mu = zeros(store['beta'].shape[0])
    Prec = diag(0.005 * ones(store['beta'].shape[0]))
    return -0.5 * dot(store['beta'].transpose(), dot(Prec, store['beta']))

def logl(store):
    """function evaluates the log - likelihood for the log - linear model"""

    code = """
    double sum = 0.0, xbeta;
    for(int i=0; i<nobs; i++){
    xbeta = 0.0;
        for(int j=0; j<kreg; j++){xbeta += xmat[i+j*kreg] * beta[j];}
        sum += yvec[i] * xbeta - exp(xbeta);
    }
    return_val = sum;
    """

    yvec = store['yvec']
    xmat = store['xmat']
    nobs, kreg = xmat.shape
    beta = store['beta']
    return weave.inline(code,['yvec','xmat', 'beta','nobs','kreg'],\
                        compiler='gcc')

    #for i in xrange(store['yvec'].shape[0]):
    #    xbeta=dot(store['xmat'][i,:],store['beta'])
    #    suml=suml+store['yvec'][i] * xbeta - exp(xbeta)
    #return suml

def posterior(store):
    """function evaluates the posterior probability for the log - linear model"""
    return logl(store) + prior(store)

def llhessian(store, beta):
    """function returns the hessian for the log - linear model"""
    nobs = store['yvec'].shape[0]
    kreg = store['xmat'].shape[1]
    lamb = exp(dot(store['xmat'], beta))
    sum = zeros((kreg, kreg))
    for i in xrange(nobs):
        sum = sum + lamb[i] * outer(store['xmat'][i], store['xmat'][i])
    return -sum

#functions required by independent Metropolis Hastings
def candidate(store):
    return random.multivariate_normal(store['betamean'], store['betavar'])

def candprob(store):
    res = store['beta'] - store['betamean']
    return -0.5 * dot(res, dot(store['betaprec'], res))


# main program
random.seed(12345)       # seed or the random number generator

data = loadtxt(os.path.join(datadir,'count.txt'),
               skiprows = 1)    # loads data from file
yvec = data[:, 0]
xmat = data[:, 1:data.shape[1]]
xmat = hstack([ones((data.shape[0], 1)), xmat])

data ={'yvec':yvec, 'xmat':xmat} 
bayesreg = BayesRegression(yvec, xmat)     # use bayesian regression to initialise
                                        # nonlinear least squares algorithm
sig, beta0 = bayesreg.posterior_mean()
init_beta, info = leastsq(minfunc, beta0, args = (yvec, xmat))
data['betaprec'] =-llhessian(data, init_beta)
scale = linalg.inv(data['betaprec'])

# indmh = IndMH(candidate, posterior, candprob, init_beta, 'beta')
samplebeta = RWMH(posterior, scale, init_beta, 'beta')
#samplebeta = OBMC(posterior,3, scale, init_beta, 'beta')
ms = MCMC(20000, 4000, data, [samplebeta], loglike = (logl, xmat.shape[1], 'yvec'))
ms.sampler()
ms.output(filename='example1c_loop.out') 

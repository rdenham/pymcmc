# bayesian MCMC estimation of the log - linear model

import os
from numpy import random, loadtxt, hstack, ones, dot, exp, zeros, outer, diag
from numpy import linalg, array, asfortranarray
from pymcmc.mcmc import Gibbs, RWMH, OBMC
from pymcmc.regtools import BayesRegression
from scipy.optimize.minpack import leastsq
import loglinear

""" Get the path for the data. If this was installed using setup.py
 it will be in the data directory of the module"""
datadir = os.path.join(os.path.dirname(pymcmc.__file__), 'data')

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
    loglike=0.0
    return loglinear.logl(store['xb'],store['xmatf'], store['beta'],store['yvec'],loglike)

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

data = loadtxt(os.path.join(datadir,'count.txt'), skiprows = 1)
yvec = data[:, 0]
xmat = data[:, 1:data.shape[1]]
xmat = hstack([ones((data.shape[0], 1)), xmat])

data ={'yvec':yvec, 'xmat':xmat} 
data['xb'] = zeros(yvec.shape[0])
data['xmatf']=asfortranarray(xmat)
bayesreg = BayesRegression(yvec, xmat)     # use bayesian regression to initialise
                                        # nonlinear least squares algorithm
sig, beta0 = bayesreg.posterior_mean()
init_beta, info = leastsq(minfunc, beta0, args = (yvec, xmat))
data['betaprec'] =-llhessian(data, init_beta)
scale = linalg.inv(data['betaprec'])

# indmh = IndMH(candidate, posterior, candprob, init_beta, 'beta')
samplebeta = RWMH(posterior, scale, init_beta, 'beta')
#samplebeta = OBMC(posterior,3, scale, init_beta, 'beta')
GS = Gibbs(20000, 4000, data, [samplebeta], loglike = (logl, xmat.shape[1], 'yvec'))
GS.sampler()
GS.output(filename='example1c_loop.out') 

#!/usr/bin/env python
# bayesian MCMC estimation of the log - linear model

import os
from numpy import random, loadtxt, hstack, ones, dot, exp, zeros, outer, diag
from numpy import linalg
from pymcmc.mcmc import MCMC, RWMH, OBMC
from pymcmc.regtools import BayesRegression
from scipy.optimize.minpack import leastsq
import pymcmc

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
    xbeta = dot(store['xmat'], store['beta'])
    lamb = exp(xbeta)
    return sum(store['yvec'] * xbeta - lamb)

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

# main program
random.seed(12345)       # seed or the random number generator

data = loadtxt(os.path.join(datadir,'count.txt'), skiprows = 1)    # loads data from file
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

samplebeta = RWMH(posterior, scale, init_beta, 'beta')
ms = MCMC(20000, 4000, data, [samplebeta], loglike = (logl, xmat.shape[1], 'yvec'))
ms.sampler()
ms.output(filename='example1c.out') 
ms.plot('beta', filename='ex_loglinear.pdf')
# ms.CODAoutput('beta')
# ms.plot('beta', elements = [0], plottypes ="trace", filename ="xx.pdf")
# ms.plot('beta', elements = [0], plottypes ="density", filename ="xx.png")
## ms.plot('beta', elements = [0], plottypes ="acf", filename ="yy.ps")

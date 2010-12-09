# example code for variable selection in regression

import os
from numpy import loadtxt, hstack, ones, random, zeros, asfortranarray, log
from pymcmc.mcmc import MCMC, CFsampler
from pymcmc.regtools import StochasticSearch, BayesRegression
import pymcmc

""" get the path for the data. If this was installed using setup.py
it will be in the data directory of the module"""
datadir = os.path.join(os.path.dirname(pymcmc.__file__),'data')

def samplegamma(store):
    """function that samples vector of indicators"""
    return store['SS'].sample_gamma(store)

# main program
random.seed(12346)

# loads data
data = loadtxt(os.path.join(datadir,'yld2.txt'))
yvec = data[:, 0]
xmat = data[:, 1:20]
xmat = hstack([ones((xmat.shape[0], 1)), xmat])

"""data is a dictionary whose elements are accessible from the functions
in the MCMC sampler"""
data ={'yvec':yvec, 'xmat':xmat}
prior = ['g_prior',zeros(xmat.shape[1]), 100.]
SSVS = StochasticSearch(yvec, xmat, prior);
data['SS'] = SSVS

"""initialise gamma"""
initgamma = zeros(xmat.shape[1], dtype ='i')
initgamma[0] = 1
simgam = CFsampler(samplegamma, initgamma, 'gamma', store ='none')

# initialise class for MCMC samper
ms = MCMC(20000, 5000, data, [simgam])
ms.sampler()
ms.output(filename ='vs.txt')
ms.output(custom = SSVS.output, filename = 'SSVS.out')
ms.output(custom = SSVS.output)

txmat = SSVS.extract_regressors(0)
g_prior = ['g_prior', 0.0, 100.]
breg = BayesRegression(yvec,txmat,prior = g_prior)
breg.output(filename = 'SSVS1.out')
breg.plot()


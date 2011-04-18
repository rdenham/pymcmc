### More detailed examples are available in
### when installing the pymcmc package
### see the directory $PREFIX/pymcmc/examples/
### where PREFIX is where the package was
### installed (eg /usr/local/lib/python2.6/dist-packages/)

### Empirical illustrations ###
### Example 1: Linear regression model:
###            Variable selection and estimation
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
simgam = CFsampler(samplegamma, initgamma, 'gamma', store ='all')


# initialise class for MCMC samper
random.seed(12346)
ms = MCMC(20000, 5000, data, [simgam])
ms.sampler()
ms.output()
ms.output(custom = SSVS.output)

txmat = SSVS.extract_regressors(0)
g_prior = ['g_prior', 0.0, 100.]
breg = BayesRegression(yvec,txmat,prior = g_prior)
breg.output()

breg.plot()

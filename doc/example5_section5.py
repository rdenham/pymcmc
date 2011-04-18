### More detailed examples are available in
### when installing the pymcmc package
### see the directory $PREFIX/pymcmc/examples/
### where PREFIX is where the package was
### installed (eg /usr/local/lib/python2.6/dist-packages/)

### PyMCMC interacting with R
import os
from numpy import random, loadtxt, hstack, ones, dot, exp, zeros, outer, diag
from numpy import linalg, array
from pymcmc.mcmc import MCMC, RWMH, OBMC
from pymcmc.regtools import BayesRegression
from scipy.optimize.minpack import leastsq

import rpy2.robjects as robjects


import pymcmc


""" get the path for the data. If this was installed using setup.py
it will be in the data directory of the module"""
datadir = os.path.join(os.path.dirname(pymcmc.__file__),'data')

def minfunc(beta, yvec, xmat ):
    """function used by nonlinear least squares routine"""
    return yvec - exp(dot(xmat, beta))

def prior(store):
    """function evaluates the prior pdf for beta"""
    mu = zeros(store['beta'].shape[0])
    Prec = diag(0.005 * ones(store['beta'].shape[0]))
    return -0.5 * dot(store['beta'].transpose(), dot(Prec, store['beta']))

def logl(store):
    """
    function evaluates the log - likelihood
    for the log - linear model
    """
    xbeta = dot(store['xmat'], store['beta'])
    lamb = exp(xbeta)
    return sum(store['yvec'] * xbeta - lamb)

def posterior(store):
    """
    function evaluates the posterior probability
    for the log - linear model
    """
    return logl(store) + prior(store)



def initial_values(yvec,xmat):
    ry = robjects.FloatVector(yvec)
    rv = robjects.FloatVector(xmat[:,1:].flatten())
    rx = robjects.r['matrix'](rv, nrow=xmat.shape[0],
                              byrow=True)
    robjects.globalenv['y'] = ry
    robjects.globalenv['x'] = rx
    mod = robjects.r.glm("y~x", family="poisson")
    init_beta =  array(robjects.r.coefficients(mod))
    modsummary = robjects.r.summary(mod)
    scale = array(modsummary.rx2('cov.unscaled'))
    return init_beta,scale

def ptitle(mystr,ncol=80):
    '''
    Little function for printing a header
    to separate the output a bit.
    '''
    ##force mystr to be a list
    if type(mystr) == type('a string'):
        mystr = [(mystr)]
    print '{0:#^{1}}'.format('',ncol)
    print '##{0:^{1}}##'.format("",ncol-4)
    for element in mystr:
        print '##{0:^{1}}##'.format(element,ncol-4)
    print '##{0:^{1}}##'.format("",ncol-4)
    print '{0:#^{1}}'.format('',ncol)

intro = '''This example uses R via Rpy2 to set    
the initial values. It also writes the output   
in CODA format, suitable for reading in with R.
You might like to run example5_section5.R from R
to see how this works.                           '''

print
ptitle(intro.split('\n'))


# main program
random.seed(12345)       # seed or the random number generator

# loads data from file
data = loadtxt(os.path.join(datadir,'count.txt'), skiprows = 1)    
yvec = data[:, 0]
xmat = data[:, 1:data.shape[1]]
xmat = hstack([ones((data.shape[0], 1)), xmat])

data ={'yvec':yvec, 'xmat':xmat}

# use R to initialise:
init_beta,scale=initial_values(yvec,xmat)

# Initialise the random walk MH algorithm
samplebeta = RWMH(posterior, scale, init_beta, 'beta')
ms = MCMC(20000, 4000, data, [samplebeta],
          loglike = (logl, xmat.shape[1], 'yvec'))
ms.sampler()
ms.output()
ms.CODAoutput(filename="loglinear_eg") 


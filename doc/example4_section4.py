## Using PyMCMC efficiently

## we use the same program as for example2
## but replace logl function:
import os
from numpy import random, loadtxt, hstack
from numpy import ones, dot, exp, zeros, outer, diag
from numpy import linalg, asfortranarray
from pymcmc.mcmc import MCMC, RWMH, OBMC
from pymcmc.regtools import BayesRegression
from scipy.optimize.minpack import leastsq
from scipy import weave
from scipy.weave import converters
import loglinear
import pymcmc

datadir = os.path.join(os.path.dirname(pymcmc.__file__),'data')

def minfunc(beta, yvec, xmat ):
    """function used by nonlinear least squares routine"""
    return yvec - exp(dot(xmat, beta))

def prior(store):
    """function evaluates the prior pdf for beta"""
    mu = zeros(store['beta'].shape[0])
    Prec = diag(0.005 * ones(store['beta'].shape[0]))
    return -0.5 * dot(store['beta'].transpose(), dot(Prec, store['beta']))


def posterior(store):
    """
    function evaluates the posterior probability
    for the log - linear model
    """
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

## Here we demonstrate four different versions of the
## loglikelihood function.

# Numpy
def loglnumpy(store):
    """function evaluates the log - likelihood for the log - linear model"""
    xbeta = dot(store['xmat'], store['beta'])
    lamb = exp(xbeta)
    return sum(store['yvec'] * xbeta - lamb)

# Loop
def loglloop(store):
    """function evaluates the log - likelihood for the log - linear model"""
    suml=0.0
    for i in xrange(store['yvec'].shape[0]):
        xbeta=dot(store['xmat'][i,:],store['beta'])
        suml=suml+store['yvec'][i] * xbeta - exp(xbeta)
    return suml

# weave
def loglweave(store):
    """function evaluates the log - likelihood for the log - linear model"""
    code = """
    double sum = 0.0, xbeta;
    for(int i=0; i<nobs; i++){
    xbeta = 0.0;
        for(int j=0; j<kreg; j++){
          xbeta += xmat(i,j) * beta(j);
        }
        sum += yvec(i) * xbeta - exp(xbeta);
    }
    return_val = sum;
    """
    yvec = store['yvec']
    xmat = store['xmat']
    nobs, kreg = xmat.shape
    beta = store['beta']
    val = weave.inline(code,['yvec','xmat', 'beta','nobs','kreg'],
                        compiler='gcc',
                       type_converters=converters.blitz
                       )
    return val

#f2py
def loglf2py(store):
    """function evaluates the log - likelihood for the log - linear model"""
    loglike=0.0
    return loglinear.logl(store['xb'],store['xmatf'], store['beta'],store['yvec'],loglike)

# loads data from file
data = loadtxt(os.path.join(datadir,'count.txt'), skiprows = 1)    
yvec = data[:, 0]
xmat = data[:, 1:data.shape[1]]
xmat = hstack([ones((data.shape[0], 1)), xmat])

data ={'yvec':yvec, 'xmat':xmat}

# use bayesian regression to initialise
bayesreg = BayesRegression(yvec, xmat)     
sig, beta0 = bayesreg.posterior_mean()

init_beta, info = leastsq(minfunc, beta0, args = (yvec, xmat))
data['betaprec'] =-llhessian(data, init_beta)
scale = linalg.inv(data['betaprec'])

# Initialise the random walk MH algorithm

samplebeta = RWMH(posterior, scale, init_beta, 'beta')

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

print
intro = '''This example shows four different ways of
programming the likelihood function. Three of these
ways are efficient, using Numpy, weave or f2py. The
fourth uses looping in python, and is thus much slower.'''

ptitle(intro.split('\n'))

print
print

ptitle("Numpy")
logl = loglnumpy
random.seed(12345)
ms = MCMC(20000, 4000, data, [samplebeta],
          loglike = (logl, xmat.shape[1], 'yvec'))
ms.sampler()
ms.output()

print

ptitle("loop (the slow one)")
logl = loglloop
random.seed(12345)
samplebeta = RWMH(posterior, scale, init_beta, 'beta')
ms = MCMC(20000, 4000, data, [samplebeta],
          loglike = (logl, xmat.shape[1], 'yvec'))
ms.sampler()
ms.output()

print
ptitle("weave")
logl = loglweave
random.seed(12345)       # seed or the random number generator
samplebeta = RWMH(posterior, scale, init_beta, 'beta')
data ={'yvec':yvec, 'xmat':xmat}
ms = MCMC(20000, 4000, data, [samplebeta],
          loglike = (logl, xmat.shape[1], 'yvec'))
ms.sampler()
ms.output()

print
f2pystring = '''
Now using f2py, this will need compiled code:    
try something like:                              
  f2py -c loglinear.f -m loglinear -lblas -latlas
or for non-standard libraries:                   
  f2py -c loglinear.f -m loglinear \             
       -L/opt/sw/fw/rsc/atlas/3.9.25//lib/ \     
       -latlas -lf77blas -lcblas                 
'''
ptitle(f2pystring.split('\n'))

logl = loglf2py
data['xb'] = zeros(yvec.shape[0])
data['xmatf']=asfortranarray(xmat)
random.seed(12345)       # seed or the random number generator
samplebeta = RWMH(posterior, scale, init_beta, 'beta')
ms = MCMC(20000, 4000, data, [samplebeta],
          loglike = (logl, xmat.shape[1], 'yvec'))
ms.sampler()
ms.output()



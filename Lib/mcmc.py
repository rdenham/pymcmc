#Main Algorithms for PyMCMC - A Python package for Bayesian estimation
#Copyright (C) 2010  Chris Strickland

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.# MCMC routines

import warnings
warnings.filterwarnings('ignore', '.*')

import os
import sys
import types
import time

import scipy as sp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from mcmcplots import *
from mcmc_utilities import *

class Attributes:
    def __init__(self, init_theta, name):
        self.ltheta = init_theta
        self.calculatestats = False
        if type(init_theta) == types.FloatType or type(init_theta) == types.IntType or \
            type(init_theta) == np.float64:
            self.nparam = [1]
            self.mean_theta = 0.
            self.var_theta = 0.
        elif type(init_theta) == np.ndarray:
            self.nparam = list(init_theta.shape)
            if init_theta.ndim == 1:
                self.mean_theta = np.zeros(init_theta.shape[0])
                self.var_theta = np.zeros(init_theta.shape[0])
            elif init_theta.ndim == 2:
                self.mean_theta = np.zeros(init_theta.shape)
                self.var_theta = np.zeros(init_theta.shape)
            else:
                print "Error"
        else:
            print "error", name

        self.name = name
        self.transformed = {}
        self.update_stats = self.__update_stats_std


    def get_nparam(self):
        """returns the number of parameters being sampled in the block"""
        return self.nparam

    def get_name(self):
        """returns the name of the parameters being sampled in the block"""
        return self.name

    def __update_stats_std(self):
        """used to update the posterior mean and variance at each iteration"""
        self.mean_theta = self.mean_theta + self.ltheta
        self.var_theta = self.var_theta + self.ltheta**2

    def __update_stats_transformed(self):
        theta = self.transformed[self.name]
        self.mean_theta = self.mean_theta + theta
        self.var_theta = self.var_theta + theta**2

    def update_transformed(self, transformed):
        self.transformed = transformed

    def use_transformed(self):
        self.update_stats = self.__update_stats_transformed


    def calculate_stats(self, nit, burn):
        """Procedures cancules estimates of the marginal posterior mean and variance for
        the MCMC estimation. The function arguments are:
        nit - thet number of iterations
        burn - is the length of the burn in
        """
        self.mean_theta = self.mean_theta/float(nit - burn)
        self.var_theta = self.var_theta - float(nit - burn) * self.mean_theta**2
        self.var_theta = self.var_theta/float(nit - burn - 1)

    def get_stats(self, nit, burn):
        """Procedure returns estimates of the marginal posterior mean and variance for
        the MCMC estimation. The function arguments are:
        nit - thet number of iterations
        burn - is the length of the burn in
        """
        
        if self.calculatestats == False:
            self.calculate_stats(nit, burn)

        self.calculatestats = True
        return self.mean_theta, self.var_theta


class BaseSampler:
    """
    
    The base class for samplers used by class MCMC.

    Arguments:
      init_theta - is the initial value for the parameters of interest
      name - is the name of the parameters of interest
      kwargs - optional parameters:
        store - 'all'; (default) stores every iterate for parameter of
                interest
              - 'none'; do not store any of the iterates
        output - list; provide an index in the form of a list for the parameters to be
                       that output is to be provide for. If not provided print all of
                      theta               
        fixed_parameter - Is used if the user wants to fix the parameter
        value that is returned. This is used for testing. This is used for testing MCMC sampling schemes.
    """

    def __init__ (self, init_theta, name, **kwargs):
        if type(name) != type([]):
            self.attrib = Attributes(init_theta, name)
            self.mblock_ind = False
            self.number_groups = 1
            self.nparam = self.attrib.get_nparam()

        else:
            assert type(init_theta) == type([])
            assert len(init_theta) == len(name)
            if 'fixed_parameter' in kwargs:
                assert type(kwargs['fixed_parameter']) == type([]) 
                assert len(kwargs['fixed_parameter']) == len(name) 
            self.attrib = []
            self.nparam = []
            self.mblock_ind = True
            self.number_groups = len(name)
            self.ltheta = []
            for i in xrange(len(name)):
                self.attrib.append(Attributes(init_theta[i], name[i]))
                self.nparam.append(self.attrib[i].get_nparam())
                self.ltheta.append(init_theta[i])


        if 'fixed_parameter' in kwargs.keys():
            self.fixed_parameter = kwargs['fixed_parameter']
            self.sample = self.__sample_fixed_parameter
            self.update_ltheta(self.fixed_parameter)
            self.accept = 1
            self.count = 1
        else:
            self.sample = self.sampler
            self.accept = 0
            self.count = 0
           
        self.name = name
        
        if 'store' in kwargs.keys():
            self.store = kwargs['store']
            if self.store not in ['all', 'none']:
                self.store ='all'
        else:
            self.store ='all'

        if self.number_groups == 1:
            self.update_stats = self.__update_stats_ng
            self.update_transformed = self.__update_transformed

        else:
            self.update_stats = self.__update_stats_g
            self.update_transformed = self.__update_transformed_groups

        if 'index' in kwargs:
            self.index = kwargs['index']
        else:
            self.index = 0

    
    def get_ltheta(self):
        if self.number_groups == 1:
            return self.attrib.ltheta
        else:
            for i in xrange(self.number_groups):
                self.ltheta[i] = self.attrib[i].ltheta
            return self.ltheta

    def update_ltheta(self, ltheta):
        if self.number_groups == 1:
            self.attrib.ltheta = ltheta
        else:
            for i in xrange(self.number_groups):
                self.attrib[i].ltheta = ltheta[i]


    def get_number_groups(self):
        return self.number_groups
    
    def __sample_fixed_parameter(self, store):
        return self.fixed_parameter
        
    def acceptance_rate(self):
        """returns the acceptance rate for the MCMC sampler"""
        return float(self.accept)/self.count

    def get_nparam(self):
        """returns the number of parameters being sampled in the block"""
        return self.nparam

    def get_name(self):
        """returns the name of the parameters being sampled in the block"""
        return self.name

    def get_index(self):
        return self.index


    def __update_stats_ng(self):
        self.attrib.update_stats()

    def __update_stats_g(self):
        for i in xrange(self.number_groups):
            self.attrib[i].update_stats()

    def __update_transformed(self, transformed):
        self.attrib.update_transformed(transformed)

    def __update_transformed_groups(self, transformed):
        for i in xrange(self.number_groups):
            self.attrib[i].update_transformed(transformed)

    def use_transformed(self, names):
        if self.number_groups == 1:
            self.attrib.use_transformed()
        else:
            for i in xrange(self.number_groups):
                name = self.attrib[i].get_name()
                if name in names:
                    self.attrib[i].use_transformed()
            

    def calculate_stats(self, nit, burn):
        """Procedures cancules estimates of the marginal posterior mean and variance for
        the MCMC estimation. The function arguments are:
        nit - the number of iterations
        burn - is the length of the burn in
        """
        
        if self.mblock_ind == False:
            self.attrib.calculate_stats(nit, burn)

        else:
            for i in xrange(len(self.name)):
                self.attrib[i].calculate_stats(nit, burn)

    def get_stats(self, nit, burn):
        """Procedure returns estimates of the marginal posterior mean and variance for
        the MCMC estimation. The function arguments are:
        nit - thet number of iterations
        burn - is the length of the burn in
        """
        
        if self.mblock_ind == False:
            return self.attrib.get_stats(nit, burn)

        else:
            meanv = []
            varv = []
            for i in xrange(len(self.name)):
                meani, vari = self.attrib[i].get_stats(nit, burn)
                meanv.append(meani)
                varv.append(vari)

            return meanv, varv


   
    def get_store(self):
        return self.store



class CFsampler(BaseSampler):
    """
    CFsampler is used to sample from closed form solutions in the MCMC sampler.
    arguments:
    func - is a function that samples from the posterior distribution of interest
    init_theta - is an initial value for theta (parameters of
    interest)
    name - name of theta
    kwargs - optional parameters:
        store - 'all'; (default) stores every iterate for parameter of
                interest
              - 'none'; do not store any of the iterates
        output - list; provide an index in the form of a list for the parameters to be
                       that output is to be provide for. If not provided print all of
                       theta               
        additional_output - function that produces additional output.                
        fixed_parameter - Is used is the user wants to fix the parameter
        value that is returned. This is used for testing.
        """

    def __init__(self, func, init_theta, name, **kwargs):
        BaseSampler.__init__(self, init_theta, name, **kwargs)
        self.func = func
        self.accept = 1
        self.count = 1

    def sampler(self, store):
        """returns a sample from the defined sampler"""
        self.update_ltheta(self.func(store))
        return self.get_ltheta()
    

class SliceSampler(BaseSampler):
    """SliceSampler is a class that can be used for the slice sampler 
    func - k dimensitonal list containing log functions
    init_theta - float used to initialise slice sampler.
    ssize - is a user defined value for the typical slice size
    sN - is an integer limiting slice size to sN * ssize 
    **kwargs - optional arguments
        store - 'all'; (default) stores every iterate for parameter of
        interest
              - 'none'; do not store any of the iterates
        fixed_parameter - Is used is the user wants to fix the parameter
        value that is returned. This is used for testing.

    """
    def __init__(self, func, ssize, sN, init_theta, name, **kwargs):
        BaseSampler.__init__(self, init_theta, name, **kwargs)
        try:
            self.init_theta = float(init_theta)
            self.ssize = ssize
            self.accept = 1
            self.count = 1
            self.sN = sN
            if type(func) == type([]):
                self.func = func
                self.k = len(func)
            else:
                self.k = 1
                self.func = [func]
            self.omega = np.zeros(self.k)
        except TypeError:
            raise TypeError("Error: SliceSampler is only used to sample scalars")

    def sampler(self, store): 
        # self.omega = [np.exp(function(self.ltheta)) * np.random.rand(1, 1)[0] for function in func]
        for i in xrange(self.k):
            store[self.attrib.name] = self.get_ltheta()
            self.omega[i] = (self.func[i](store)) + np.log(np.random.rand(1)[0])
        bounds = np.array([self.__step_out(i, store) for i in xrange(self.k)])
        max_lower = bounds[:, 0].max()
        min_upper = bounds[:, 1].min()
        return self.__pick_by_shrink(max_lower, min_upper, store)

    def __pick_by_shrink(self, max_lower, min_upper, store):
        lower = max_lower
        upper = min_upper
        falselist = [False] * self.k
        tt = False
        i = 0
        while any(falselist) == False:
            falselist = [False] * self.k
            candtheta = lower + np.random.rand(1)[0] * (upper - lower)
            store[self.attrib.name] = candtheta
            # print exp(self.omega), [function(store) for function in self.func]
            tt = True
            i = 0
            while tt == True and i < self.k:
                if self.omega[i] < self.func[i](store):
                    falselist[i] = True
                    tt = True
                    i = i + 1
                else:
                    tt = False
                    falselist[i] = False
                    i = 0
                    if candtheta < self.get_ltheta():
                        lower = candtheta
                    else:
                        upper = candtheta

        self.update_ltheta(candtheta)
        return candtheta


    def __step_out(self, i, store):
        lower_bound = self.get_ltheta() - self.ssize * np.random.rand(1)[0]
        upper_bound = lower_bound + self.ssize
        J = np.floor(self.sN * np.random.rand(1)[0])
        Z = self.sN - 1-J
        store[self.attrib.name] = lower_bound
        while J > 0 and self.omega[i] < (self.func[i](store)):
            lower_bound = lower_bound - self.ssize
            J = J - 1
        store[self.attrib.name] = upper_bound
        while Z > 0 and self.omega[i] < (self.func[i](store)):
            upper_bound = upper_bound + self.ssize
            Z = Z - 1

        return lower_bound, upper_bound

class RWMH(BaseSampler):
    """This class is used for the random walk Metropolis Hastings. Argumemts:
    post - Is a user defined function for the log of full conditional
        posterior distribution for the parameters of interest
    csig - The scale parameter for the random walk MH algorithm.
    init_theta - The initial value for the parameter of interest
    name - the name of the parameter of interest
    kwargs - Optional arguments:
        store - 'all'; (default) stores every iterate for parameter of
                interest
              - 'none'; do not store any of the iterates 
        fixed_parameter - Is used is the user wants to fix the parameter
        value that is returned. This is used for testing.


    """


    def __init__(self, post, csigsq, init_theta, name, **kwargs):
        BaseSampler.__init__(self, init_theta, name, **kwargs)
        self.__updateSig = self.__update_sig_standard
        if type(csigsq) in [types.FloatType, np.float64]:
            self.theta = 0.0
            self.Sig = np.sqrt(csigsq)
            self.__sampletheta = self.__sampletheta_float
            if 'adaptive' in kwargs.keys():
                if kwargs['adaptive'] == 'GYS':
                    self.pstar = 0.44
                    self.pstarr =  (1. - self.pstar) / self.pstar
                    self.__updateSig = self.__update_sig_adaptive


        elif type(csigsq) == np.ndarray:
            self.theta = np.zeros(init_theta.shape[0])
            self.Sig = np.linalg.cholesky(csigsq)
            self.__sampletheta = self.__sampletheta_ndarray

            if 'adaptive' in kwargs.keys():
                if kwargs['adaptive'] == 'GYS':
                    self.pstar = 0.234
                    alpha = -sp.stats.norm.ppf(self.pstar / 2.)
                    self.m = csigsq.shape[0]
                    self. const = (1. - 1. / self.m) * np.sqrt(2. * np.pi) * np.exp(alpha**2/2.)/\
                            (2. * alpha) + 1. / (self.m * self.pstar * (1. - self.pstar))
                    self.thetabar = np.zeros(self.m)
                    self.Sigsq = csigsq
                    self.sig = np.mean(np.abs(init_theta)) / 10.
                    if self.sig == 0.0:
                        self.sig = 1.0
                    self.eye = np.eye(self.m)
                    self.__updateSig = self.__update_sig_adaptive_ndarray

        else:
            print "Error"

        self.post = post

        
    def sampler(self, store): # note theta is a dummy argument 
        self.count = self.count + 1.
        self.randomvec = np.random.randn(self.attrib.nparam[0])
        candtheta = self.__sampletheta(store)
        store[self.attrib.name] = candtheta
        lnpr = self.post(store)
        store[self.attrib.name] = self.get_ltheta()
        llnpr = self.post(store)
    
        alpha = np.exp(lnpr - llnpr)
        if np.random.rand(1) < alpha:
            self.update_ltheta(candtheta)
            self.accept = self.accept + 1.
            self.theta = candtheta
            self.__updateSig(True, store)
        else:
            self.__updateSig(False, store)
            self.theta = self.get_ltheta()

        return self.theta
    
    def __sampletheta_float(self, store):
        return store[self.attrib.name] + self.Sig * self.randomvec[0]

    def __update_sig_standard(self, accept, store):
        self.Sig = self.Sig

    def __update_sig_adaptive(self, accept, store):
        c = self.Sig * self.pstarr
        if accept == False:
            self.Sig = self.Sig - c * self.pstar / (store['iteration'] + 1)
        else:
            self.Sig = self.Sig + c * (1. - self.pstar) / (store['iteration'] + 1)

    def __update_sig_adaptive_ndarray(self, accept, store):
        it = store['iteration']
        if it > 200:
            self.__update_Sigsq_ndarray(it)
            self.__update_sigma_ndarray(accept, it)
            self.Sig = self.sig * np.linalg.cholesky(self.Sigsq + \
                                 (self.sig ** 2 / (it + 1)) * self.eye)


        elif it > 100:
            self.__update_Sigsq_ndarray(it)
            self.Sig = self.sig * np.linalg.cholesky(self.Sigsq + \
                                 (self.sig ** 2 / (it + 1)) * self.eye)

        else:
            self.Sig = self.Sig
            self.Sigsq = self.Sigsq + np.outer(self.theta, self.theta)
            self.thetabar = self.thetabar + self.theta
            if it == 100:
                self.Sigsq = self.Sigsq / 100.
                self.thetabar = self.thetabar / 100.

    def __update_sigma_ndarray(self, accept, it):
        c = self.sig * self.const
        if accept == True:
            self.sig = self.sig + c * (1. - self.pstar) / \
                    max([200., float(it) / self.m])
        else:
            self.sig = self.sig - c * self.pstar / max([200.,  float(it) / self.m])

    def __update_Sigsq_ndarray(self, it):
            lthetabar = self.thetabar.copy()
            self.thetabar =  1. / (it + 1.) * (it * self.thetabar +  self.theta)
            self.Sigsq = (it - 1.) / it * self.Sigsq + np.outer(lthetabar, lthetabar) -\
                    float(it + 1.) / it * np.outer(self.thetabar, self.thetabar) + \
                    1. / it * np.outer(self.theta, self.theta)



    def __sampletheta_ndarray(self, store):
        return store[self.attrib.name] + np.dot(self.Sig, self.randomvec)

class OBMC(BaseSampler):
    """This is a a class for the orientational bias Monte Carlo algorithm.
    The arguments:
    post - is a user defined function for the log of full conditional posterior
        distribution for the parameters of interest. 
    ntry - the number of candidates. A scalar.
    csig - the scale parameter it can be a float or a Numpy array.
    init_theta - The initial value for the parameter of interest. Scalar or
      1-d numpy array.

    kwargs - Optional arguments:
        store - 'all'; (default) stores every iterate for parameter of
                interest
              - 'none'; do not store any of the iterates 
        fixed_parameter - Is used is the user wants to fix the parameter
        value that is returned. This is used for testing.
    """


    def __init__(self, post, ntry, csig, init_theta, name, **kwargs):
        BaseSampler.__init__(self, init_theta, name, **kwargs)

        ##first type checking for ntry:
        if not np.isscalar(ntry):
            raise TypeError("ntry must be a scalar")

        self.post = post
        self.ntry = ntry
        try:
            if np.isscalar(init_theta):
                nelements = sum([ntry] + self.attrib.nparam)
            else:
                if init_theta.ndim > 1:
                    raise TypeError(
                        "init_theta must be either scalar or 1-d")
                nelements = [ntry] + self.attrib.nparam
            self.xtil = np.zeros(nelements)
            self.randnvec = np.zeros(nelements)
            self.multicand = np.zeros(nelements)
            self.multicandcum = np.zeros(ntry)

            self.multicandcump = np.zeros(ntry + 1)
            self.numvec = np.zeros(ntry)
            self.denomvec = np.zeros(ntry)
            if type(csig) == types.FloatType:
                self.CholCsig = np.sqrt(csig)
                self.__sampletheta = self.__sampletheta_float
            else:
                self.CholCsig = np.linalg.cholesky(csig)
                self.__sampletheta = self.__sampletheta_ndarray
            self.candtheta = np.zeros(self.attrib.nparam)

        except TypeError as e:
            print e
            raise TypeError("argument init_theta seems to be the wrong type")
        except Exception as e:
            print "unexpected error"
            print e
            raise


    def sampler(self, store):             
        self.count = self.count + 1
        self.randnvec = np.random.randn(self.ntry, self.attrib.nparam[0])
        # self.__sampletheta(self.ntry, self.multicand, self.numvec, self.ltheta, store)
        for i in xrange(self.ntry):
            self.multicand[i] = self.get_ltheta() + np.dot(self.CholCsig, self.randnvec[i])
            store[self.attrib.name] = self.multicand[i]
            self.numvec[i] = self.post(store)
        
        intconst = self.numvec.max()
        self.multicandcum = np.add.accumulate(np.exp(self.numvec - intconst))
        self.multicandum = self.multicandcum/self.multicandcum[self.ntry - 1]
        
        randu = np.random.rand(1)
        self.multicandcump[0:self.ntry] = self.multicandcum
        self.multicandcump[self.ntry] = randu
        self.multicandcump.sort()
        index = self.multicandcump.searchsorted(randu)
        self.candtheta = self.multicand[index[0]]    
        
        self.xtil[self.ntry - 1] = self.get_ltheta()
        store[self.attrib.name] = self.xtil[self.ntry - 1]
        self.denomvec[self.ntry - 1] = self.post(store)
        if self.ntry > 1:
            self.randnvec = np.random.randn(self.ntry - 1, self.attrib.nparam[0])
            # self.__sampletheta(self.ntry - 1, self.xtil, self.denomvec, self.candtheta, store)
            for i in xrange(self.ntry - 1):
                self.xtil[i] = self.candtheta + np.dot(self.CholCsig, self.randnvec[i])
                store[self.attrib.name] = self.xtil[i]
                self.denomvec[i] = self.post(store)

        # intconst = np.hstack((self.numvec, self.denomvec)).max()
        sumdenom = sum(np.exp(self.denomvec - intconst))
        sumnum = sum(np.exp(self.numvec - intconst))
        alpha = sumnum/sumdenom

        # print alpha, sumnum, sumdenom, self.candtheta, self.numvec, self.multicand
        # print alpha, sumnum, sumdenom, self.denomvec, intconst 
        if np.random.rand(1) < alpha:
            self.update_ltheta(self.candtheta)
            self.accept = self.accept + 1.
            return self.candtheta
        else:
            return self.get_ltheta()
    def __sampletheta_float(self, nt, lhs, lhs2, meanv, store):
        for i in xrange(nt):
            lhs[i] = meanv + self.CholCsig * self.randnvec[i, 0]
            store[self.attrib.name] = lhs[i]
            lhs2[i] = self.post(store)

    def __sampletheta_ndarray(self, nt, lhs, lhs2, meanv, store):
        meanvtmp = meanv.copy()
        for i in xrange(nt):
            lhs[i][:] = (meanvtmp + np.dot(self.CholCsig, self.randnvec[i]))[:]
            store[self.nattrib.ame][:] = lhs[i][:]
            lhs2[i] = self.post(store)
 

class MH(BaseSampler):
    """This class is used for the Metropolis Hastings algorithm. The function
    arguments are:
    func - Is a user defined function that returns a sample for the
    parameter of interest. 
    actualprob - Is a user defined function that returns the log probability of
                 the parameters of interest evaluated using the target  density.
    probcandgprev - Is a user defined function that returns the log probability
                    of the parameters of interest evaluated at the previous
                    iteration conditional on the candidate.
    init_theta - Initial value for the parameters of interest. 
    name - The name of the parameter of interest. 
    kwargs - optional parameters:
        store - 'all'; (default) stores every iterate for parameter of
                interest
              - 'none'; do not store any of the iterates
        output - list; provide an index in the form of a list for the parameters to be
                       that output is to be provide for. If not provided print all of
                       theta               
        fixed_parameter - Is used is the user wants to fix the parameter
        value that is returned. This is used for testing.
    """


    def __init__(self, func, actualprob, probcandgprev, probprevgcand, init_theta, name, **kwargs):
        BaseSampler.__init__(self, init_theta, name, **kwargs)

        self.func = func
        self.actualprob = actualprob
        self.probcandgprev = probcandgprev
        self.probprevgcand = probprevgcand
        if self.number_groups == 1:
            self.previous_name ='previous_' + self.name
        else:
            self.previous_name = ['previous_' + x for x in self.name]

        if type(init_theta) == types.FloatType or type(init_theta) == types.IntType:
            self.attrib.nparam = [1]
            self.candtheta = 0.
              
        elif type(init_theta) == np.ndarray:
            self.attrib.nparam = list(init_theta.shape)
            if init_theta.ndim == 1:
                self.candtheta = np.zeros(init_theta.shape[0])
            elif init_theta.ndim == 2:
                dim = list(init_theta.shape)
                self.candtheta = np.zeros(dim)
                dim.insert(0, len(self.range))

        if self.number_groups == 1:
            self.__set_store = self.__set_store_single
        else:
            self.__set_store = self.__set_store_multiple


    def sampler(self, store):
        self.count = self.count + 1.
        candtheta = self.func(store)
        #store[self.attrib.name] = self.get_ltheta()
        self.__set_store(store, self.name, self.get_ltheta())
        lnprprev = self.actualprob(store)
        #store[self.attrib.name] = candtheta
        self.__set_store(store, self.name, candtheta)
        lnprcand = self.actualprob(store)
        #store[self.previous_name] = self.get_ltheta()
        self.__set_store(store, self.previous_name, self.get_ltheta())
        llnpr = self.probprevgcand(store)
        llncand = self.probcandgprev(store)
    
        num = (lnprcand - lnprprev)
        denom = (llncand - llnpr)
        alpha = np.exp(num - denom)
                                 
        if np.random.rand(1) < alpha:
            self.update_ltheta(candtheta)
            self.accept = self.accept + 1.
            return candtheta
        else:
            return self.get_ltheta()

    def __set_store_single(self, store, name, theta):
        store[name] = theta

    def __set_store_multiple(self, store, name, theta):
        for i in xrange(self.number_groups):
            store[name[i]] = theta[i]

class IndMH(BaseSampler):
    """
    IndMH is for the independent Metropolis Hastings algorithm
    arguments:
    func - is a function that calculates the candidate for theta
    actualprob - Is a user defined function that returns the log
                 probability of the parameters of interest
                 evaluated using the target density.
    candprob - Is a user defined function that returns the log
               probability of the parameters of interest
               evaluated using the canditate density.
    init_theta - is the initial value for theta
    name - is the name of the parameter of interest
    kwargs - optional parameters:
        store - 'all'; (default) stores every iterate for parameter of
                interest
              - 'none'; do not store any of the iterates 
        fixed_parameter - Is used is the user wants to fix the parameter
        value that is returned. This is used for testing.
  """


    def __init__(self, func, actualprob, candprob, init_theta, name, **kwargs):
        BaseSampler.__init__(self, init_theta, name, **kwargs) 

        self.func = func
        self.actualprob = actualprob
        self.candprob = candprob
        
        if self.number_groups == 1:
            self.__set_store = self.__set_store_single
        else:
            self.__set_store = self.__set_store_multiple


    def sampler(self, store):
        self.count = self.count + 1.
        self.candtheta = self.func(store)
        #store[self.attrib.name] = self.candtheta
        self.__set_store(store, self.name, self.candtheta)

        lnpr = self.actualprob(store)
        lncand = self.candprob(store)
        #store[self.attrib.name] = self.get_ltheta()
        self.__set_store(store, self.name, self.get_ltheta())
        llnpr = self.actualprob(store)
        llncand = self.candprob(store)

        num = lnpr - llnpr
        denom = lncand - llncand
        alpha = np.exp(num - denom)
        if np.random.rand(1) < alpha:
            self.update_ltheta(self.candtheta)
            self.accept = self.accept + 1.
            return self.candtheta
        else:
            return self.get_ltheta()
        
    def __set_store_single(self, store, name, theta):
        store[name] = theta

    def __set_store_multiple(self, store, name, theta):
        for i in xrange(self.number_groups):
            store[name[i]] = theta[i]


"""Note data is a dictionary of the data""" 

class MCMC:
    """
    class for MCMC sampler. This class is initialised with:
    nit - the number of iterations
    burn - the burn in for the MCMC sampler
    data - A dictionary containing any data, functions or classes
           that may be required by any of the functions 
    blocks - a list containing functions that are used to sample from the
    full conditional posterior disitrbutions of interest
    kwargs - allows for optional arguments
        loglike - tuple containing a function that evaluates the
        log-likelihood, number of parameters in the likelihood and the name of
        the dataset. Eg: loglike = (loglike, nparam, 'yvec')
        
        transform - [function(store), [parameternames]]. function(store) is the
        function that is used to transform the parameters, while [parameternames]
        is a list of parameter names that are being transformed. Summary statistics
        will be re-calculated for each of the parameters in the list. All of the
        parameters used in the transformation are required to be stored.
    """
    def __init__(self, nit, burn, data, blocks, **kwargs):
        self.burn = burn
        self.nit = nit
        assert(type(blocks) == type([]) or type(blocks) == type(()))
        self.nblocks = len(blocks)
        self.storeparam = {}
        self.totaltime = 0
        self.keys = []
        self.all_keys = []
        self.currentparam = data
        self.storeblock = {}
        self.currentparam['iteration'] = 0
        self.currentparam['number_of_iterations'] = nit
        self.currentparam['length_of_burnin'] = burn
        self.currentparam['index'] = 0
        self.ngroups = []
        self.group_names = []
        self.name_group = {}
        self.update_param = {}
        self.not_stored_param = []
        self.current_transformed = {}

        for i in xrange(len(blocks)):
            self.group_names.append('group'+str(i))
            name = blocks[i].get_name()
            ngroups = blocks[i].get_number_groups()
            self.ngroups.append(ngroups)
            self.storeblock[self.group_names[i]] = blocks[i]
            if ngroups == 1:
                self.keys.append(name)
                self.all_keys.append(name)
                self.name_group[name] = self.group_names[i]
            else:
                for iname in name:
                    self.all_keys.append(iname)
                    self.name_group[iname] = self.group_names[i]

                for key in self.not_stored_param:
                    for j in xrange(ngroups):
                        self.keys.append(name[j])

            nparam = self.storeblock[self.group_names[i]].get_nparam()
            if ngroups == 1:
                if self.storeblock[self.group_names[i]].get_store() == 'all':
                    self.storeparam[name] = np.zeros([self.nit] + nparam)
                else:
                    self.not_stored_param.append(name)
                self.currentparam[name] = self.storeblock[self.group_names[i]].get_ltheta()

            else:
                ltheta = self.storeblock[self.group_names[i]].get_ltheta()
                for j in xrange(ngroups):
                    if self.storeblock[self.group_names[i]].get_store() == 'all':
                        self.storeparam[name[j]] = np.zeros([self.nit]+nparam[j])
                    self.currentparam[name[j]] = ltheta[j]

        self.numdec = 3
        self.meanstore = {}
        self.varstore = {}
        if 'loglike' in kwargs:
            if type(kwargs['loglike']) == type(()) and len(kwargs['loglike']) == 3:
                self.loglike = kwargs['loglike'][0]
                self.nparamlike = kwargs['loglike'][1]
                self.calcbic = True
                self.dataname = kwargs['loglike'][2]
                
            else:
                self.calcbic = False
                print "Warning; specification of tuple loglike in incorrect"
                print "Will not calculate BIC"
                
        else:
            self.calcbic = False

        #if 'transform' in kwargs:
        #    assert type(kwargs['transform']) == type([])
        #    assert len(kwargs['transform']) == 2
        #      
        #    self.transformfunc = kwargs['transform'][0]
        #    self.transform_list = kwargs['transform'][1]
        #    assert type(self.transform_list) == type([])
        #               
        #    self.transformfunc_ind = True
        #else:
        #    self.transformfunc_ind = False

        for key in self.all_keys:
            self.update_param[key] = self.__simple_update

        if 'transform' in kwargs: 
            assert type(kwargs['transform2']) == type({})
            self.transform2_ind = True
            self.transform2 = kwargs['transform2']
            for key in self.transform2:
                assert key in self.all_keys
                self.update_param[key] = self.__transform_update
            for key in self.not_stored_param:
                if key not in self.transform2:
                    self.not_stored_param.remove(key)
            #code 

            for i in xrange(self.nblocks):
                names = self.storeblock[self.group_names[i]].get_name()[:]
                if self.ngroups[i] == 1:
                    if names in self.transform2:
                        self.storeblock[self.group_names[i]].use_transformed(names)

                else:
                    for name in names:
                        if name not in self.transform2:
                            names.remove(name)
                    if len(names) > 0:
                        self.storeblock[self.group_names[i]].use_transformed(names)
        else:
            self.transform2_ind = False



    def sampler(self):
        """Runs the MCMC sampler"""
        starttime = time.clock()

        for it in xrange(self.nit):
            self.__scan_blocks(it)

        self.totaltime = time.clock() - starttime
        #if self.transformfunc_ind == True:
        #    try:
        #        self.transformfunc(self.storeparam)
        #    except:
        #        print "Could not transform iterates, specified parameter was not stored"

    def __scan_blocks(self, it):
        self.currentparam['iteration'] = it
        for i in xrange(self.nblocks):
            self.currentparam['index'] = self.storeblock[self.group_names[i]].get_index()
            sample = self.storeblock[self.group_names[i]].sample(self.currentparam)
            name = self.storeblock[self.group_names[i]].get_name()
            if self.ngroups[i] == 1:
                self.currentparam[name] = sample
                if self.storeblock[self.group_names[i]].get_store() == 'all':
                    self.storeparam[name][it] = self.update_param[name](name)
            else:
                for j in xrange(self.ngroups[i]):
                    self.currentparam[name[j]] = sample[j]
                                          
                if self.storeblock[self.group_names[i]].get_store() == 'all':
                    for j in xrange(self.ngroups[i]):
		        try:
                       	    self.storeparam[name[j]][it] = self.update_param[name[j]](name[j])
                        except ValueError as err:
                            print "ERROR, problem with %s" % name[j]
                            print "\t%s has shape" % name[j],self.storeparam[name[j]][it].shape
                            print "\tupdate_param has shape", self.update_param[name[j]](name[j]).shape
                            print err
                            sys.exit(1)
            if it >= self.burn:
                if self.transform2_ind == True:
                    self.__update_transformed_not_stored()
                    if self.ngroups[i] == 1:
                        if name in self.transform2:
                            self.storeblock[self.group_names[i]].\
                                    update_transformed(self.current_transformed)
                    else:
                        if self.__eitherstored(name, self.transform2):
                            self.storeblock[self.group_names[i]].\
                                    update_transformed(self.current_transformed)

                self.storeblock[self.group_names[i]].update_stats()

    def __eitherstored(self, names, transform2):
        for name in names:
            if name in transform2:
                return True
        return False

    def __simple_update(self, name):
        return self.currentparam[name]

    def __transform_update(self, name):
        theta = self.transform2[name](self.currentparam)
        self.current_transformed[name] = theta
        return theta

    def __update_transformed_not_stored(self):
        for name in self.not_stored_param:
            self.current_transformed[name] = self.transform2[name](self.currentparam)

    def return_array(self, arr):
        if arr.shape[0] > 1:
            return arr
        else:
            return arr[0]
    
    def get_parameter(self, name):
        """Returns an array of the parameter iterates including burnin"""
        try:
            return self.storeparam[name]
        except KeyError as e:
            print e
            raise KeyError("ERROR: %s has not been stored!!" % name)
   
    def get_parameter_exburn(self, name):
        """Returns an array of the parameter iterates excluding burnin"""
        try:
            return self.storeparam[name][self.burn:self.nit, :]
        except KeyError as e:
            print e
            raise KeyError("ERROR: %s has not been stored!!" % name)
   
   
    def get_mean_cov(self, listname):
        """returns the posterior covariance matrix for the parameters named
        in listname"""
        assert(type(listname) == type([]))
        i = 0
        for name in listname:
            tmp = self.storeparam[name]
            if i == 0:
                mat = tmp
            else:
                mat = np.hstack([mat, tmp])
            i = i + 1
        #    mat.append(self.storeparam[name])
        return np.mean(mat, axis = 0), np.cov(mat.T)

    def get_mean_var(self, name):
        """Returns the estimate from the MCMC estimation for the posterior mean and
        variance"""
        ngroups = self.storeblock[self.name_group[name]].get_number_groups()
        if ngroups == 1:
             return  self.storeblock[self.name_group[name]].\
                    get_stats(self.nit, self.burn)

        else:
            meanvar = self.storeblock[self.name_group[name]].\
                    get_stats(self.nit, self.burn)
            names = self.storeblock[self.name_group[name]].get_name()
            index = names.index(name)
            meanp = meanvar[0][index]
            varp = meanvar[1][index]
            return meanp, varp
    
            

        
    def set_number_decimals(self, num):
        """Sets the number of decimal places for the output"""
        self.numdec = num

    # def AddBlock(self, block):
    #    self.storeblock[block[0].get_name()] = block

    def calc_BIC(self):
        loglike = self.loglike(self.currentparam)
        numparam = self.nparamlike
        nobs = self.currentparam[self.dataname].size 
        bic =-2.*loglike + float(numparam) * np.log(nobs) 
        return bic, loglike 

    def get_plot_suffix(self):
        '''
        get a suitable string for the
        plot type. This depends on the backend
        '''
        backend = matplotlib.get_backend()
        ## later 
        return backend

    def get_default_filename(self, basename ="pymcmc"):
        '''
        get a suitable default filename that suits
        the plot type.
        '''
        output_backends = ['svg', 'pdf', 'ps', 'gdk',
                           'agg', 'emf', 'svgz', "jpg", 'Qt4Agg']
        output_suffixes = ['.svg', '.pdf', '.ps', '.gdk',
                           '.png', '.emf', '.svgz', ".jpg", '.png']
        thisbackend = matplotlib.get_backend().lower()
        for i in range(len(output_backends)):
            if thisbackend == output_backends[i]:
                filename = "%s%s" % (basename, output_suffixes[i])
                return filename
        return None

    def get_plot_dimensions(self, kwargs):
        nelements = len(kwargs['elements'])
        ## now work out the dimension
        totalplots = nelements * len(kwargs['plottypes'])
        if kwargs.has_key('individual') and kwargs['individual']:
            cols = 1
            rows = 1
        elif kwargs.has_key('rows') and not kwargs.has_key('cols'):
            ## work out the cols from the rows
            cols = np.ceil(totalplots/float(kwargs['rows']))
            rows = kwargs['rows']
        elif kwargs.has_key('cols') and not kwargs.has_key('rows'):
            rows = np.ceil(totalplots/float(kwargs['cols']))
            cols = kwargs['cols']
        elif not kwargs.has_key('cols') and not kwargs.has_key('rows'):
            cols = len(kwargs['plottypes']) * np.floor(np.sqrt(totalplots)/len(
                kwargs['plottypes']))
            if cols == 0:
                cols = len(kwargs['plottypes'])
            rows = int(np.ceil(totalplots/cols))
        else:
            rows = kwargs['rows']
            cols = kwargs['cols']

        totalpages = np.ceil(totalplots/(cols*rows))
        plotdims = {'totalplots':totalplots,
                    'cols':int(cols),
                    'rows':int(rows),
                    'figsperplot':int(rows * cols),
                    'totalpages':int(totalpages)}
        return plotdims

        
    def plot(self, blockname, **kwargs):
        '''
        The basic plotting approach for the MCMC class.

        Create summary plots of the MCMC sampler. By default, a plot
        of the marginal posterior density, an ACF plot and a trace
        plot are produced for each parameter in the block. The
        plotting page is divided into a number of subfigures. By
        default, the number of number of columns are approximately
        equal to the square root of the total number of subfigures
        divided by the number of different plot types.
        
        Arguments:

          blockname: The name of the parameter for which summary plots
          are to be generated.

        Keyword arguments:

          elements: a list of integers specifying which elements are
          to be plotted. For example, if the blockname is beta and
          beta has n elements in it, you may specify elements as
          elements = [0, 2, 5], where any of the list containing
          integers less than n.

          plottypes: a list giving the type of plot for each
          parameter. By default the plots are density, acf and
          trace. A single string is also acceptable.

          filename: A string providing the name of an output file for
          the plot. Since a plot of a block may be made up of a number
          of sub figures, the output name will be modified to give a
          separate filename for each subfigure. For example, if the
          filename is passed as plot.png, this will be interpreted
          as plot%03d.png, and will produce the files plot001.png,
          plot002.png, etc. The type of file is determined by the
          extension of the filename, but the output format will also
          depend on the plotting backend being used. If the filename
          does not have a suffix, a default format will be chosen
          based on the graphics backend. Most backends support png,
          pdf, ps, eps and svg, but see the documentation for
          matplotlib for more details.

          individual: A boolean option. If true, then each sub plot
          will be done on an individual page.

          rows: Integer specifying the number of rows of subfigures on
          a plotting page.

          cols: Integer specifying the number of columns of subfigures
          on a plotting page.

        '''
        ## plt.figure()
        paramstore = self.get_parameter_exburn(blockname)
        if not kwargs.has_key('elements'):
            ## we assume you want all the parameters
            kwargs['elements'] = range(paramstore.shape[1])
            
        ## which plots do you want
        if kwargs.has_key('plottypes'):
            ## if you pass a single string, it should
            ## still work:
            if isinstance(kwargs['plottypes'], basestring):
                kwargs['plottypes'] = [kwargs['plottypes']]
        else:
            ## then we assume you want
            ## the following
            kwargs['plottypes'] = ['density', 'acf', 'trace']
            
        nelements = len(kwargs['elements'])
        
        ## now work out the dimension
        plotdims = self.get_plot_dimensions(kwargs)

        ## see if we need a filename 
        defaultfilename = self.get_default_filename()
        if not kwargs.has_key('filename') and defaultfilename:
            ## then we need a default filename
            kwargs['filename'] = defaultfilename

        ## if you need a filename, then not in interactive
        if defaultfilename:
            interactive = False
        else:
            interactive = True
        
        
        ## set up the subfigure
        ## I think you can set up a function in here
        ## until then I'll just use a loop
        plotcounter = 0
        pagecounter = 0
        
        ## check to see if blockname is a latex word
        try:
            aa = latexysmbols.index(blockname)
            ptitle = r'$\%s$' % blockname
        except:
            ptitle = blockname
        for i in kwargs['elements']:
            for plottype in kwargs['plottypes']:
                ## do we need a new figure?
                if plotcounter % plotdims['figsperplot'] == 0:
                    if plotcounter > 0:
                        pagecounter = pagecounter + 1
                        ## then already plotted something,
                        ## we might want to save it
                        if kwargs.has_key('filename'):
                            (base, suffix) = os.path.splitext(kwargs['filename'])
                            fname = "%s%03d%s" % (base, pagecounter, suffix)
                            plt.savefig(fname) 
                    plotcounter = 0
                    plt.figure()
                plotcounter = plotcounter + 1
                plt.subplot(plotdims['rows'], plotdims['cols'], plotcounter)
                try:
                    aa = latexsymbols.index(blockname)
                    title = r'$\%s_%d$' % (blockname, i)
                except:
                    title = "%s_%d" % (blockname, i)
                if plottype == 'acf':
                    bwcalc = InefficiencyFactor()
                    bw = bwcalc.calc_b(paramstore[:, i])
                    maxlag = max(round(1.5 * bw/10.) * 10, 10)
                    PlotACF(paramstore[:, i], maxlag, "ACF Plot %s" % title)
                elif plottype == "density":
                    ##avoid overlapping labels
                    if plotdims['figsperplot'] > 4:
                        ntick = 5
                    else:
                        ntick = 10
                    PlotMarginalPost(paramstore[:, i],
                                     "MPD %s" % title, plottype="both",
                                     maxntick=ntick)
                elif plottype == "trace":
                    ##avoid overlapping labels
                    if plotdims['figsperplot'] > 4:
                        ntick = 5
                    else:
                        ntick = 10
                    PlotIterates(paramstore[:, i], "Trace %s" % title,ntick)
                else:
                    pass
        pagecounter = pagecounter + 1
        ## then already plotted something,
        ## we might want to save it
        if kwargs.has_key('filename'):
            if plotdims['totalpages'] > 1:
                (base, suffix) = os.path.splitext(kwargs['filename'])
                fname = "%s%03d%s" % (base, pagecounter, suffix)
            else:
                fname = kwargs['filename']
            plt.savefig(fname) 
        if interactive:
            plt.show()


    def showplot(self):
        '''
        show any plots you have created.
        '''
        plt.show()
        

    def CODAwrite(self, param, paramname, fobj, findixobj,
                  prange, start, thin, offset):
        '''
        Writes the stored results to file in CODA format.
        Each simulation is written on a single line, preceded
        by the simulation number.

        Arguments:
          param: the data from store (numpy array)
          paramname: the name of the parameter
          fobj: a file handle to write the iterates to.
          findixobj: a file handle to write the index to
          prange: a list of indices. If false, we assume
                  you want all components.
          start: the start the iterates. If burnin is
                 included, this will be 1, otherwise
                 it will be the burnin
          thin: In case you want to thin the output. Every
                jth line will be written, with j=thin.
          offset: Required to write the index file. The
                index file requires line numbers, so if
                you have already written some values to
                file, you need to know the offset.

        Output:
          offset: this will be the value of offset +
                the total number of lines written.

        '''
        nrow = param.shape[0]
        dim = param.shape[1:]
        if not prange:
            prange = [np.unravel_index(i,dim) for i in range(np.prod(dim))]
        itnumbers = np.arange(start, start + nrow, thin)
        nitems = len(itnumbers)
        for pos in prange:
            if len(pos) == 1:
                myslice = np.index_exp[::thin,pos[0]]
                if np.prod(dim)==1:
                    pname = paramname
                else:
                    pname = "%s[%d]" % (paramname,pos[0])
            elif len(pos) == 2:
                myslice = np.index_exp[::thin,pos[0],pos[1]]
                pname = "%s[%d,%d]" % (paramname,pos[0],pos[1])
            elif len(pos) == 3:
                myslice = np.index_exp[::thin,pos[0],pos[1],pos[2]]
                pname = "%s[%d,%d,%d]" % (paramname,pos[0],pos[1],pos[2])
            elif len(pos) == 4:
                myslice = np.index_exp[::thin,pos[0],pos[1],pos[2],pos[3]]
                pname = "%s[%d,%d,%d,%d]" % (
                    paramname,pos[0],pos[1],pos[2],pos[3])
            else:
                print "Can't write coda output for arrays with dim > 4"
            tmp = np.transpose(np.array([itnumbers, param[myslice]]))
            np.savetxt(fobj, tmp, ["%d", "%.06f"])
            findixobj.write("%s %d %d\n" % (pname, start, start+nitems -1))
            start = start + nitems
            offset = offset + nitems
        return offset
        
            
    def CODAoutput(self, **kwargs):
        '''
        
        Output the results in a format suitable for reading in using CODA.

        Write the output to file  in a format that can be read in by CODA.
        By default, there will be two files created, coda.txt and coda.ind.

        Keyword arguments:

         filename: A string to provide an alternative filename for the
          output. If the file has an extension, this will form the
          basis for the data file, and the index file will be named by
          replacing the extension with ind. If no extension is in the
          filename, then two files will be created and named by adding
          the extensions .txt and .ind to the given filename.

         parameters: a string, a list or a dictionary.
          As in output, kwargs can contain a parameters arguemnt. 
          This tells us what we want to save to file.
          It can be is something like 'alpha'
           or it can be a list (eg ['alpha', 'beta'])
           or it can be a dictionary (eg {'alpha':{'range':[0, 1, 5]}},
          If you supply a dictionary, the key is the parameter name
          then you can have a range key with a range of elements.
          If the range isnt supplied, we assume that we want all the elements.
          You can use, for example,
          parameters = {'beta':{'range':[0, 2, 4]}}

         thin: integer specifying how to thin the output. 
          
        '''
        if kwargs.has_key('thin'):
            thin = kwargs['thin']
        else:
            thin = 1
        ## delete any previous output
        if kwargs.has_key('filename'):
            ## work out the two file names
            fname = kwargs['filename']
            ## see if it has an extension:
            basename, extension = os.path.splitext(fname)
            if len(extension) == 0:
                ## you didn't give an extension
                ## so we make one up
                fname = "%s.txt" % basename
            findexname = "%s.ind" % basename
        else:
            fname = "coda.txt"
            findexname = "coda.ind"
        if os.path.exists(fname):
            os.unlink(fname)
        if os.path.exists(findexname):
            os.unlink(findexname)
        fobj = open(fname, "a")
        findobj = open(findexname, "a")
        start = 1
        #indrow refers to the line number of the
        #txt file for the parameter. It is used in
        #the ind file
        indrow = 1
        if 'parameters' in kwargs:
            ## since parameters could be a dictionary
            ## but not necessarily,
            ## we will force it to be one.
            ## If it is passed as a dictionary, it might have
            ## a range argument, which we check later.
            parameters = kwargs['parameters']            
            ## first see if it is a single string
            if type(kwargs['parameters']) == types.StringType:
                parameters = {kwargs['parameters']:{}}
            elif type(kwargs['parameters']) == types.ListType:
                parameters = {}
                for pname in kwargs['parameters']:
                    parameters[pname] = {}
            else:
                parameters = kwargs['parameters']
        else:
            ## then we make up a dictionary ourselves:
            parameters = {}
            for i in range(self.nblocks):
                groupname = self.group_names[i]
                thisblock = self.storeblock[groupname]
                thisname = thisblock.get_name()
                if thisblock.get_store() == 'all':
                    parameters[thisname] = {}
        offset = 0
        for blockname in parameters.keys():
            if kwargs.has_key('exclude_burnin') and kwargs['exclude_burnin']:
                paramstore = self.get_parameter_exburn(blockname)
                nitems = (self.nit - self.burn)/thin
                start = self.burn
            else:
                paramstore = self.get_parameter(blockname)
                nitems = self.nit/thin
                start = 1
            if 'range' in parameters[blockname]:
                #then we want a subset
                tmprange = parameters[blockname]['range']
                prange = []
                try:
                    for i in tmprange:
                        if type(i) == types.IntType:
                            prange.append( (i, ) )
                        else:
                            ## assume already a tuple
                            prange.append(i)
                except:
                    print "Couldn't make sense of your selected range"
                    print "Using all elements"
                    prange = False
            else:
                prange = False
            offset = self.CODAwrite(paramstore, blockname, fobj, findobj,
                           prange, start, thin, offset)
        fobj.close()
            


    def print_header(self, destination, totaltime, nblocks, colwidth):
        '''
        Print a generic header for the output
        '''
        print >>destination, ""
        print >>destination, \
              "--------------------------------------------------------"
        print >>destination, ""
        print >>destination, "The time (seconds) for the MCMC sampler = ", \
        totaltime
        print >>destination, "Number of blocks in MCMC sampler = ", nblocks
        print >>destination, ""
        print >>destination, "{0: >{2}}{1: >{3}}".format("mean", "sd",
                                                         colwidth * 2, colwidth),
        print >>destination, "{0: >{2}}{1: >{2}}".   \
              format("2.5%", "97.5%", colwidth),
        print >>destination, "{0: >{1}}".format("IFactor", colwidth)

        
    def formatPosition(self, position):
        '''a position is the index of an element.
        eg (2, 1). I want to format it in a particular way
        eg (2, 1) -> [2, 1]
        but (2, ) -> [2]
        and () ->
        '''
        aa = format(position)
        if len(position) == 0:
            aa = ""
        elif len(position) == 1:
            aa = '[%d]' % position[0]
        else:
            aa = '[%s]' % ', '.join([str(i) for i in position])
        return aa
    
        


    def print_summary(self, destination, paramname, meanval, sdval,
                     ifactor, hpdintervals, hpd05,
                     prange, colwidth, sigfigs):
        '''
        format the output for a single line.
        Arguments are the name of the parameter, its
        mean value, the standard deviation and the ifactor.
        '''
        ## now, the elements might be a single no.
        ## or a 1 d array,
        ## or a 2 d array
        b = np.ndenumerate(meanval)
        name = paramname
        all_summary_vals = np.array([])
        summary_names = []
        for position, value in b:
            summary_vals = np.zeros( (5,), float)
            #if the position is found in prange,
            #or if you didn't specify a prange,
            if (prange and position in prange) or not prange:
                if meanval.flatten().shape[0] > 1:
                    name = "{0}{1}".format(
                        paramname, self.formatPosition(position)
                        )
                print >>destination, "{name: >{colwidth}}\
{val1: >0{colwidth}.{sigfigs}g}{val2: >0{colwidth}.{sigfigs}g}".format(
                    name = name,
                    val1 = value,
                    val2 = sdval[position],
                    colwidth = colwidth, sigfigs = sigfigs),
                summary_names.append(name)
                summary_vals[0] = value
                summary_vals[1] = sdval[position]
                if hpdintervals:
                ## now for the hpd's
                    print  >>destination, "{val1: >0{colwidth}.{sigfigs}g}\
{val5: >0{colwidth}.{sigfigs}g}".format(
                        val1 = hpd05[0][position],
                        val5 = hpd05[1][position],
                        colwidth = colwidth, sigfigs = sigfigs),
                    summary_vals[2] = hpd05[0][position]
                    summary_vals[3] = hpd05[1][position]
                else:
                    for i in range(2):
                       print  >>destination, \
                        "{0: >0{colwidth}}".format("NA", colwidth = colwidth - 1),
                    summary_vals[2] = np.nan
                    summary_vals[3] = np.nan

                thisifactor = ifactor[position[0]]
                ##check for Chris's nan value
                if thisifactor == -9999:
                    thisifactor = 'NA'
		if type(thisifactor) == type('a string'):
                    ## then not numeric.
                    ## Is there a better way of testing?
		    print  >>destination,     \
                       "{val1: >0{colwidth}}".format(val1 = thisifactor,
                                                     colwidth = colwidth)
                    summary_vals[4] = np.nan
                else:
                    print >>destination,      \
                          "{val1: >0{colwidth}.{sigfigs}g}".format(
                        val1 = thisifactor,
                        colwidth = colwidth, sigfigs = sigfigs)
                    summary_vals[4] = ifactor[position]
                all_summary_vals = np.r_[all_summary_vals,summary_vals]
        return summary_names,all_summary_vals


    def output(self, **kwargs):
        """
        Produce output for MCMC sampler.

        By default output is produced for all parameters. Function
        takes the following options:
           * parameters: A dictionary, list or string
             specifying which parameters are going to be presented.

             If a string (eg 'beta'), all elements of that parameter
             are given.

             If a list, (eg ['alpha', 'beta']), all elements of each
             parameter in the list are given.

             If a dictionary (eg {'alpha':{'range':[range(5)]}}), then
             there is the possibility to add an additional argument
             'range', which tells the output to only print a subset
             of the parameters. The above example will print
             information for alpha[0],alpha[1],...alpha[4] only.
             For 2d and higher arrays, the range should be specified
             so for a 3d array, it would look like:
                'range':( (i,j,k),(l,m,n) )

           * custom - A user define function that produces custom output.
           * filename - A filename to which the output is printed. By
             default output will be printed to stdout.
        """
        summary_vals = np.array([])
        summary_names = []
        acceptance_rates = {}
        if kwargs.has_key("filename"):
            destination = open(kwargs['filename'], 'w')
        else:
            destination = sys.stdout
        if kwargs.has_key("custom"):
            kwargs['custom'](destination)
        else:
            if 'parameters' in kwargs:
                ## since parameters could be a dictionary
                ## but not necessarily,
                ## we will force it to be one.
                ## If it is passed as a dictionary, it might have
                ## a range argument, which we check later.
                parameters = kwargs['parameters']            
                ## first see if it is a single string
                if type(kwargs['parameters']) == types.StringType:
                    parameters = {kwargs['parameters']:{}}
                elif type(kwargs['parameters']) == types.ListType:
                    parameters = {}
                    for pname in kwargs['parameters']:
                        parameters[pname] = {}
                else:
                    parameters = kwargs['parameters']
            else:
                ## then we make up a dictionary ourselves:
                parameters = {}
                for i in range(len(self.all_keys)):
                    parameters[self.all_keys[i]] = {}


            IF = InefficiencyFactor()
            ## these should be set somewhere within 
            colwidth = 12
            sigfigs = self.numdec

            self.print_header(destination, self.totaltime, self.nblocks, colwidth)

            for paramname in parameters.keys():
                meanp, varp = self.get_mean_var(paramname)
                #if self.transformfunc_ind == True:
                #    if paramname in self.transform_list:
                #        meanp = np.mean(self.get_parameter_exburn(paramname), axis = 0)
                #        varp = np.var(self.get_parameter_exburn(paramname), axis = 0)
                
                self.meanstore[paramname] = meanp
                self.varstore[paramname] = varp
                if self.storeblock[self.name_group[paramname]].get_store() =='all':
                    ifactor = IF.calculate(self.get_parameter_exburn(paramname)).\
                      round(self.numdec)
                else:
                    if type(meanp) == types.FloatType or type(meanp) == np.float64:
                        ifactor = ['NA']
                    elif np.array(meanp).ndim == 1:
                        ifactor = ['NA'] * len(meanp)
                    elif np.array(meanp).ndim == 2:
                        ifactor =  np.resize(np.array(['NA']),meanp.shape)
                    else:
                        print "ERROR: I don't know how to deal with arrays of shape",meanp.shape,"yet"
                        return None
                ## ifactor.shape = meanp.shape
                ## and calc hpds'
                if self.storeblock[self.name_group[paramname]].get_store() =='all':
                    paramstore = self.get_parameter_exburn(paramname)
                    ## and we calculate the hpd's for this
                    out05 = np.apply_along_axis(hpd, 0, paramstore, 0.05)
                    hpdintervals = True
                else:
                    hpdintervals = False
                    out05 = None
                if 'range' in parameters[paramname]:
                    ## then we want a subset
                    tmprange = parameters[paramname]['range']
                    ## we do some minimal massaging of this list
                    ## each element should be a tuple. If its not
                    ## we assume it should be
                    prange = []
                    try:
                        for i in tmprange:
                            if type(i) == types.IntType:
                                prange.append( (i, ) )
                            else:
                                ## assume already a tuple
                                prange.append(i)
                    except:
                        print "Couldn't make sense of your selected range"
                        prange = False
                    
                else:
                    ## not sure how we should deal with
                    ## 2/3d arrays...
                    prange = False
                these_names,thisval = (self.print_summary(destination, paramname,
                                  np.atleast_1d(meanp),
                                  np.atleast_1d(np.sqrt(varp)),
                                  np.atleast_1d(ifactor),
                                  hpdintervals,
                                  np.atleast_1d(out05),
                                  prange,
                                  colwidth, sigfigs))
                summary_vals = np.r_[summary_vals,thisval]
                summary_names.extend(these_names)
                ## this is where you would put the acceptance rate info in.
            if self.calcbic == True:
                for paramname in parameters.keys():
                    meanp, varp = self.get_mean_var(paramname)
                    self.meanstore[paramname] = meanp
                    self.varstore[paramname] = varp
                    self.currentparam[paramname] = self.meanstore[paramname]
                BIC, LOGLIKE = self.calc_BIC();

            print ""
            for name in parameters.keys():
                print >>destination, 'Acceptance rate ', name, ' = ', \
                self.storeblock[self.name_group[name]].acceptance_rate()
                acceptance_rates[name] = \
                      self.storeblock[self.name_group[name]].acceptance_rate()
            summary_vals = summary_vals.reshape( (-1,5))
            self.output_dictionary = {'parameter names':summary_names,
                                      'summary values':summary_vals,
                                      'acceptance rates':acceptance_rates
                                      }
            if self.calcbic == True:
                #convert to float if needed
                try:
                    BIC = BIC[0]
                    LOGLIKE = LOGLIKE[0]
                except:
                    pass
                print >>destination, "BIC = {bic: .{sigfigs}f}".format(
                    bic = BIC,sigfigs=sigfigs)
                print >>destination, \
                      "Log likelihood = {loglik: .{sigfigs}f}".format(
                    loglik = LOGLIKE,sigfigs=sigfigs)
                self.output_dictionary['BIC'] = BIC
                self.output_dictionary['LOGLIKE'] = LOGLIKE

                
                

# A Bayesian regression module for PyMCMC. PyMCMC is a Python package for
# Bayesian analysis.
# Copyright (C) 2010  Chris Strickland

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# python file for conjugate priors

import types
import os
from os import sys
import numpy as np
from scipy import special
from stochsearch import*
import matplotlib.pyplot as plt
import scipy.stats.distributions as dstn
import wishart
import pdb


class StochasticSearch:
    """
    StochasticSearch is a class that is called from RegSampler and used
    when the user wishes to use the stochastic search to select the
    regressors in a regression
    """
    def __init__(self, yvec, xmat, prior):
        self.nobs = yvec.shape[0]
        self.kreg = xmat.shape[1]
        self.yvec = yvec
        self.xmat = xmat
        self.ypy = np.dot(yvec.T, yvec)
        self.xpy = np.dot(xmat.T, yvec)
        self.xpx = np.asfortranarray(np.dot(xmat.T, xmat))
        self.xgxg = np.zeros((self.kreg, self.kreg), order = 'F' )
        self.work2 = np.zeros((self.kreg, self.kreg), order = 'F' )
        self.xgy = np.zeros(self.kreg)
        self.gam = np.zeros(self.kreg, dtype = 'i')
        self.gam[0] = 1
        self.ru = np.zeros(self.kreg)
        self.rn = np.zeros(self.kreg)
        self.work = np.zeros((self.kreg, 6), order = 'F')
        self.ifo = np.array(0, dtype = 'i')
        self.ifo2 = np.array(0, dtype = 'i')
        if prior[0] == 'g_prior':
            self.work[:, 0] = prior[1]
            self.g = prior[2]
            self.__samplegam = self.__sim_gamma_gprior
        elif prior[0] == 'normal_inverted_gamma':
            self.nu = prior[1]
            self.nuo = self.nu + self.nobs
            self.nus = prior[2]
            self.R = np.asfortranarray(prior[3])
            self.D = np.asfortranarray(prior[4])
            self.logdetR = 2.0 * np.sum(np.diag(np.linalg.cholesky(self.R)))
            self.vxy = self.xgy
            self.vobar = self.xgxg
            self.vubar = self.work2
            self.nuobar = self.nu + self.nobs
            self.__samplegam = self.__sim_gamma_nig
            self.__samplegam_cond_beta = self.__sim_gamma_nig_cond_beta
        else:
            raise NameError("prior incorrectly specified")

        # internal storage for stochastic search
        self.store = [[], []]

    def __sim_gamma_gprior(self):
        ssreg(self.ypy, self.g, self.xpx, self.xpy,
              self.xgxg, self.xgy, self.gam, \
              self.ru, self.work, self.work2,
              self.ifo, self.ifo2, self.nobs)

    def __sim_gamma_nig(self):
        self.initialise_vubar()
        ssreg_nig(self.ypy, self.logdetR, self.nus, self.vxy, self.vobar,
                 self.vubar, self.gam, self.xpx, self.xpy, self.D,
                 self.R, self.nuobar, self.ru)


    def __sim_gamma_nig_cond_beta(self,sig, beta):
        """samples gamma conditional on beta"""
        self.initialise_vubar()
        ssregcbetas_nig(beta, sig, self.vxy, self.logdetR, self.vubar,
                       self.gam, self.D, self.R, self.ru)

    def initialise_vubar(self):
        initialise_vubar(self.vubar, self.gam, self.D, self.R)

    def sample_gamma(self, store):
        it = store['iteration']
        burn = store['length_of_burnin']
        # self.gam = gamvec.astype('i')
        self.ru = np.random.rand(self.kreg)
        self.__samplegam()
        if it >= burn:
            self.update_store()
        return self.gam

    def sample_gamma_cond_beta(self, store,sig, beta):
        it = store['iteration']
        burn = store['length_of_burnin']
        self.ru = np.random.rand(self.kreg)
        self.__samplegam_cond_beta(sig, beta)
        if it >= burn:
            self.update_store()
        return self.gam
        

    def update_store(self):
        """function updates internal storage for gamma"""
        gammai = int("".join([str(i) for i in self.gam]))
        if gammai in self.store[0]:
            index = self.store[0].index(gammai)
            self.store[1][index] = self.store[1][index] + 1
        else:
            self.store[0].append(gammai)
            self.store[1].append(1)

    def __extract_position(self,i,ind):
        modstr = str(self.store[0][ind[i]])
        modnum=len(modstr)*[False]
        for j in xrange(len(modstr)):
            if modstr[j] =='1':
                modnum[j] = True
        return modnum
    
    def extract_regressors(self, model_number):
        '''returnrs a design matrix containing just the regressors
        correponding to the specified model_number
        '''
        arrind = np.array(self.store[1])
        ind = np.argsort(arrind)[::-1]
        modnum = self.__extract_position(model_number, ind)
        tmpxmat = np.compress(modnum, self.xmat, axis = 1)
        return tmpxmat
        
    def output(self, destination):
        """
        produce additional output for StochasticSearch
        This is an example of a custom output. The requirement
        is it needs to have a destination which is handled by
        the generic output function.

        """
        
        colwidth = 12
        sigfigs = 7
        arrind = np.array(self.store[1])
        ind = np.argsort(arrind)[::-1]
        total = sum(arrind)
        hline = "-".ljust(5 * colwidth, '-')
        print >>destination
        print >>destination,\
        "Most likely models ordered by decreasing posterior probability"
        print >>destination
        print >>destination, hline        
        print >>destination, """\
{0: <{colwidth}s}| \
{1: <{colwidth}s}""".format("probability", "model", colwidth = colwidth)
        print >>destination, hline
        for i in xrange(min(10, ind.shape[0])):
            modnum = self.__extract_position(i, ind)
            modstr = [] 
            for j in range(len(modnum)):
                if modnum[j]:
                    modstr.append(str(j))
            print >>destination, """\
{1: <{colwidth}.{sigfigs}g}| \
{0: <{colwidth}s}""".format(
                ', '.join(modstr),
                float(self.store[1][ind[i]])/total,
                colwidth = colwidth,
                sigfigs = sigfigs)
        print >>destination, hline


class BayesRegression:
    """
    BayesRegression is a class for Bayesian regression. By default this class uses
    Jeffrey's prior. Arguments:

        yvec - Is a one dimensional numpy array containing the dependent
               variable.
        xmat - Is a two dimensional numpy array conting the regressors.
        kwargs - Optional arguments:
            prior - a list containing the name of the prior and the
               corresponding hyperparameters. Examples: 
               prior = ['normal_gamma', nuubar, Subar, betaubar, Vubar]
               prior = ['normal_inverted_gamma', nuubar, Subar, betaubar, Vubar]

               prior = ['g_prior', betaubar, g].
               If none of these options are chosen or they are
               miss - specified then BayesRegression will default to
               Jeffreys prior.
        
    """
    def __init__(self, yvec, xmat, **kwargs):
        self.nobs = yvec.shape[0]
        self.yvec = yvec
        if xmat.ndim == 1:
            self.xmat = xmat.reshape(self.nobs, 1)
        else:
            self.xmat = xmat
        self.xpx = np.dot(self.xmat.T, self.xmat)
        self.xpy = np.dot(self.xmat.T, yvec)
        self.kreg = self.xmat.shape[1]
        self.vobar = np.zeros((self.kreg, self.kreg))
        self.betaobar = np.zeros(self.kreg)
        self.updateind_xmat = 0
        self.updateind_yvec = 0
        self.calculated = False
        self.nuobar = 0.0
        self.sobar = 0.0
        self.vbobar = np.zeros(self.kreg)
        self.cholvobar = np.zeros((self.kreg, self.kreg))
        if 'prior' not in kwargs:
            # default: Jeffreys prior
            self.res = np.zeros(self.nobs)
            self.__calculate = self.__calculate_jeffreys
            self.__sample_scale = self.__sample_standard_deviation
            self.__log_cand_prob = self.__log_cand_pr_sig_jeff
            self.prior = ['Jeffreys']
            self.__posterior_variance_scale = self.__posterior_sigma_var
            self.__posterior_mean_scale = self.__posterior_sigma_mean
            self.__log_marginal_likelihood = self.__log_marginal_likelihood_jeff
        else:                   # Normal - gamma prior
            self.prior = kwargs['prior']
            if type(self.prior[0])!= types.StringType:
                print "Warning: Jefferys prior used as prior was \
incorectly specified"
                self.res = np.zeros(self.nobs)
                self.__calculate = self.__calculate_jeffreys
                self.__sample_scale = self.__sample_standard_deviation
                self.__log_cand_prob = self.__log_cand_pr_sig
                self.__posterior_variance_scale = self.__posterior_sigma_var
                self.__posterior_mean_scale = self.__posterior_sigma_mean
                self.__log_marginal_likelihood = self.__log_marginal_likelihood_jeff
            else:
                ptype = self.prior[0]
                if ptype not in ['normal_gamma', 'normal_inverted_gamma',
                                 'g_prior']:
                    print "Warning: Jeffery's prior used as prior was \
incorectly specified"
                    self.res = np.zeros(self.nobs)
                    self.__sample_scale = self.__sample_standard_deviation
                    self.__calculate = self.__calculate_jeffreys
                    self.__log_cand_prob = self.__log_cand_pr_sig
                    self.__posterior_variance_scale = \
                                               self.__posterior_sigma_var
                    self.__posterior_mean_scale = \
                                              self.__posterior_sigma_mean
                    self.__log_marginal_likelihood = self.__log_marginal_likelihood_jeff
                else:
                    self.vbubar = np.zeros(self.kreg)
                    if ptype =='normal_gamma':
                        self.__calculate = self.__calculate_normal_gamma
                        self.__sample_scale = self.__sample_precision
                        self.__log_cand_prob = self.__log_cand_pr_kappa
                        self.__posterior_variance_scale = \
                                               self.__posterior_kappa_var
                        self.__posterior_mean_scale = \
                                              self.__posterior_kappa_mean
                        self.__log_marginal_likelihood = \
                                self.__log_marginal_likelihood_nig
                        self.nuubar = self.prior[1]
                        self.subar = self.prior[2]
                        self.betaubar = self.prior[3]
                        self.vubar = np.atleast_2d(self.prior[4])
                        self.lndetvubar = 2.0 * \
                        np.sum(np.log(np.diag(np.linalg.cholesky(self.vubar))))

                    elif ptype =='normal_inverted_gamma':
                        self.__calculate = self.__calculate_normal_gamma
                        self.__sample_scale = \
                                         self.__sample_standard_deviation
                        self.__log_cand_prob = self.__log_cand_pr_sig
                        self.__posterior_variance_scale = \
                                               self.__posterior_sigma_var
                        self.__posterior_mean_scale = \
                                               self.__posterior_sigma_mean
                        self.nuubar = self.prior[1]
                        self.subar = self.prior[2]
                        self.betaubar = self.prior[3]
                        self.vubar = np.atleast_2d(self.prior[4])
                        self.lndetvubar = 2.0 * \
                        np.sum(np.log(np.diag(np.linalg.cholesky(self.vubar))))
                        self.__log_marginal_likelihood = \
                                 self.__log_marginal_likelihood_nig
                        
                    else:
                        # g - prior
                        self.betaubar = self.prior[1]
                        self.g = self.prior[2]
                        self.betahat = np.zeros(self.kreg)
                        self.betadiff = np.zeros(self.kreg)
                        self.res = np.zeros(self.nobs)
                        assert(type(self.g) == types.FloatType)
                        self.gratio = self.g/(self.g + 1.)
                        self.__sample_scale = \
                                        self.__sample_standard_deviation
                        self.__calculate = self.__calculate_g_prior
                        self.__log_cand_prob = self.__log_canc_pr_sig_gprior
                        self.__posterior_variance_scale = \
                                               self.__posterior_sigma_var
                        self.__posterior_mean_scale = \
                                             self.__posterior_sigma_mean
                        self.__log_marginal_likelihood = self.__log_marginal_likelihood_gprior
                        self.vubar = self.xpx / self.g
                        self.lndetvubar = 2.0 * \
                        np.sum(np.log(np.diag(np.linalg.cholesky(self.vubar))))

    def update_prior(self, prior):
        if prior[0] == 'normal_inverted_gamma' or prior[0] == 'normal_gamma':
            self.nuubar = self.prior[1]
            self.subar = self.prior[2]
            self.betaubar = self.prior[3]
            self.vubar = self.prior[4]
            self.lndetvubar = 2.0 * \
            np.sum(np.log(np.diag(np.linalg.cholesky(self.vubar))))

        elif prior[0] == 'g_prior':
            self.vubar = self.xpx / self.g
            self.lndetvubar = 2.0 * \
            np.sum(np.log(np.diag(np.linalg.cholesky(self.vubar))))


    def log_posterior_probability(self, scale, beta, **kwargs):
        return self.__log_cand_prob(scale, beta, **kwargs)

    def __calculate_jeffreys(self):
        self.calculated = True
        if  self.updateind_xmat == 1 or self.updateind_yvec == 1:
            self.xpy = np.dot(self.xmat.transpose(), self.yvec)
            if self.updateind_xmat == 1: 
                self.xpx = np.dot(self.xmat.transpose(), self.xmat)
            self.updateind_xmat = 0
            self.updateind_yvec = 0

        self.betaobar = np.linalg.solve(self.xpx, self.xpy)
        self.vobar = self.xpx

        self.nuobar = self.nobs - self.kreg
        self.res = self.yvec - np.dot(self.xmat, self.betaobar)
        self.sobar = np.dot(self.res, self.res)

    def __calculate_normal_gamma(self):
        self.calculated = True
        self.vbubar = np.dot(self.vubar, self.betaubar)
        if  self.updateind_xmat == 1 or self.updateind_yvec == 1:
            self.xpy = np.dot(self.xmat.transpose(), self.yvec)
            if self.updateind_xmat == 1: 
                self.xpx = np.dot(self.xmat.transpose(), self.xmat)
            self.updateind_xmat = 0
            self.updateind_yvec = 0
        self.vobar = self.vubar + self.xpx
        self.vbobar = self.xpy + self.vbubar
        self.betaobar = np.linalg.solve(self.vobar, self.vbobar)
        
        self.nuobar = self.nuubar + self.nobs
        self.sobar = self.subar + sum(self.yvec**2)+ \
        np.dot(self.betaubar, self.vbubar)- \
        np.dot(self.betaobar, self.vbobar)

    def __calculate_g_prior(self):
        self.calculated = True
        if  self.updateind_xmat == 1 or self.updateind_yvec == 1:
            self.xpy = np.dot(self.xmat.transpose(), self.yvec)
            if self.updateind_xmat == 1: 
                self.xpx = np.dot(self.xmat.transpose(), self.xmat)
            self.updateind_xmat = 0
            self.updateind_yvec = 0
        self.betahat = np.linalg.solve(self.xpx, self.xpy)
        self.betaobar = self.gratio * (self.betahat +
                                       self.betaubar/self.g)
        self.vobar = 1./self.gratio * self.xpx
        self.nuobar = self.nobs
        self.betadiff = self.betahat - self.betaubar
        self.res = self.yvec - np.dot(self.xmat, self.betahat)
        self.sobar = np.dot(self.res.T, self.res)+ \
        np.dot(self.betadiff.T, np.dot(self.xpx, self.betadiff))/(
            self.g + 1.)


    def sample(self):
        self.__calculate()
        sig = self.__sample_scale()
        beta = self.betaobar + np.linalg.solve(self.cholvobar.T,
                                               np.random.randn(self.kreg))
        return sig, beta

    def __log_cand_pr_sig(self, sigma, beta, **kwargs):
        """
        calculates the log of the candiate probability given scale = sigma
        """
        loglike = self.loglike(sigma, beta)

        dbeta = beta - self.betaubar
        kern = -self.kreg * np.log(sigma) -0.5 / sigma ** 2 *\
                np.dot(dbeta, np.dot(self.vubar, dbeta))

        kerns = -(self.nuubar + 1) * np.log(sigma) - self.subar/(2.0 * sigma ** 2)
        
        if 'kernel_only' in kwargs and kwargs['kernel_only'] == True:
              return loglike + kern + kerns

        else:
            const = -0.5 * self.kreg * np.log(2 * np.pi) + 0.5 * self.lndetvubar
            consts = np.log(2) - special.gammaln(self.nuubar / 2.) +\
                    self.nuubar / 2. * np.log(self.subar / 2.)
            return loglike + kern + kerns + const + consts


    def __log_cand_pr_sig_jeff(self, sigma, beta, **kwargs):
        
        loglike = self.loglike(sigma, beta)
        return loglike - np.log(sigma)

    def __log_canc_pr_sig_gprior(self, sigma, beta, **kwargs):
        loglike = self.loglike(sigma, beta)
        
        dbeta = beta - self.betaubar
        kern = -self.kreg * np.log(sigma) -0.5 * self.kreg * np.log(self.g) \
                -0.5 / (self.g * sigma ** 2) \
                * np.dot(dbeta, dot(self.vubar, dbeta))

        
        if 'kernel_only' in kwargs and kwargs['kernel_only'] == True:
                return kern - np.log(sigma)
        else:
            const = -0.5 * self.kreg * np.log(2 * np.pi) + 0.5 * self.lndetvubar
            return loglike + kern + const - np.log(sigma)


    def __log_cand_pr_kappa(self, kappa, beta, **kwargs):
        loglike = self.loglike(sigma, beta)

        dbeta = beta - betaubar
        kern = 0.5 * self.kreg * np.log(kappa) -0.5 * kappa \
                * np.dot(dbeta, np.dot(self.vubar, dbeta))

        kerns = (nu + 1) / np.log(kappa) - ns * kappa /2.0 
        
        if 'kernel_only' in kwargs and kwargs['kernel_only'] == True:
            return loglike + kern + kerns

        else:
            const = -0.5 * self.kreg * log(2 * np.pi) + 0.5 * self.lndetvubar
            consts = np.log(2) - special.gammaln(self.nuubar / 2.) +\
                    self.nuubar / 2. * np.log(self.subar/2.)
            return loglike + kern + kerns + const + consts


    def __sample_standard_deviation(self):
        sig = 1.0/np.sqrt(np.random.gamma(self.nuobar/2., 2./self.sobar, 1))
        # self.cholvobar = 1.0/sig * np.linalg.cholesky(self.vobar)
        self.cholvobar = 1.0/sig * np.linalg.cholesky(self.vobar)
        return sig

    def __sample_precision(self):
        kappa = np.random.gamma(self.nuobar/2., 2./self.sobar, 1)
        self.cholvobar = np.linalg.cholesky(kappa * self.vobar)
        return kappa        

    def loglike(self, scale, beta):
        if self.calculated == False:
            self.__calculate()
        if self.prior[0] == 'normal_gamma':
            sig = 1. / np.sqrt(scale)
        else:
            sig = scale
        diff = self.yvec - np.dot(self.xmat, beta)
        sigsq = sig**2
        nobs = self.yvec.shape[0]
        return -0.5 * nobs * np.log(2.0 * np.pi * sigsq) - \
               0.5/sigsq * np.dot(diff, diff)

    def log_marginal_likelihood(self):
        if self.calculated == False:
            self.__calculate()
        return self.__log_marginal_likelihood()

    def __log_marginal_likelihood_nig(self):
        if self.calculated == False:
            self.__calculate()

        logdet_vubar = 2.0 * sum(np.log(np.diag(np.linalg.cholesky(self.vubar)))) 
        logdet_vobar = 2.0 * sum(np.log(np.diag(np.linalg.cholesky(self.vobar))))
        c1 = -0.5 * self.nobs * np.log(2. * np.pi)
        c2 = 0.5 * (logdet_vubar - logdet_vobar)
        c3 = special.gammaln(self.nuobar / 2.) - special.gammaln(self.nuubar / 2.)
        c4 = 0.5 * (self.nuubar * np.log(self.subar / 2.) - self.nuobar * np.log(self.sobar / 2.))
        return c1 + c2 + c3 + c4

    def __log_marginal_likelihood_jeff(self):
        return np.nan
    
    def __log_marginal_likelihood_gprior(self):
        return np.nan
    
    def posterior_mean(self):
        if self.calculated == False:
            self.__calculate()
        betamean = self.betaobar
        sigmamean = self.__posterior_mean_scale()

        return sigmamean, betamean

    def __posterior_sigma_mean(self):
        """Zelner (1971), pp 371"""

        S = np.sqrt(self.sobar/self.nuobar)

        return np.exp(special.gammaln((self.nuobar - 1)/2.)-\
                special.gammaln(self.nuobar/2.))*np.sqrt(self.nuobar/2.) * S

    def __posterior_kappa_mean(self):
        # return self.nuubar/self.sobar
        return self.nuobar/self.sobar

    def __posterior_sigma_var(self):
        """function returns the estimate of the posterior variance for
        sigma, Zelner (1971), pp 373"""

        if self.calculated == False:
            self.__calculate()
        sigmamean = self.posterior_mean()[0]
        var = self.sobar/(self.nuobar - 2) -sigmamean**2
        return var

    def __posterior_kappa_var(self):
        if self.calculated == False:
            self.__calculate()
        s2 = self.sobar/self.nuobar
        return 4./(self.nuobar * s2**2)
       

    def get_posterior_covmat(self):
        '''
        return the posterior covariance
        matrix for beta
        '''
        if self.calculated == False:
            self.__calculate()
        s2 = self.sobar/self.nuobar
        Am = np.linalg.inv(self.vobar)
        nuobar = self.nuobar
        covmat = (nuobar/(nuobar - 2)) * s2 * Am
        return covmat

    def bic(self):
        '''
        Return BIC
        '''
        if self.calculated == False:
            self.__calculate()
        sig,beta = self.posterior_mean()
        loglike = self.loglike(sig,beta)
                               
        return -2 * loglike + (self.kreg + 1) * np.log(self.nobs) 
        
    
    def __thpd(self, nu, bbar, sd):
        '''
        Get the hpd interval for the t-dist.
        '''
        ## and plot it
        rv = dstn.t(nu, bbar, sd)
        xl = rv.ppf(0.025)
        xu = rv.ppf(0.975)
        return np.array([xl, xu])

    def __plottdist(self, nu, bbar, sd, title):
        '''
        Plot t distribution
        '''
        ## and plot it
        rv = dstn.t(nu, bbar, sd)
        xmin =  rv.ppf(0.001)
        xmax = rv.ppf(0.999)
        x = np.linspace(xmin, xmax, 100)
        h = plt.plot(x, rv.pdf(x))
        plt.title(title)
        ## add the hpd's
        xl = rv.ppf(0.025)
        xu = rv.ppf(0.975)
        ltx = np.linspace(xmin, xl, 50)
        lty = rv.pdf(ltx)
        plt.fill(np.r_[ltx, ltx[-1]],
                 np.r_[lty, 0], facecolor ="blue", alpha = 0.5)
        utx = np.linspace(xu, xmax, 50)
        uty = rv.pdf(utx)
        plt.fill(np.r_[utx, utx[0]],
                 np.r_[uty, 0], facecolor ="blue", alpha = 0.5)
        ## return rv

        


    def __plotinvertedgamma(self, nu, s2, title):
        '''
        plots inverted gamma,
        Zellner 1971 for details

        '''
        mode = np.sqrt(s2)*np.sqrt( nu/(nu+1.0) )
        minx = 1E-3
        if minx > 0.01*mode:
            minx = 0.0
            # note this will induce a warning
            # due to divide by zero
        ## centre x on the mode
        x = np.linspace(minx, mode * 2, num = 200)
        d1 = 2.0/special.gamma(nu/2.0)
        d2 = ( (nu * s2)/2.0)**(nu/2.0)
        d3 = 1/(x**(nu + 1.0))
        d4 = (nu * s2)/(2 * (x**2))
        y = d1 * d2 * d3 * np.exp(-d4)
        plt.plot(x, y)
        plt.title(title)

    def get_plot_dimensions(self, kwargs):
        totalplots = self.kreg + 1
        if kwargs.has_key('individual'):
            cols = 1
            rows = 1
        elif kwargs.has_key('rows') and not kwargs.has_key('cols'):
            ## work out the cols from the rows
            cols = ceil(totalplots/float(kwargs['rows']))
            rows = kwargs['rows']
        elif kwargs.has_key('cols') and not kwargs.has_key('rows'):
            rows = ceil(totalplots/float(kwargs['cols']))
            cols = kwargs['cols']
        elif not kwargs.has_key('cols') and not kwargs.has_key('rows'):
            cols = np.floor(np.sqrt(totalplots))
            if cols == 0:
                cols = 1
            rows = int(np.ceil(totalplots/cols))
        else:
            rows = kwargs['rows']
            cols = kwargs['cols']

        plotdims = {'totalplots':totalplots,
                    'cols':int(cols),
                    'rows':int(rows),
                    'figsperplot':int(rows * cols)}
        return plotdims


    def plot(self, **kwargs):
        '''
        Basic plotting function for regression objects.
        '''
        if not self.calculated:
            self.__calculate()
        s2 = self.sobar/self.nuobar
        betasd = np.sqrt(np.diag(self.get_posterior_covmat()))
        
        plotdims = self.get_plot_dimensions(kwargs)
        plotcounter = 0
        pagecounter = 0
        for i in range(plotdims['totalplots'] - 1):
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
            title = r'$\beta_{%d}$' % i
            self.__plottdist(self.nuobar,
                             self.betaobar[i],
                             betasd[i], title)
        ## and the final plot..
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
        title = r'$\sigma$'
        self.__plotinvertedgamma(self.nuobar, s2, title)
        pagecounter = pagecounter + 1
        ## then already plotted something,
        ## we might want to save it
        if kwargs.has_key('filename'):
            (base, suffix) = os.path.splitext(kwargs['filename'])
            fname = "%s%03d%s" % (base, pagecounter, suffix)
            plt.savefig(fname)
        else:
            plt.show()
        
    
    def update_yvec(self, yvec):
        self.yvec = yvec
        self.updateind_yvec = 1
        self.calculated = False

    def update_xmat(self, xmat):
        if xmat.ndim == 1:
            self.xmat = xmat.reshape(xmat.shape[0], 1)
        else:
            self.xmat = xmat
        self.calculated = False
        self.updateind_xmat = 1

    def residuals(self):
        if self.calculated == False:
            self.__calculate()
        sigma,beta=self.posterior_mean()
        return self.yvec-np.dot(self.xmat,beta)
            

    def print_header(self, destination, colwidth, sigfigs):
        '''
        print a generic header for the output:
        '''
        print >>destination, ""
        hline =  "{hline: ^{totalwidth}}".format(
            hline ="---------------------------------------------------",
            totalwidth = 6 * colwidth)
        print >>destination, hline
        print >>destination, \
              "{title: ^{totalwidth}}".format(
            title ="Bayesian Linear Regression Summary",
            totalwidth = 6 * colwidth)
        print >>destination, \
              "{priorname: ^{totalwidth}}".format(
            priorname = self.prior[0],
            totalwidth = 6 * colwidth)
        print >>destination, hline
        print >>destination, """\
{0: >{colwidth}.{colwidth}s}\
{1: >{colwidth}.{colwidth}s}\
{2: >{colwidth}.{colwidth}s} \
{3: >{colwidth}.{colwidth}s}\
{4: >{colwidth}.{colwidth}s}""".format(" ", "mean", "sd", "2.5%", "97.5%",
                                           colwidth = colwidth,
                                           sigfigs = sigfigs)

    def print_summary(self, destination, paramname, meanval, sdval,
                      hpdintervals, hpd05, colwidth, sigfigs):
        '''
        format the output for a single line.
        Arguments are the name of the parameter, its
        mean value, the standard deviation and the hpd (if present).
        Presumably only for a vector.
        '''
        name = paramname
        print >>destination, """\
{name: >{colwidth}.{colwidth}}\
{val1: >0{colwidth}.{sigfigs}g}\
{val2: >0{colwidth}.{sigfigs}g}""".format(
            name = name,
            val1 = meanval,
            val2 = sdval,
            colwidth = colwidth, sigfigs = sigfigs),
        if hpdintervals:
            ## now for the hpd's
            print  >>destination, """\
{val1: >0{colwidth}.{sigfigs}g}\
{val5: >0{colwidth}.{sigfigs}g}""".format(
                val1 = hpd05[0],
                val5 = hpd05[1],
                colwidth = colwidth,
                sigfigs = sigfigs)
        else:
            print  >>destination, """\
{0: >0{colwidth}.{colwidth}s}\
{0: >0{colwidth}.{colwidth}s}""".format("NA", colwidth = colwidth - 1)


    def output(self, **kwargs):
        '''
        Output for the regression summary.
        '''
        colwidth = 12
        sigfigs = 4
        if not self.calculated:
            self.__calculate()
        if kwargs.has_key("filename"):
            destination = open(kwargs['filename'], 'w')
        else:
            destination = sys.stdout
        self.print_header(destination, colwidth, sigfigs)
        sigmean, betamean = self.posterior_mean()
        betasd = np.sqrt(np.diag(self.get_posterior_covmat()))
        for i in range(len(betamean)):
            paramname = "beta[%d]" % i
            hpd = self.__thpd(self.nuobar,
                              betamean[i],
                              betasd[i])
            
            self.print_summary(destination, paramname,
                          betamean[i],
                          betasd[i],
                          True, hpd, colwidth, sigfigs)

        
        ## and now for sigma
        if self.prior[0] =="normal_gamma":
            scale_name = "kappa"
        else:
            scale_name = "sigma"
        sigsd = np.sqrt(self.__posterior_variance_scale())
        self.print_summary(destination, scale_name,
                      sigmean,
                      sigsd,
                      False, None, colwidth, sigfigs) 
        ## and print loglikelihood:
        print >>destination
        print >>destination, \
        "loglikelihood = {loglik: <0{colwidth}.{sigfigs}g}".format(
            loglik=self.loglike(sigmean,betamean),
            colwidth=colwidth,
            sigfigs=sigfigs)
        print >>destination,\
        "log marginal likelihood = {marglik: <0{colwidth}.{sigfigs}g}".format(
            marglik = self.log_marginal_likelihood(),
            colwidth = colwidth,
            sigfigs = sigfigs)

        print >>destination, \
        "BIC  = {bic: <0{colwidth}.{sigfigs}g}".format(
            bic = self.bic(),
            colwidth = colwidth,
            sigfigs = sigfigs)


class CondBetaRegSampler:
    """

    This class samples beta assuming it is generated from a linear
    regression model where the scale parameter is known. This class is
    initialised with the following arguments:
      yvec - a one dimensional numpy array containing the data.
      xmat - a two dimensional numpy array containing the regressors.
      kwargs - optional arguments:
      prior - a list containing the name of the prior and the
        corresponding  hyperparameters.
        Examples:
          prior = ['normal', betaubar, Vubar] or
          prior = ['g_prior', betaubar, g].
        If none of these options are chosen or they are miss-specified
        then CondBetaRegSampler will default to Jeffrey's prior.
    """

    def __init__(self, yvec, xmat, **kwargs):
        self.nobs = yvec.shape[0]
        if xmat.ndim == 1:
            xmat = xmat.reshape(xmat.shape[0], 1)
        self.kreg = xmat.shape[1]
        self.yvec = yvec
        self.xmat = xmat
        self.xpx = np.dot(xmat.T, xmat)
        self.xpy = np.dot(xmat.T, yvec)
        self.updateind_xmat = 0
        self.updateind_yvec = 0
        self.betaobar = np.zeros(self.kreg)
        self.vobar = np.zeros((self.kreg, self.kreg))
        self.vbobar = np.zeros(self.kreg)
        self.cholvobar = np.zeros((self.kreg, self.kreg))

        if 'prior' not in kwargs:      # default: Jeffrey's prior
            self.__calculate = self.__calculate_jeffreys
        
        else:                   # Normal - gamma prior
            self.prior = kwargs['prior']
            if type(self.prior[0])!= types.StringType:
                print "Warning: Jeffery's prior used as prior was \
incorectly specified"
                self.__calculate = self.__calculate_jeffreys

            else:
                ptype = self.prior[0]
                if ptype not in['normal', 'g_prior']:
                    print "Warning: Jeffery's prior used as prior was \
incorectly specified"
                    self.__calculate = self.__calculate_jeffreys
                elif ptype =='normal': 
                    assert(len(self.prior) == 3)
                    self.betaubar = self.prior[1]
                    self.vubar = self.prior[2]
                    self.__calculate = self.__calculate_normal

                else:
                    # g_prior
                    assert(len(self.prior) == 3)
                    self.betaubar = self.prior[1]
                    self.g = float(self.prior[2])
                    self.gratio = self.g/(1.+self.g)
                    self.betahat = np.zeros(self.kreg)
                    self.__calculate = self.__calculate_g_prior
                    

    def calculate(self, sigma):
        self.__calculate()
    def __calculate_jeffreys(self, sigma):
        if  self.updateind_xmat == 1 or self.updateind_yvec == 1:
            self.xpy = np.dot(self.xmat.transpose(), self.yvec)
            if self.updateind_xmat == 1: 
                self.xpx = np.dot(self.xmat.transpose(), self.xmat)
            self.updateind_xmat = 0
            self.updateind_yvec = 0

        self.betaobar = np.linalg.solve(self.xpx, self.xpy)
        self.vobar = self.xpx/sigma**2

    def __calculate_normal(self, sigma):
        self.vbubar = np.dot(self.vubar, self.betaubar)
        if  self.updateind_xmat == 1 or self.updateind_yvec == 1:
            self.xpy = np.dot(self.xmat.transpose(), self.yvec)
            if self.updateind_xmat == 1: 
                self.xpx = np.dot(self.xmat.transpose(), self.xmat)
            self.updateind_xmat = 0
            self.updateind_yvec = 0
        self.vbobar = self.xpy + self.vbubar
        self.vobar = self.vubar + self.xpx/sigma**2
        self.vbobar = self.xpy/sigma**2 + self.vbubar
        self.betaobar = np.linalg.solve(self.vobar, self.vbobar)

    def __calculate_g_prior(self, sigma):
        if  self.updateind_xmat == 1 or self.updateind_yvec == 1:
            self.xpy = np.dot(self.xmat.transpose(), self.yvec)
            if self.updateind_xmat == 1: 
                self.xpx = np.dot(self.xmat.transpose(), self.xmat)
            self.updateind_xmat = 0
            self.updateind_yvec = 0
        self.betahat = np.linalg.solve(self.xpx, self.xpy)
        self.betaobar = self.gratio * (self.betahat + self.betaubar/self.g)
        self.vobar = 1.0/(sigma**2 * self.gratio) * self.xpx

    def sample(self, sigma):
        """This function returns a sample of beta"""
        self.__calculate(sigma)
        self.cholvobar = np.linalg.cholesky(self.vobar)
        beta = self.betaobar + np.linalg.solve(self.cholvobar.T,
                                               np.random.randn(self.kreg))
        return beta
        
    def get_marginal_posterior_mean(self):
        return self.betaobar

    def get_marginal_posterior_precision(self):
        return self.vobar
    
    def update_yvec(self, yvec):
        """
        This function updates yvec in CondRegSampler. This is often useful
        when the class is being used as a part of the MCMC sampling
        scheme.
        """
        self.yvec = yvec
        self.updateind_yvec = 1

    def update_xmat(self, xmat):
        """
        This function updates xmat in CondRegSampler. This is often useful
        when the class is being used as a part of the MCMC sampling
        scheme.
        """

        if xmat.ndim == 1:
            self.xmat = xmat.reshape(xmat.shape[0], 1)
        else:
            self.xmat = xmat
        self.updateind_xmat = 1

class CondScaleSampler:
    def __init__(self, **kwargs):
        """class is used to sample sigma assuming the model is linear
        kwargs - optional arguments
            prior - is a tuple or list containing the hyperparamers that
              describe the prior. If it is not specified the Jeffrey's
              prior is used instead    
        """
        
        self.nuubar = 0.
        self.Subar = 0.
        self.__sample = self.__sample_inverted_gamma

        if 'prior' in kwargs:
            self.prior = kwargs['prior']
            priorname = self.prior[0]
            if type(self.prior[0])!= types.StringType:
                print "Warning: Jeffery's prior used as prior was \
incorectly specified"

            else: 
                if priorname not in ['gamma', 'inverted_gamma', 'wishart']:
                    print """\nWarning: Prior type unknown for \
CondSigSample. Defaulting to Jeffrey's prior\n"""

                elif priorname =='gamma':
                    self.nuubar = self.prior[1]
                    self.Subar = self.prior[2]
                    self.__sample = self.__sample_gamma

                elif priorname == 'inverted_gamma':
                    self.nuubar = self.prior[1]
                    self.Subar = self.prior[2]
                    self.__sample = self.__sample_inverted_gamma

                else:
                    #wishart prior is used
                    self.nuubar = self.prior[1]
                    self.Subar = np.atleast_2d(self.prior[2])
                    self.__sample = self.__sample_wishart2
                    self.p = self.Subar.shape[0]
                    self.work_chisq = np.arange(self.p)
                    self.n_randn = (self.p * (self.p - 1)) / 2
                    self.randnvec = np.zeros(self.n_randn)
                    self.randchivec = np.zeros(self.p)
                    self.cmat = np.zeros((self.p, self.p), order = 'F')
                    self.rmat = np.zeros((self.p, self.p), order = 'F')
                    self.umat = np.zeros((self.p, self.p), order = 'F')
                    self.Sobar = np.zeros((self.p, self.p), order = 'F')


    def sample(self, residual):
        return self.__sample(residual)

    def __sample_gamma(self, residual):
        nuobar = self.nuubar + residual.shape[0]
        Sobar = self.Subar + np.sum(residual**2, axis = 0)
        return np.random.gamma(nuobar/2., 2./Sobar)

    def __sample_inverted_gamma(self, residual):
        nuobar = self.nuubar + residual.shape[0]
        Sobar = self.Subar + np.sum(residual**2, axis = 0)
        return 1./np.sqrt(np.random.gamma(nuobar/2., 2./Sobar))

    def __sample_wishart(self, residual):
        residual = np.atleast_2d(residual)
        assert residual.shape[1] == self.p
        self.nuobar = self.nuubar + residual.shape[0]
        self.randnvec = np.random.randn(self.n_randn)
        self.randchivec = np.random.chisquare(self.nuobar - self.work_chisq)
        wishart.calc_sobar(self.Subar, self.Sobar, residual)
        self.cmat = np.asfortranarray(np.linalg.cholesky(np.linalg.inv(self.Sobar)).T)
        wishart.chol_wishart(self.randnvec, self.randchivec, self.umat,
                             self.cmat, self.rmat)

        return np.dot(self.rmat.T, self.rmat)

    def __sample_wishart2(self, residual):
        residual = np.atleast_2d(residual)
        assert residual.shape[1] == self.p
        self.nuobar = self.nuubar + residual.shape[0]
        self.randnvec = np.random.randn(self.n_randn)
        self.randchivec = np.random.chisquare(self.nuobar - self.work_chisq)
        wishart.calc_sobar(self.Subar, self.Sobar, residual)
        info = np.array(0)
        wishart.chol_wishart2(self.randnvec, self.randchivec, self.umat,
                  self.cmat, self.Sobar, info)
        assert info == 0
        return self.umat







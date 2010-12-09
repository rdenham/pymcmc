#MCMC utilities for PyMCMC - A Python package for Bayesian estimation
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
#along with this program.  If not, see <http://www.gnu.org/licenses/>.# file containing mcmc_utilites used by pymcmc.

import numpy as np
import timeseriesfunc

def hpd(x, alpha):
    '''
    highest posterior density interval
    '''
    n = len(x)
    m = max(1, np.ceil(alpha * n))
    x2 = x.copy()
    x2.sort()
    a = x2[0:m]
    b = x2[(n - m):n]
    i = np.argmin( (b - a) )
    return [a[i], b[i]]


class InefficiencyFactor:
    def __init__(self):
        self.IFactor = 0.0
        self.Bandwidth = 0.0
        self.MCSE = 0.0

    def calculate(self, mc):
        if mc.ndim == 1:
            try:
                return self.compute(mc)
            except:
                return -9999


        elif mc.ndim == 2:
            ifvec = np.zeros(mc.shape[1])
            for i in xrange(mc.shape[1]):
                try:
                    ifvec[i] = self.compute(mc[:, i])
                except:
                    ifvec[i] = -9999
            return ifvec

        else:
            ifmat = np.zeros((mc.shape[1], mc.shape[2]))
            for i in xrange(mc.shape[1]):
                for j in xrange(mc.shape[2]):
                    try:
                        ifmat[i,j] = self.compute(mc[:,i,j])
                    except:
                        ifmat[i,j] = -9999

            return ifmat

    
    def compute(self, mc):
        self.Bandwidth = np.ceil(self.calc_b(mc)) + 1
        QS = self.QSkernel(self.Bandwidth)
        corr = np.zeros(self.Bandwidth)
        timeseriesfunc.acf(mc, corr)
        product = QS * corr
        sumproduct = sum(product)
        IF = 1.+2.*(float(self.Bandwidth)/(float(self.Bandwidth) - 1.)) * sumproduct;
        return IF
    

    def QSkernel(self, B):
        ind = map(lambda x: x/B, range(1, int(B) + 1))
        ind = np.array(ind)
        d = 6.*np.pi * ind/5.
        a = 25./(12.*np.pi**2 * ind**2)
        b = np.sin(d)/d
        c = np.cos(d)
        QS = a * (b - c)
        return QS

    def calc_b(self, mc):
        n = mc.shape[0]
        xmat = np.vstack([np.ones(n - 1), mc[0:n - 1]]).transpose()
        yvec = mc[1:n]
        xpx = np.dot(xmat.transpose(), xmat)
        xpy = np.dot(xmat.transpose(), yvec)
        beta = np.linalg.solve(xpx, xpy)
        res = mc[1:n] - np.dot(xmat, beta)
        sigsq = sum(res**2)/float(n - 2)
        a = 4.*beta[1]**2 * sigsq**2/((1.-beta[1])**8)
        b = sigsq**2/((1 - beta[1])**4)
        alpha = a/b
        B = 1.3221 * (alpha * n)**(1./5.)
        return B

    # def __init__(self):
    #    self.IFactor = 0.0
    #    self.Bandwidth = 0.0
    #    self.MCSE = 0.0

    # def calculate(self, mc):
    #    if mc.ndim == 1:
    #        try:
    #            return self.compute(mc)
    #        except:
    #            return -9999


    #    else:
    #        ifvec = np.zeros(mc.shape[1])
    #        for i in xrange(mc.shape[1]):
    #            try:
    #                ifvec[i] = self.compute(mc[:, i])
    #            except:
    #                ifvec[i] =-9999
    #    return ifvec
   # 
    # def compute(self, mc):
    #    self.Bandwidth = np.ceil(self.calc_b(mc)) + 1
    #    QS = self.QSkernel(self.Bandwidth)
    #    corr = np.zeros(self.Bandwidth)
    #    timeseriesfunc.acf(mc, corr)
    #    product = QS * corr
    #    sumproduct = sum(product)
    #    IF = 1.+2.*(float(self.Bandwidth)/(float(self.Bandwidth) - 1.)) * sumproduct;
    #    return IF
   # 

    # def QSkernel(self, B):
    #    ind = map(lambda x: x/B, range(1, int(B) + 1))
    #    ind = np.array(ind)
    #    d = 6.*np.pi * ind/5.
    #    a = 25./(12.*np.pi**2 * ind**2)
    #    b = np.sin(d)/d
    #    c = np.cos(d)
    #    QS = a * (b - c)
    #    return QS

    # def calc_b(self, mc):
    #    n = mc.shape[0]
    #    xmat = np.vstack([np.ones(n - 1), mc[0:n - 1]]).transpose()
    #    yvec = mc[1:n]
    #    xpx = np.dot(xmat.transpose(), xmat)
    #    xpy = np.dot(xmat.transpose(), yvec)
    #    beta = np.linalg.solve(xpx, xpy)
    #    res = mc[1:n] - np.dot(xmat, beta)
    #    sigsq = sum(res**2)/float(n - 2)
    #    a = 4.*beta[1]**2 * sigsq**2/((1.-beta[1])**8)
    #    b = sigsq**2/((1 - beta[1])**4)
    #    alpha = a/b
    #    B = 1.3221 * (alpha * n)**(1./5.)
    #    return B


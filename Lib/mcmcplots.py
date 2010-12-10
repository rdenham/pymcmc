#Plotting tools for PyMCMC. PyMCMC is a Python package for Bayesian Analysis
#Copyright (C) 2010  Chris Strickland

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.import matplotlib.mlab as mlab

import matplotlib.pyplot as plt
import scipy.stats as stats
import timeseriesfunc
import numpy as np


latexsymbols = ['alpha', 'theta', 'tau',  \
                'beta',        'vartheta',   'pi',         'upsilon',    \
                'gamma',       'gamma',      'varpi',      'phi',        \
                'delta',       'kappa',      'rho',        'varphi',     \
                'epsilon',     'lambda',     'varrho',     'chi',        \
                'varepsilon',  'mu',         'sigma',      'psi',        \
                'zeta',        'nu',         'varsigma',   'omega',      \
                'eta',         'xi',                                         \
                'Gamma',       'Lambda',     'Sigma',      'Psi',        \
                'Delta',       'Xi',         'Upsilon',    'Omega',      \
                'Theta',       'Pi',         'Phi']


def PlotACF(x, maxlag, title):
    plt.plot([-1, maxlag], [0, 0], '-', color ='k')    
    plt.grid(True)
    # maxlag = 30
    corrv = np.zeros(maxlag)
    aa = timeseriesfunc.acf(x, corrv)
    for xtick, yval in enumerate(corrv):
        plt.plot( [xtick, xtick], [yval, 0], '-', color ='b')
    plt.title(title)
    plt.xlabel("lag")
    ## add the ci
    ci = 2/np.sqrt(len(x))
    plt.plot([0, maxlag], [ci, ci], 'b--')
    plt.plot([0, maxlag], [-ci, -ci], 'b--')
    plt.ylim(ymax = 1.2)
    return None

def PlotIterates(x, title,maxntick = 10):
    ## try only showing some of the iterates
    ## on the axis
    ##go through tick sizes until you get the right one
    for ticksize in [1,10,100,200,500,1000,2000,5000,10000,100000]:
        ntick = len(x)/ticksize
        if ntick <= maxntick:
            break
    xticks = np.arange(ntick)*ticksize
    plt.plot(x, 'k-')
    plt.xticks(xticks)
    plt.title(title)
    plt.xlabel("Iteration")


def PlotMarginalPost(x, title, plottype ="both", maxntick=20):
    '''
    Plot the marginal posterior density.
    type can be both, line, histogram
    '''
    plt.grid(True)
    ## see if we want histogram
    if plottype.startswith('b') or plottype.startswith('h'):
        n, bins, patches = plt.hist(x, 50, normed = 1, facecolor ='green', alpha = 0.75)
    if plottype.startswith('b') or plottype.startswith('l'):
        ind = np.linspace(min(x) * 1.0, max(x) * 1.0, 101)
        gkde = stats.gaussian_kde(x)
        kdepdf = gkde.evaluate(ind)
        plt.plot(ind, kdepdf, label ='kde', color ="k")
    xticks = np.round(np.linspace(min(x) * 1.0, max(x) * 1.0, maxntick),2)
    plt.xticks(xticks)
    plt.title(title)
    

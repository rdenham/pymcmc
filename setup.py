#!/usr/bin/env python

import sys
import os

## Some of the fortran files require atlas
## Most builds will use the available shared libraries
## but some need the static libraries.

# this works for ubuntu using libatlas-base-dev
# also libatlas3gf-sse
#atlas_libs = ['lapack_atlas','blas']

atlas_libs = ['atlas','lapack','f77blas','cblas']

extra_link_args = []

## you might need this if you aren't 
## using gfortran as a fortran compiler
#libs = ['gfortran']
libs = []
library_dirs=["/opt/sw/fw/rsc/atlas/3.9.25//lib/"]


#library_dirs=["/usr/lib/sse2/atlas","/usr/lib/sse2/"]
#library_dirs=['/usr/lib64/atlas/']
#######################################################
#                                                     #
# Building using Static atlas libraries...            #
#                                                     #
#######################################################

# ATLASLIB="/opt/sw/fw/rsc/atlas/3.8.3/lib/"
# extra_link_args=[""" {0}/liblapack.a {0}/libcblas.a \
# {0}/libf77blas.a {0}/libatlas.a \
# -lgfortran """.format(ATLASLIB) ]
#atlas_libs = []



## scypy_distutils Script
from numpy.distutils.core import setup, Extension
    
## setup the python module
setup(name = "pymcmc", # name of the package to import later,
      version = '1.0',
      description = """A python package for Bayesian estimation \
using Markov chain Monte Carlo""",
      author = "Christopher Strickland",
      author_email = 'christopher.strickland@qut.edu.au',
      license = "GNU GPLv3",
      ## Build fortran wrappers, uses f2py
      ext_modules = [Extension('stochsearch',
                               ['Src/stochsearch.f'],
                               libraries=atlas_libs,
                               library_dirs = library_dirs,
                               extra_link_args=extra_link_args,
                               ),
                     Extension('timeseriesfunc',
                               ['Src/timeseriesfunc.f'],
                               libraries=libs,
                               extra_link_args=[],
                               ),
                     Extension('wishart',
                               ['Src/wishart.f'],
                               libraries=atlas_libs,
                               library_dirs = library_dirs,
                               extra_link_args=extra_link_args,
                              )

                     ],
      package_dir={'pymcmc':'Lib'},
      data_files=[(os.path.join('pymcmc','examples'),
                   ['examples/ex_loglinear.py',
                    'examples/ex_AR1.py',
                    'examples/ex_variable_selection.py',
                    'examples/ex_loglinear_loop.py',
                    'examples/ex_loglinear_f2py.py',
                    'examples/ex_loglinear_weave.py',
                    'examples/loglinear.f',
                    'examples/matplotlibrc']),
                  (os.path.join('pymcmc','data'),
                   ['data/count.txt',
                   'data/yld2.txt']),
                  (os.path.join('pymcmc','doc'),
                   ['doc/PyMCMC.lyx',
                    'doc/PyMCMC.tex',
                    'doc/PyMCMC.pdf'])],
      packages = ["pymcmc"],
     )





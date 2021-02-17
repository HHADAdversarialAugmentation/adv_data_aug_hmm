# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:05:43 2021

@author: franc
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(name="funzione_gradiente", sources=["funzione_gradiente.pyx"], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize(ext, annotate=True)) 
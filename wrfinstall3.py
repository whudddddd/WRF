#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:24:53 2019

@author: ionolab
"""
import os

homepath='/home/ionolab/'

os.system('export CC=gcc')
os.system('export FC=gfortran')
os.system('export CXX=g++')
os.system('export FCFLAGS=-m64')
os.system('export F77=gfortran')
os.system('export FFLAGS=-m64')



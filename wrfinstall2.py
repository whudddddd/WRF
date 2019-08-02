"""
Created on Thu Aug  1 21:03:05 2019

@author: ionolab
"""
import os

os.chdir('/usr/local/src/hdf5-1.10.5')
os.system('./configure --prefix=/usr/local/HDF5')
os.system('make')
os.system('make check')
os.system('make install')
os.system('make check-install')

os.chdir('/usr/local/src/mpich-3.0.4')
os.system('./configure --prefix=/usr/local/HDF5')
os.system('make')
os.system('make install')

os.chdir('/usr/local/src/netcdf-4.1.3/')
os.system('export CPPFLAGS=-I/usr/local/HDF5/include')
os.system('export LDFLAGS=-L/usr/local/HDF5/lib')
os.system('export LD_LIBRARY_PATH=$/usr/local/HDF5/lib')
os.system('sudo ./configure --prefix=/usr/local/NETCDF --disable-netcdf-4')
os.system('make')
os.system('make check')
os.system('make install')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:03:05 2019

@author: ionolab
"""
import os

#try:
#    os.mkdir('/usr/bin/gccbackup')
#    os.mkdir('/usr/bin/g++backup')
#    os.mkdir('/usr/bin/gfortranbackup')
#except :
#    pass
#os.system('sudo mv /usr/bin/gcc /usr/bin/gccbackup')
#os.system('sudo mv /usr/bin/g++ /usr/bin/g++backup')
#os.system('sudo mv /usr/bin/gfortran /usr/bin/gfortranbackup')
#os.system('sudo ln -s /usr/bin/gcc-5 /usr/bin/gcc')
#os.system('sudo ln -s /usr/bin/gfortran-5 /usr/bin/gfortran')
#os.system('sudo ln -s /usr/bin/g++-5 /usr/bin/g++')
#
#os.system('sudo tar -zxf zlib-1.2.7.tar.gz  -C /usr/local/src')
#os.system('sudo tar -zxf jasper-1.900.1.tar.gz  -C /usr/local/src')
#os.system('sudo tar -zxf libpng-1.2.50.tar.gz  -C /usr/local/src')
#os.system('sudo tar -zxf mpich-3.0.4.tar.gz  -C /usr/local/src')
#os.system('sudo tar -zxf netcdf-4.1.3.tar.gz  -C /usr/local/src')
#os.system('sudo tar -zxf netcdf-4.1.3.tar.gz  -C /usr/local/src')
#os.system('sudo tar -zxf hdf5-1.10.5.tar.gz  -C /usr/local/src')

#os.chdir('/usr/local/src/zlib-1.2.7')
#os.system('sudo ./configure  --prefix=/usr/local/zlib')
#os.system('sudo make')
#os.system('sudo make check')
#os.system('sudo make install')
#os.system('sudo ./configure')
#os.system('sudo make')
#os.system('sudo make check')
#os.system('sudo make install')

os.system('sudo apt-get install curl')
os.system('/usr/local/src/libpng')
os.chdir('/usr/local/src/libpng-1.2.50')
os.system('export LDFLAGS=-L/usr/local/zlib/lib')
os.system('export CPPFLAGS=-I/usr/local/zlib/include')
os.system('./configure --prefix=/usr/local/libpng')
os.system('sudo make')
os.system('sudo make install')

os.chdir('/usr/local/src/jasper-1.900.1')
os.system('./configure --prefix=/usr/local/jasper')
os.system('sudo make')
os.system('sudo make install')

os.mkdir('/usr/local/JASPER')
os.mkdir('/usr/local/JASPER/lib')
os.mkdir('/usr/local/JASPER/include')
os.system('cp -r /usr/local/zlib/lib/* /usr/local/JASPER/lib')
os.system('cp -r /usr/local/libpng/lib/* /usr/local/JASPER/lib')
os.system('cp -r /usr/local/jasper/lib/* /usr/local/JASPER/lib')
os.system('cp -r /usr/local/zlib/include/* /usr/local/JASPER/include')
os.system('cp -r /usr/local/libpng/include/* /usr/local/JASPER/include')
os.system('cp -r /usr/local/jasper/include/* /usr/local/JASPER/include')










 




 





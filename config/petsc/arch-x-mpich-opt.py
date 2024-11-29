#!/usr/bin/env python3
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-debugging=0',
    'COPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-tau-perfstubs=0',
    
    '--with-macos-firewall-rules=1',
    '--with-clean=1',
    '--with-precision=double',
    '--with-ssl=0',
    '--with-clanguage=c',
    '--with-fc=0',
    '--download-f2cblaslapack',
    #'--with-f2cblaslapack-float128-bindings=1',
    '--download-hypre',
    '--download-make=1',
    '--download-mpich',
    '--download-mpich-device=ch3:sock',
    '--download-mpich-pm=hydra',
    '--with-cc=gcc-14',
    '--with-cxx=g++-14',
    '--with-strict-petscerrorcode',
    '--with-petsc-arch=arch-x-mpich-opt',
  ]
  configure.petsc_configure(configure_options)

#!/usr/bin/env python3
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-clean=1',
    '--with-debugging=1',
    
    '--download-metis',
    '--download-parmetis=1',
    '--with-precision=double',
    '--with-ssl=0',
    '--with-tau-perfstubs=0',
    '--with-clanguage=c',
    '--with-fc=0',   
    '--download-f2cblaslapack',
    '--download-make=1',
    '--download-mpich',
    '--download-hypre',
    '--download-mpich-device=ch4:ofi',
    '--download-mpich-pm=hydra',
    '--download-cmake=1',
    '--download-zlib=1',
    '--with-cc=gcc-14',
    '--with-cxx=g++-14',
    
    '--with-strict-petscerrorcode',
    '--with-petsc-arch=arch-x-mpich-debug',

    
  ]
  configure.petsc_configure(configure_options)

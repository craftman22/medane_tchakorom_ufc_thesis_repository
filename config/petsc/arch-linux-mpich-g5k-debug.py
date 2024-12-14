#!/usr/bin/env python3
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-debugging=1',
    '--with-clean=1',
    '--with-precision=double',
    '--with-clanguage=c',
    '--with-fc=0',   
    '--with-hypre=1',
    '--download-hypre',
    '--with-metis=1',
    '--download-metis=1'
    '--with-parmetis=1',
    '--download-parmetis=1',
    '--with-superlu_dist=1',
    '--download-superlu_dist=1',
    '--with-zlib=1',
    '--download-szlib=1',
    '--download-f2cblaslapack',
    '--with-strict-petscerrorcode',
    '--with-mumps=1',
    #'--download-mumps=1',
    #'--download-mumps-avoid-mpi-in-place=1',
    
    '--with-hwloc=1',
    '--download-hwloc=1',
    
    


    #'--with-f2cblaslapack-float128-bindings=1',
    '--download-make=1',
    '--download-mpich',
    '--download-mpich-device=ch4:ofi',
    '--download-mpich-pm=hydra',
    '--download-mpich-configure-arguments='
    '--with-cc=gcc',
    '--with-cxx=g++',
    
    
    '--download-c2html=0',
    '--download-hwloc=0',
    '--download-sowing=0',
    '--with-64-bit-indices=0',
    '--with-cgns=0',
    '--with-cuda=0',
    '--with-exodusii=0',
    '--with-fftw=0',
    '--with-giflib=0',
    '--with-gmp=0',
    '--with-hdf5=1',
    '--with-hip=0',
    '--with-hwloc=0',
    '--with-kokkos-kernels=0',
    '--with-kokkos=0',
    '--with-libjpeg=0',
    '--with-libpng=0',
    '--with-memkind=0',
    '--with-mmg=0',
    '--with-moab=0',
    '--with-mpfr=0',
    '--with-mumps=0',
    '--with-netcdf=0',
    '--with-openmp=0',
    '--with-p4est=0',
    '--with-parmmg=0',
    '--with-pnetcdf=0',
    '--with-ptscotch=0',
    '--with-random123=0',
    '--with-saws=0',
    '--with-scalapack=0',
    '--with-scalar-type=real',
    '--with-shared-libraries=1',
    '--with-ssl=0',
    '--with-strumpack=0',
    '--with-suitesparse=0',
    '--with-tetgen=0',
    '--with-trilinos=0',
    '--with-valgrind=0',
    '--with-x=0',
    '--with-yaml=0',
    
    

    '--with-petsc-arch=arch-linux-mpich-g5k-debug',
    

   
  ]
  configure.petsc_configure(configure_options)

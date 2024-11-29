#!python3
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-debugging=1',
    '--with-coverage',
    '--with-tau-perfstubs=1',
    '--download-saws',

    '--with-macos-firewall-rules=1',
    '--with-clean=1',
    '--with-precision=double',
    '--with-make-np=24',
    '--with-make-test-np=8'
    '--with-ssl=0',
    '--with-clanguage=c',
    '--with-fc=0',   
    '--download-f2cblaslapack',
    '--with-f2cblaslapack-float128-bindings=1',
    '--download-hypre',
    '--download-make=1',
    '--download-mpich',
    '--download-mpich-device=ch3:sock',
    '--download-mpich-pm=hydra',
    '--with-cc=gcc-14',
    '--with-cxx=g++-14',
    '--with-strict-petscerrorcode',
    '--with-petsc-arch=arch-x-mpich-debug',

    # '--download-spai=1',
    #'--download-hpddm=1', ***
    #'--download-slepc=1',
    #'--download-mpe=1',  ***
    # '--download-parms', ***
    #'--download-elemental=1', ***
    # '--download-fftw=1',
    #'--download-kokkos',
    #'--download-kokkos-kernels',
    #'--with-mumps',
    #'--download-zlib', ***
    #'--download-hdf5',
    #'--download-sprng',
    #'--download-suitesparse'
    #'--download-ml',
    #'--download-metis',
    #'--download-parmetis',
    #'--download-magma',
    #'--with-magma-fortran-bindings=0',
    # '--download-mfem',
    #'--download-openmpi=https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.bz2'
  ]
  configure.petsc_configure(configure_options)

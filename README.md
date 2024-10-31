
# Design of asynchronous iterative methods for domain decomposition with asynchronous minimization

This repository is dedicated to the implementation of different numerical iteratives methods for solving large linear systems of equations in the context of my thesis at University of Franche-Comte.


## Table of Contents

- [Design of asynchronous iterative methods for domain decomposition with asynchronous minimization](#project-name)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)
- [Contact](#contact)

## Features



## Installation

<!-- 
### Pre-requirements

This implementations of numerical iteratives methods is based on PETSc (Portable, Extensible Toolkit for Scientific Computation), a suite of data structures and routines for scalable (parallel) solution of scientific applications modeled by parallel differential equations (PDEs) and other related problems.

PETSc provides a comprehensive set of tools for solving linear and nonlinear equations, time-dependent problems, optimization, and other tasks related to PDEs on parallel computers. It supports parallel computations using message-passing via MPI (Message Passing Interface), making it highly efficient for large-scale distributed computations.

PETSc application is hosted on petsc.org website along with manual and tutorials. Find below the steps to install PETSc 3.22.0, the current version as this lines are written

```bash
# Clone the repository
git clone -b release https://gitlab.com/petsc/petsc.git petsc


## Installation

#### Clone the repository

```bash
git clone https://github.com/craftman22/medane_tchakorom_ufc_thesis_repository.git
```


#### Install the required dependencies

- [make](https://www.gnu.org/software/make/)
- [Python3](https://www.python.org)
- C and C++ compilers (e.g. gcc or clang)
- [OPTIONAL] Fortran compiler (e.g. gfortran)

Installating PETSc is pretty forward process. Here is a quick start tutorial for doing so. An in-place installation is recommended for all-in-one folder manageable installation. After installing PETSc, remember to checkout the PETSc v3.22.0 or above.

```bash
# Change directory into the cloned repository
cd petsc

# To anchor to a release version (without intermediate fixes), use
git checkout vMAJOR.MINOR.PATCH
```

After installation, PETSc needs to be configured with minimum options in order to be used. Below is a basic configuration using MPICH as MPI implementation and setting PETSc in debug mode.

Note: You should set the env variables PETSC_DIR and PETSC_ARCH respectively to "petsc installation folder" and "petsc arch folder"

```bash
# Change directory into the PETSc repository
cd petsc

# Run the command to configure PETSc and follow all the guidelines
./configure --download-mpich --debug=1
```

### Clone repository

After PETSc installation and configuraiton, you can clone the current repository in the desired path. Checkout the main branch for lastest release.

```bash
# Clone the repository
git clone git@github.com:craftman22/medane_tchakorom_ufc_thesis_repository.git

# Change directory into the application repository
cd medane_tchakorom_ufc_thesis_repository
``` -->


## Usage


<!-- 

```

In order to be able to build the project, one need to configure PETSc with minimal options. Below is an example of basic configuration command. This should be sufficient for building the project.

```bash
# Change directory into the PETSc installation directory
cd petsc

# Run the command to configure PETSc and follow all the guidelines
./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-mpich --download-f2cblaslapack --download-hypre --with-debugging=no
```

**Warning** : For best performances, one should turn off PETSc debugging option by setting configuration option *--with-debugging=no*. -->


<!-- 
```bash
# Change directory into the project directory
cd medane_tchakorom_ufc_thesis_repository

# Run the command make for building the project
make
``` -->





## Configuration


## License

<!-- This project is licensed under the MIT License. See the LICENSE file for more information. -->

## Contact


For questions, feedback, or collaboration requests, please contact:

* **Name:** TCHAKOROM Affo Médane
* **Email(s):** medane.tchakorom@univ-fcomte.com  or medane.tchakorom@gmail.com




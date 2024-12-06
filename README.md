
# Design of asynchronous iterative methods for domain decomposition with asynchronous minimization

This repository is dedicated to the implementation of Krylov-based iterative methods for solving large sparse linear systems of equations derived from partial differential equations (PDEs).




## Table of Contents

- [Design of asynchronous iterative methods for domain decomposition with asynchronous minimization](#project-name)
- [Installation instructions](#installation-instructions)
- [Usage examples](#usage-examples)
- [License](#license)
- [Contact](#contact)



## Installation instructions

We offer two distinct methods for deploying these implementations. The first option involves installing locally all the prerequisite packages listed in the prerequisites section. The second option entails utilizing the Docker script file included within the repository. This script facilitates the construction of a Debian-11-based image that encompasses all the prerequisite packages and this repository.



### Prerequisites


- Shell environnement (e.g bash)
- [Python 3](https://www.python.org)
- [Git](https://git-scm.com/)
- [PETSc](https://petsc.org) v3.22.1 - In-place installation is strongly recommended (referring to [PETSc manual](https://petsc.org/release/manual/) ).
- [Docker](https://www.docker.com) (OPTIONAL)

The packages listed below are not required to be pre-installed on your local system. Furthermore, PETSc strongly recommends installing these packages through its options configure system for enhanced compatibility (referring to the [configuring PETSc page](https://petsc.org/main/install/install/)). 

- Make 
- C and C++ and Fortran compilers (e.g. GNU gcc or Apple clang)
- MPI implementation - MPICH
- BLAS/LAPACK
- ...more pakages listed in the config/XXX.py files


### Environement setup

Cloning this repository on your local system is relatively straightforward using Git.

```bash
# Clone the remote repository on your local system

$ git clone https://github.com/craftman22/medane_tchakorom_ufc_thesis_repository.git
```

Next, you need to install PETSc and configure it.
Although there are many example configure scripts at config/examples/XXX.py, in the PETSc installation directory (in-place installation), we provide some bootstrap scripts for configuration. You can find them enclosed in path/to/this/repository/config/petsc/XXX.py directory.
A typical workflow to download and configure PETSc should look like this example:


```bash
# Clone the remote repository of petsc
$ git clone -b release https://gitlab.com/petsc/petsc.git petsc

# Change the working directory
$ cd petsc

# Obtain new releases fixes (since a prior clone or pull)
$ git pull

# Anchor to a release version (here we are using v3.22.1)
$ git checkout v3.22.1

# Configure petsc with your own options:
$ python3 path/to/this/repository/config/petsc/XXX.py

# Follow the remaining steps indicated upon each command termination
```

**Warning:** PETSc is configured by default in debug mode to facilitate testing and debugging tasks. However, when transitioning to the production stage, it is imperative to re-configure PETSc to disable debug mode. This modification will enable subsequent improvements in running time. In time-sensitive applications, this adjustment can be pivotal.


### Use docker script file

If you prefer not to undertake the extensive installation and configuration process, we've got your back :). Utilizing Docker, you can effortlessly construct an image based on the Docker file included in the repository. 

[Updating this part ....]





## Usage examples

Open an shell interactive environnement and change your directory to this repository local clone.

```bash
# Change the working directory to the cloned local repository

$ cd medane_tchakorom_ufc_thesis_repository
```

Few commands are available through Make utility command line in order to provide common build options:


```bash
$ make build  # Compile and link source code in binary, which will be stored in ./bin folder

$ make clean # Delete the previous build files

$ make print # Print infos about compilers or wrapper compilers, libraries and flags
```


A shell script (iSolve) is available in order to launch experiment in a more user-friendly approach.
A basic workflow should look like this.

```bash
# Open ./config/default_run_variables file with your favorite editor (We use nano here)
# This file contains values that can be used directly from the specified binary to be run. 
# We describe each entry option in the file (algorithm type, number of processes ....)
# Some options are taged with the text (DO NOT MODIFY). This is self explanatory
# Set the options to the desired values, then save file and close the editor

$ nano ./config/default_run_variables

# Next, launch the experiment. You should get in this order, some infos on your screen
# 1. Summary of key-value options from default_run_variables file
# 2. The summary of the command to be run
# 3. Results of the expiriment

$ ./iSolve
```

The script iSolve can be useful, if you want to override default values provided in the default_run_variables file. Run the command below to show the different options that can be passed to the script

```bash
# Run this command to get an overview of the options for the iSolve script

$ ./iSolve --help
```

For example, if you edited default_run_variables file, and then want you modify only the algorithm that is run, no need to edit again the default_run_variables file, you could simply override the default value of the algorithm variable by specifying it on the command line like this.

```bash
# Here we ran the SMSM algorithm, which will clearly override the algorith name mentionned in the default_run_variables file

$ ./iSolve --alg SMSM_DirRes
```

To override the total number of processes and number of processes per block, keeping the other default values intact, run the command below

```bash
# Here we ran the AMAM algorithm, which will clearly override the algorith name mentionned in the default_run_variables file

$ ./iSolve --alg SMSM_DirRes --np 8 --npb 4
```

Last example consist on overriding the size of the mesh. Pretty straight forward:

```bash
# Here we ran the AMAM algorithm, which will clearly override the algorith name mentionned in the default_run_variables file

$ ./iSolve --alg SMSM_DirRes --m 200 --n 200
```



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact


For questions, feedback, or collaboration requests, please contact:

* **Name:** TCHAKOROM Affo Médane
* **Email(s):** medane.tchakorom@univ-fcomte.fr  or medane.tchakorom@gmail.com




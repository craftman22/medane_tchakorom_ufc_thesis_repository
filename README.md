# Design of asynchronous iterative methods for domain decomposition with asynchronous minimization

This repository is dedicated to the implementation of different numerical iteratives methods for solving large linear systems of equations in the context of my thesis at University of Franche-Comte.


## Table of Contents

- [Design of asynchronous iterative methods for domain decomposition with asynchronous minimization](#project-name)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- List the main features or benefits of your project here.
- Feature 1: Brief description of the feature.
- Feature 2: Another cool feature you want to highlight.

## Installation

This implementations of numerical iteratives methods is based on PETSc (Portable, Extensible Toolkit for Scientific Computation), a suite of data structures and routines for scalable (parallel) solution of scientific applications modeled by parallel differential equations (PDEs) and other related problems.

PETSc provides a comprehensive set of tools for solving linear and nonlinear equations, time-dependent problems, optimization, and other tasks related to PDEs on parallel computers. It supports parallel computations using message-passing via MPI (Message Passing Interface), making it highly efficient for large-scale distributed computations.

PETSc application is hosted on petsc.org website along with manual and tutorials. Find below the steps to install PETSc ..., the current version as this line are written

```bash
# Clone the repository
git clone -b release https://gitlab.com/petsc/petsc.git petsc

# Change directory into the cloned repository
cd petsc

# To anchor to a release version (without intermediate fixes), use
git checkout vMAJOR.MINOR.PATCH



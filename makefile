# -*- mode: makefile -*-

#  This SAMPLE GNU Makefile can be used to compile PETSc applications
#  It relies on pkg_config tool (see $PETSC_DIR/share/petsc/Makefile.basic.user if you cannot use pkg_config)
#  Copy this file to your source directory as "Makefile" and MODIFY AS NEEDED (see the comment above CFLAGS below).
#
#  You must set the environmental variable(s) PETSC_DIR (and PETSC_ARCH if PETSc was not configured with the --prefix option)
#  See also share/petsc/Makefile.basic.user for a makefile that does not require pkg_config
#
#  For example - a single source file (ex1.c or ex1.F90) can be compiled with:
#
#      make ex1
#
#  You do not need to edit this makefile at all.
#
#  For a multi-file case, suppose you have the source files a.c, b.c, and c.cxx
#  This can be built by uncommenting the following two lines.
#
# app : a.o b.o c.o
# 	$(LINK.C) -o $@ $^ $(LDLIBS)
#
#  When linking in a multi-files with Fortran source files a.F90, b.c, and c.cxx
#  You may need to use
#
# app : a.o b.o c.o
# 	$(LINK.F) -o $@ $^ $(LDLIBS)

#  If the file c.cxx needs to link with a C++ standard library -lstdc++ , then
#  you'll need to add it explicitly.  It can go in the rule above or be added to
#  a target-specific variable by uncommenting the line below.
#
# app : LDLIBS += -lstdc++
#
#  The following variable must either be a path to petsc.pc or just "petsc" if petsc.pc
#  has been installed to a system location or can be found in PKG_CONFIG_PATH.
petsc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/petsc.pc

# Additional libraries that support pkg-config can be added to the list of PACKAGES below.
PACKAGES := $(petsc.pc)

# The following variables may be removed if desired
# They pass all the flags that were used to compile PETSc to compile your code, they are generally not needed

CFLAGS_OTHER := $(shell pkg-config --cflags-only-other $(PACKAGES))
CFLAGS := $(shell pkg-config --variable=cflags_extra $(PACKAGES)) $(CFLAGS_OTHER)
CXXFLAGS := $(shell pkg-config --variable=cxxflags_extra $(PACKAGES)) $(CFLAGS_OTHER)
FFLAGS := $(shell pkg-config --variable=fflags_extra $(PACKAGES))
CPPFLAGS := $(shell pkg-config --cflags-only-I $(PACKAGES))

CC := $(shell pkg-config --variable=ccompiler $(PACKAGES))
CXX := $(shell pkg-config --variable=cxxcompiler $(PACKAGES))
FC := $(shell pkg-config --variable=fcompiler $(PACKAGES))
LDFLAGS := $(shell pkg-config --libs-only-L --libs-only-other $(PACKAGES))
LDFLAGS += $(patsubst -L%, $(shell pkg-config --variable=ldflag_rpath $(PACKAGES))%, $(shell pkg-config --libs-only-L $(PACKAGES)))
LDLIBS := $(shell pkg-config --libs-only-l $(PACKAGES)) -lm
CUDAC := $(shell pkg-config --variable=cudacompiler $(PACKAGES))
CUDAC_FLAGS := $(shell pkg-config --variable=cudaflags_extra $(PACKAGES))
CUDA_LIB := $(shell pkg-config --variable=cudalib $(PACKAGES))
CUDA_INCLUDE := $(shell pkg-config --variable=cudainclude $(PACKAGES))




print:
	@echo CC=$(CC)
	@echo CXX=$(CXX)
	@echo FC=$(FC)
	@echo CFLAGS=$(CFLAGS)
	@echo CXXFLAGS=$(CXXFLAGS)
	@echo FFLAGS=$(FFLAGS)
	@echo CPPFLAGS=$(CPPFLAGS)
	@echo LDFLAGS=$(LDFLAGS)
	@echo LDLIBS=$(LDLIBS)
	@echo CUDAC=$(CUDAC)
	@echo CUDAC_FLAGS=$(CUDAC_FLAGS)
	@echo CUDA_LIB=$(CUDA_LIB)
	@echo CUDA_INCLUDE=$(CUDA_INCLUDE)

# Many suffixes are covered by implicit rules, but you may need to write custom rules
# such as these if you use suffixes that do not have implicit rules.
# https://www.gnu.org/software/make/manual/html_node/Catalogue-of-Rules.html#Catalogue-of-Rules
% : %.F90
	$(LINK.F) -o $@ $^ $(LDLIBS)
%.o: %.F90
	$(COMPILE.F) $(OUTPUT_OPTION) $<
% : %.cxx
	$(LINK.cc) -o $@ $^ $(LDLIBS)
%.o: %.cxx
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
%.o : %.cu
	$(CUDAC) -c $(CPPFLAGS) $(CUDAC_FLAGS) $(CUDA_INCLUDE) -o $@ $<
# %: %.c
# $(LINK.c) $^ $(LOADLIBES) $(LDLIBS) -o $@





.PHONY: all clean build help
.DEFAULT_GOAL:= all



# List the source directories you want to compile from

# All
# SRC_DIRS :=  src/synchronous-multisplitting  src/asynchronous-multisplitting  src/synchronous-multisplitting-synchronous-minimization-local src/synchronous-multisplitting-synchronous-minimization-semi-local src/synchronous-multisplitting-synchronous-minimization-global  src/asynchronous-multisplitting-asynchronous-minimization-global src/asynchronous-multisplitting-asynchronous-minimization-local src/asynchronous-multisplitting-asynchronous-minimization-semi-local src/gmres_solution 

# Asynchronous
# SRC_DIRS :=  src/synchronous-multisplitting   src/synchronous-multisplitting-synchronous-minimization-local src/synchronous-multisplitting-synchronous-minimization-semi-local src/synchronous-multisplitting-synchronous-minimization-global   src/gmres_solution

# Synchronous
#SRC_DIRS :=   src/asynchronous-multisplitting src/asynchronous-multisplitting-asynchronous-minimization-global src/asynchronous-multisplitting-asynchronous-minimization-local src/asynchronous-multisplitting-asynchronous-minimization-semi-local src/gmres_solution

#experimenting
# SRC_DIRS :=  src/experimenting
# SRC_DIRS :=  src/synchronous-multisplitting  src/asynchronous-multisplitting  src/synchronous-multisplitting-synchronous-minimization-local  src/synchronous-multisplitting-synchronous-minimization-global  src/asynchronous-multisplitting-asynchronous-minimization-global src/asynchronous-multisplitting-asynchronous-minimization-local src/synchronous-multisplitting-synchronous-minimization-semi-local src/gmres_solution 
SRC_DIRS :=  src/asynchronous-multisplitting





# Define the directory where binaries will be stored
BIN_DIR := bin

# List of source files in chosen directories
SOURCES := $(wildcard $(foreach dir, $(SRC_DIRS), $(dir)/*.c))

# Corresponding binaries in the bin directory
#BINARIES := $(patsubst src/%.c, $(BIN_DIR)/% ,  $(SOURCES))


BINARIES := $(patsubst src/%.c, $(BIN_DIR)/%,   $(SOURCES))


# Default rule to build all binaries
all: $(BINARIES) 
	@echo Build finish!


build: all

# Rule to compile each binary
$(BIN_DIR)/%: src/%.c | $(BIN_DIR)
	@if echo "$(@)" | grep -q "asynchronous"; then \
		$(LINK.c) -DVERSION_1_0 src/utils/utils.c src/utils/comm.c src/utils/conv_detection.c  $^ $(LOADLIBES) $(LDLIBS) -I./include   -o $(BIN_DIR)/$(notdir $@)"-v1.0"; \
		$(LINK.c) -DVERSION_1_1 src/utils/utils.c src/utils/comm.c src/utils/conv_detection.c  $^ $(LOADLIBES) $(LDLIBS) -I./include   -o $(BIN_DIR)/$(notdir $@)"-v1.1"; \
	else \
		$(LINK.c) -DVERSION_1_0 src/utils/utils.c src/utils/comm.c src/utils/conv_detection.c $^ $(LOADLIBES) $(LDLIBS) -I./include   -o $(BIN_DIR)/$(notdir $@)"-v1.0"; \
	fi



# $(BIN_DIR)/%: src/%.c | $(BIN_DIR)
# 	@if echo "$(@)" | grep -q "local"; then \
# 		$(LINK.c) -DVERSION_1_0 src/utils/utils.c src/utils/comm.c  $^ $(LOADLIBES) $(LDLIBS) -I./include   -o $(BIN_DIR)/$(notdir $@)"-v1.0"; \
# 		$(LINK.c) -DVERSION_1_1 src/utils/utils.c src/utils/comm.c  $^ $(LOADLIBES) $(LDLIBS) -I./include   -o $(BIN_DIR)/$(notdir $@)"-v1.1"; \
# 	elif [ "$(notdir $@)" = "asynchronous-multisplitting" ]; then \
# 		$(LINK.c) -DVERSION_1_0 src/utils/utils.c src/utils/comm.c  $^ $(LOADLIBES) $(LDLIBS) -I./include   -o $(BIN_DIR)/$(notdir $@)"-v1.0"; \
# 		$(LINK.c) -DVERSION_1_1 src/utils/utils.c src/utils/comm.c  $^ $(LOADLIBES) $(LDLIBS) -I./include   -o $(BIN_DIR)/$(notdir $@)"-v1.1"; \
# 	else \
# 		$(LINK.c)  src/utils/utils.c src/utils/comm.c  $^ $(LOADLIBES) $(LDLIBS) -I./include   -o $(BIN_DIR)/$(notdir $@); \
# 	fi

	
	
# Ensure the bin directory exists
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)



run:
	@echo Not implemented yet...
	@echo Future feature

check:
	@$(LINK.c) src/utils/utils.c  src/Unity/unity.c  ./src/tests/utils_test.c $(LOADLIBES) $(LDLIBS) -I./include -I./src/Unity  -o $(BIN_DIR)/utils_test
	@printf "\n***************************** Running with 4 process *****************************\n"
	@mpirun -n 4 ./bin/utils_test

docs:
	@echo Not implemented yet...
	@echo Future feature

rebuild: clean all

# Clean up all binaries
clean:
	@echo Removing all build files from bin...
	@rm -rf $(BIN_DIR)/*
	@echo Finish!	
	



# Global env variable
export PETSC_DIR=/home/mtchakorom/petsc
export PETSC_ARCH=arch-linux-c-opt-g5k



#Open MP configurations env variables
export OMP_DISPLAY_ENV=false
export OMP_DISPLAY_AFFINITY=false
export OMP_PROC_BIND=close
# export OMP_PLACES=cores
export OMP_PLACES=threads
export OMP_NUM_THREADS=1




#OpenMPI configurations env variables

export OMPI_MCA_opal_net_private_ipv4="192.168.0.0/16"
export OMPI_MCA_btl_tcp_if_exclude=ib0,lo
export OMPI_MCA_pml=^ucx,cm
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_orte_keep_fqdn_hostnames=1

#overwritten by petsc "mpiexec --oversubscribe" config
export OMPI_MCA_rmaps_base_oversubscribe=1

export OMPI_MCA_hwloc_base_report_bindings=1
export OMPI_MCA_rmaps_base_mapping_policy=socket
export OMPI_MCA_hwloc_base_binding_policy=l2cache
# export OMPI_MCA_hwloc_base_binding_policy=core
export OMPI_MCA_rmaps_base_ranking_policy=slot

# export MPI_BINDING="-hostfile ./hostfiles/default"
export MPI_BINDING=""

#===================================================


# unset PETSC_DIR
# unset PETSC_ARCH




# unset OMP_DISPLAY_ENV
# unset OMP_DISPLAY_AFFINITY
# unset OMP_PROC_BIND
# unset OMP_PLACES
# unset OMP_NUM_THREADS





# unset OMPI_MCA_opal_net_private_ipv4
# unset OMPI_MCA_btl_tcp_if_exclude
# unset OMPI_MCA_pml
# unset OMPI_MCA_btl
# unset OMPI_MCA_orte_keep_fqdn_hostnames


# unset OMPI_MCA_hwloc_base_report_bindings
# unset OMPI_MCA_rmaps_base_oversubscribe
# unset OMPI_MCA_rmaps_base_mapping_policy
# unset OMPI_MCA_hwloc_base_binding_policy
# unset OMPI_MCA_rmaps_base_ranking_policy


# unset MPI_BINDING
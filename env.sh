# Global env variable
export PETSC_DIR=/home/mtchakorom/petsc
export PETSC_ARCH=arch-linux-c-opt-g5k



#Open MP configurations env variables
export OMP_DISPLAY_ENV=false
export OMP_DISPLAY_AFFINITY=false
export OMP_PROC_BIND=false
export OMP_PLACES=cores
export OMP_NUM_THREADS=1
# unset OMP_PLACES



#OpenMPI configurations env variables

export OMPI_MCA_opal_net_private_ipv4="192.168.0.0/16"
export OMPI_MCA_btl_tcp_if_exclude=ib0,lo
export OMPI_MCA_pml=^ucx,cm
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_orte_keep_fqdn_hostnames=1


export OMPI_MCA_hwloc_base_report_bindings=1
export OMPI_MCA_rmaps_base_oversubscribe=0
export OMPI_MCA_hwloc_base_binding_policy=core
export OMPI_MCA_rmaps_base_mapping_policy=socket
export OMPI_MCA_rmaps_base_ranking_policy=slot

export MPI_BINDING="-hostfile ./hostfiles/default"


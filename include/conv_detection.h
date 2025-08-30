
#ifndef SHARED_CONV_DETECTION_FUNCTIONS_H
#define SHARED_CONV_DETECTION_FUNCTIONS_H

PetscErrorCode comm_async_convDetection(PetscMPIInt rank_jacobi_block, PetscInt nbNeighbors, PetscInt *nbNeigNotLCV, PetscInt *neighbors, PetscInt *prevIterNumS, PetscInt *prevIterNumC, PetscInt *nbIterPreLocalCV, PetscBool *preLocalCV, PetscBool *sLocalCV, PetscBool *globalCV, PetscMPIInt *dest_node, PetscInt THRESHOLD_SLCV, PetscInt current_iteration, PetscMPIInt *cancelSPartialBuffer, MPI_Request *cancelSPartialRequest, PetscMPIInt *sendSPartialBuffer, MPI_Request *sendSPartialRequest);

PetscErrorCode comm_async_recvCancelSPartialCV(PetscMPIInt rank_jacobi_block, PetscInt *nbNeigNotLCV, PetscInt nbNeighbors,PetscInt *prevIterNumS, PetscInt *prevIterNumC, PetscBool *globalCV);

PetscErrorCode comm_async_recvSPartialCV(PetscMPIInt rank_jacobi_block, PetscInt *nbNeigNotLCV, PetscInt *prevIterNumS, PetscInt *prevIterNumC);

PetscErrorCode comm_async_recvGlobalCV(PetscMPIInt rank_jacobi_block, PetscBool *globalCV);

PetscErrorCode comm_async_sendGlobalCV(PetscMPIInt rank_jacobi_block, PetscInt nbNeighbors, PetscInt *neighbors, PetscBool *globalCV, MPI_Request *requests);




PetscErrorCode build_spanning_tree(PetscMPIInt rank_jacobi_block, PetscInt *neighbors, PetscInt *nbNeighbors, PetscMPIInt proc_local_rank, PetscMPIInt nprocs_per_jacobi_block );

// PetscErrorCode comm_sync_measure_latency_between_two_nodes(PetscMPIInt proc_rank_node_1, PetscMPIInt proc_rank_node_2, PetscMPIInt actual_rank);

#endif // SHARED_CONV_DETECTION_FUNCTIONS_H
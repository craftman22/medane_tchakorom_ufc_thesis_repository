#include "constants.h"
#include "utils.h"
// #include "comm.h"
#include "conv_detection.h"

PetscErrorCode comm_async_convDetection(PetscMPIInt rank_jacobi_block, PetscInt nbNeighbors, PetscInt *nbNeigNotLCV, PetscInt *neighbors, PetscInt *prevIterNumS, PetscInt *prevIterNumC, PetscInt *nbIterPreLocalCV, PetscBool *preLocalCV, PetscBool *sLocalCV, PetscBool *globalCV, PetscMPIInt *dest_node, PetscInt THRESHOLD_SLCV, PetscInt current_iteration, PetscMPIInt *cancelSPartialBuffer, MPI_Request *cancelSPartialRequest, PetscMPIInt *sendSPartialBuffer, MPI_Request *sendSPartialRequest)
{
    PetscFunctionBeginUser;
    PetscInt flag = 0;

    if ((*sLocalCV) == PETSC_FALSE)
    {
        if ((*preLocalCV) == PETSC_TRUE)
        {
            (*nbIterPreLocalCV) = (*nbIterPreLocalCV) + 1;
            if ((*nbIterPreLocalCV) == THRESHOLD_SLCV)
            {
                (*sLocalCV) = PETSC_TRUE;
            }
        }
        else
        {
            (*nbIterPreLocalCV) = 0;
        }
    }
    else
    {

        if ((*preLocalCV) == PETSC_FALSE)
        {
            (*sLocalCV) = PETSC_FALSE;
            (*nbIterPreLocalCV) = 0;
            /* if a sPartialCV message has already been sent, cancel it , "dest_node"
            is the last neighbor to which convergence message has been sent .
            To obtain it, whe should loop through an array containing the state of all node neighbors OR
            check the neighbor for which prevIterNumC[i] < prevIterNumS[i].
            dest_node == -1 indicate that no cv message has been sent yet from this node, otherwise, it contains the
            number of the node to which we should send the cancelation message */
            if ((*dest_node) != -1)
            {
                if ((*cancelSPartialRequest) != MPI_REQUEST_NULL)
                    PetscCallMPI(MPI_Test(cancelSPartialRequest, &flag, MPI_STATUS_IGNORE));
                else
                    flag = 1;

                if (flag)
                {
                    (*cancelSPartialBuffer) = current_iteration;
                    PetscCallMPI(MPI_Isend(cancelSPartialBuffer, 1, MPIU_INT, (*dest_node), TAG_CANCEL_CV, MPI_COMM_WORLD, cancelSPartialRequest));
                }
            }
        }
        else
        {
            if ((*nbNeigNotLCV) == 0)
            {
                (*globalCV) = PETSC_TRUE;
            }
            else
            {

                if ((*nbNeigNotLCV) == 1)
                {
                    (*dest_node) = neighbors[0]; // This is straighforward as there is just 2 nodes involved, each one has only one neighbor.
                    if ((*sendSPartialRequest) != MPI_REQUEST_NULL)
                        PetscCallMPI(MPI_Test(sendSPartialRequest, &flag, MPI_STATUS_IGNORE));
                    else
                        flag = 1;

                    if (flag)
                    {
                        (*sendSPartialBuffer) = current_iteration;
                        PetscCallMPI(MPI_Isend(sendSPartialBuffer, 1, MPIU_INT, (*dest_node), TAG_SEND_CV, MPI_COMM_WORLD, sendSPartialRequest));
                    }
                }
            }
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_async_recvSPartialCV(PetscMPIInt rank_jacobi_block, PetscInt *nbNeigNotLCV, PetscInt *prevIterNumS, PetscInt *prevIterNumC)
{

    PetscFunctionBeginUser;

    PetscMPIInt flag = 0;
    MPI_Status status;
    PetscInt buff = -1;
    PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_CV, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
    if (flag)
    {

        PetscCallMPI(MPI_Recv(&buff, 1, MPIU_INT, MPI_ANY_SOURCE, TAG_SEND_CV, MPI_COMM_WORLD, &status));

        PetscInt srcNode = status.MPI_SOURCE;
        PetscInt currentIterNum = buff;

        if ((prevIterNumS[srcNode] < prevIterNumC[srcNode]) && (prevIterNumC[srcNode] < currentIterNum))
        {
            (*nbNeigNotLCV) = (*nbNeigNotLCV) - 1;
            if ((*nbNeigNotLCV) < 0)
                (*nbNeigNotLCV) = 0;
        }

        if (prevIterNumS[srcNode] < currentIterNum)
        {
            prevIterNumS[srcNode] = currentIterNum;
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_async_recvCancelSPartialCV(PetscMPIInt rank_jacobi_block, PetscInt *nbNeigNotLCV, PetscInt nbNeighbors, PetscInt *prevIterNumS, PetscInt *prevIterNumC, PetscBool *globalCV)
{

    PetscFunctionBeginUser;

    PetscMPIInt flag;
    PetscInt buff;
    MPI_Status status;
    PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_CANCEL_CV, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
    if (flag)
    {
        PetscCallMPI(MPI_Recv(&buff, 1, MPIU_INT, MPI_ANY_SOURCE, TAG_CANCEL_CV, MPI_COMM_WORLD, &status));

        PetscInt srcNode = status.MPI_SOURCE;
        PetscInt currentIterNum = buff;

        if ((prevIterNumC[srcNode] < prevIterNumS[srcNode]) && (prevIterNumS[srcNode] < currentIterNum))
        {
            (*nbNeigNotLCV) = (*nbNeigNotLCV) + 1;
            if ((*nbNeigNotLCV) > nbNeighbors)
                (*nbNeigNotLCV) = nbNeighbors;
            (*globalCV) = PETSC_FALSE;
        }

        if (prevIterNumC[srcNode] < currentIterNum)
        {
            prevIterNumC[srcNode] = currentIterNum;
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_async_recvGlobalCV(PetscMPIInt rank_jacobi_block, PetscBool *globalCV)
{

    PetscFunctionBeginUser;
    PetscMPIInt flag = 0;
    PetscCallMPI(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_RCV_GLOBAL_CV, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
    if (flag)
    {
        PetscCallMPI(MPI_Recv(globalCV, 1, MPIU_BOOL, MPI_ANY_SOURCE, TAG_SEND_RCV_GLOBAL_CV, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_async_sendGlobalCV(PetscMPIInt rank_jacobi_block, PetscInt nbNeighbors, PetscInt *neighbors, PetscBool *globalCV, MPI_Request *requests)
{
    PetscFunctionBeginUser;

    for (PetscInt i = 0; i < nbNeighbors; i++)
    {
        PetscCallMPI(MPI_Isend(globalCV, 1, MPIU_BOOL, neighbors[i], TAG_SEND_RCV_GLOBAL_CV, MPI_COMM_WORLD, &requests[i]));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    This function is built specifically for 2 blocks.
    So using just local root node in the spanning tree,
    each local root node is the neighbor of the other in the PETSC_COMM_WORLD
    No further complexity for the moment
*/
PetscErrorCode build_spanning_tree(PetscMPIInt rank_jacobi_block, PetscInt *neighbors, PetscInt *nbNeighbors, PetscMPIInt proc_local_rank, PetscMPIInt nprocs_per_jacobi_block)
{
    PetscFunctionBeginUser;

    (*nbNeighbors) = 1;
    if (rank_jacobi_block == BLOCK_RANK_ZERO && proc_local_rank == 0)
    {
        neighbors[0] = nprocs_per_jacobi_block;
    }

    if (rank_jacobi_block == BLOCK_RANK_ONE && proc_local_rank == 0)
    {
        neighbors[0] = 0;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscErrorCode build_spanning_tree(PetscMPIInt rank_jacobi_block, PetscInt *neighbors, PetscInt *nbNeighbors, PetscMPIInt proc_local_rank, PetscMPIInt proc_global_rank, PetscMPIInt nprocs_per_jacobi_block)
// {
//     PetscFunctionBeginUser;

//     (*nbNeighbors) = 1;

//     if (proc_local_rank == 0)
//     {
//         (*nbNeighbors) = nprocs_per_jacobi_block;
//         for (PetscMPIInt i = 0; i < (*nbNeighbors); i++)
//         {
//             neighbors[i] = proc_global_rank + (i + 1);
//         }

//         if (proc_global_rank == 0)
//         {
//             neighbors[(*nbNeighbors) - 1] = nprocs_per_jacobi_block;
//         }
//         else
//         {
//             neighbors[(*nbNeighbors) - 1] = 0;
//         }
//     }
//     else
//     {
//         (*nbNeighbors) = 1;
//         neighbors[0] = (rank_jacobi_block * nprocs_per_jacobi_block);
//     }

//     PetscFunctionReturn(PETSC_SUCCESS);
// }
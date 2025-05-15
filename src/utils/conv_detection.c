#include "constants.h"
#include "utils.h"
#include "comm.h"
#include "conv_detection.h"

PetscErrorCode comm_async_convDetection(PetscMPIInt rank_jacobi_block, PetscInt nbNeighbors, PetscInt *nbNeigNotLCV, PetscInt *neighbors, PetscInt *prevIterNumS, PetscInt *prevIterNumC, PetscInt *nbIterPreLocalCV, PetscBool *preLocalCV, PetscBool *sLocalCV, PetscBool *globalCV, PetscMPIInt *dest_node, PetscInt THRESHOLD_SLCV, PetscInt current_iteration, PetscMPIInt *cancelSPartialBuffer, MPI_Request *cancelSPartialRequest, PetscMPIInt *sendSPartialBuffer, MPI_Request *sendSPartialRequest)
{
    PetscFunctionBeginUser;

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

        // printf("ARRET ICI  rank block %d sPartialCV  iteration %d \n", rank_jacobi_block, (*cancelSPartialBuffer));
        if ((*preLocalCV) == PETSC_FALSE)
        {
            (*sLocalCV) = PETSC_FALSE;
            (*nbIterPreLocalCV) = 0;
            // if a sPartialCV message has already been sent
            //  then
            //   send a cancelSPartialCV message with the
            //   current iteration number to the same destination neighbor
            // end if
            if ((*dest_node) >= 0)
            {
                (*cancelSPartialBuffer) = current_iteration;
                PetscCallMPI(MPI_Isend(cancelSPartialBuffer, 1, MPIU_INT, (*dest_node), TAG_CANCEL_CV, MPI_COMM_WORLD, cancelSPartialRequest));
                // printf("ARRET ICI envoi cancel rank block %d destination %d , iteration %d \n", rank_jacobi_block, (*dest_node), (*cancelSPartialBuffer));
            }
        }
        else
        {
            // printf("ARRET ICI  presque arrive rank block  %d  iteration %d  nbNeigNotLCV %d\n", rank_jacobi_block, (*sendSPartialBuffer), (*nbNeigNotLCV));
            if ((*nbNeigNotLCV) == 0)
            {
                (*globalCV) = PETSC_TRUE;
                // printf("ARRET ICI  rank block %d GLOBAL CV  iteration %d \n", rank_jacobi_block, (*sendSPartialBuffer));
            }
            else
            {
                if ((*nbNeigNotLCV) == 1)
                {
                    // send a sPartialCV message with the
                    // current iteration to the last neighbor
                    // corresponding to nbNeigNotLCV
                    // for (PetscInt i = 0; i < nbNeighbors; i++)
                    // {
                    //     if (prevIterNumS[neighbors[i]] < prevIterNumC[neighbors[i]])
                    //     {
                    (*dest_node) = neighbors[0]; //TODO: revoir cette partie
                    //     }
                    // }

                    PetscCallMPI(MPI_Isend(sendSPartialBuffer, 1, MPIU_INT, (*dest_node), TAG_SEND_CV, MPI_COMM_WORLD, sendSPartialRequest));
                    // printf("ARRET ICI envoi rank block %d destination %d , iteration %d \n", rank_jacobi_block, (*dest_node), (*sendSPartialBuffer));
                }
            }
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_async_recvSPartialCV(PetscMPIInt rank_jacobi_block, PetscInt *nbNeigNotLCV, PetscInt *prevIterNumS, PetscInt *prevIterNumC)
{

    PetscFunctionBeginUser;

    PetscMPIInt flag;
    MPI_Status status;
    PetscMPIInt buff;
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
        // printf("ARRET ICI reception rank block %d nbNeigNotLCV  %d\n", rank_jacobi_block, (*nbNeigNotLCV));

        // PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_CV, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_async_recvCancelSPartialCV(PetscMPIInt rank_jacobi_block, PetscInt *nbNeigNotLCV, PetscInt *prevIterNumS, PetscInt *prevIterNumC, PetscBool *globalCV)
{

    PetscFunctionBeginUser;

    PetscMPIInt flag;
    PetscMPIInt buff;
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
            if ((*nbNeigNotLCV) > 1)
                (*nbNeigNotLCV) = 1;
            (*globalCV) = PETSC_FALSE;
        }

        if (prevIterNumC[srcNode] < currentIterNum)
        {
            prevIterNumC[srcNode] = currentIterNum;
        }

        // printf("ARRET ICI  rank block %d reception cancel sPartialcv , source node %d  \n", rank_jacobi_block, srcNode);
        // PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_CANCEL_CV, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_async_recvGlobalCV(PetscMPIInt rank_jacobi_block, PetscBool *globalCV)
{

    PetscFunctionBeginUser;
    PetscMPIInt flag = 0;
    PetscMPIInt globalCVBuffer;
    PetscCallMPI(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_RCV_GLOBAL_CV, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
    if (flag)
    {
        PetscCallMPI(MPI_Recv(&globalCVBuffer, 1, MPIU_INT, MPI_ANY_SOURCE, TAG_SEND_RCV_GLOBAL_CV, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        (*globalCV) = PETSC_TRUE;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_async_sendGlobalCV(PetscMPIInt rank_jacobi_block, PetscInt nbNeighbors, PetscInt *neighbors, PetscMPIInt *buff, MPI_Request *requests)
{
    PetscFunctionBeginUser;

    for (PetscInt i = 0; i < nbNeighbors; i++)
    {
        PetscCallMPI(MPI_Isend(buff, 1, MPIU_INT, neighbors[i], TAG_SEND_RCV_GLOBAL_CV, MPI_COMM_WORLD, &requests[i]));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode build_spanning_tree(PetscMPIInt rank_jacobi_block, PetscInt *neighbors, PetscInt *nbNeighbors, PetscMPIInt proc_local_rank, PetscMPIInt proc_global_rank, PetscMPIInt nprocs_per_jacobi_block)
{
    PetscFunctionBeginUser;

    if (proc_local_rank == 0)
    {
        (*nbNeighbors) = nprocs_per_jacobi_block;
        for (PetscMPIInt i = 0; i < (*nbNeighbors); i++)
        {
            neighbors[i] = proc_global_rank + (i + 1);
        }

        if (proc_global_rank == 0)
        {
            neighbors[(*nbNeighbors) - 1] = nprocs_per_jacobi_block;
        }
        else
        {
            neighbors[(*nbNeighbors) - 1] = 0;
        }
    }
    else
    {
        (*nbNeighbors) = 1;
        neighbors[0] = (rank_jacobi_block * nprocs_per_jacobi_block);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

#include "constants.h"
#include "utils.h"
#include "conv_detection_prime.h"

/*
    Algorithm 5.15 (Practical version of convergence detection)
    from the book : Parallel Iterative Algorithms: from Sequential to Grid Computing

*/

PetscErrorCode comm_async_convDetection_prime(PARAMS)
{
    PetscFunctionBeginUser;

    if ((*state) == NORMAL)
    {
        if ((*UnderThreashold) == PETSC_FALSE)
        {
            PetscCall(reinitialize_pseudo_period(ACTUAL_PARAMS));
        }
        else
        {
            if ((*PseudoPeriodBegin) == PETSC_FALSE)
            {
                (*PseudoPeriodBegin) = PETSC_TRUE;
            }
            else
            {
                if ((*PseudoPeriodEnd) == PETSC_TRUE)
                {
                    (*LocalCV) = PETSC_TRUE;
                    if ((*NbNotRecvd) == 0)
                    {
                        (*ElectedNode) = PETSC_TRUE;
                        PetscCall(initialize_verification(ACTUAL_PARAMS));
                        // Broadcast a verification message to all its neighbors
                        (*state) = VERIFICATION;
                    }
                    else
                    {
                        if ((*NbNotRecvd) == 1)
                        {
                            // Send a PartialCV message to the neighbor corresponding to the unique cell of RecvdPCV[] being false
                            (*PartialCVSent) = PETSC_TRUE;
                            (*state) = WAIT4VERIFICATION;
                        }
                    }
                }
                else
                {
                    if (1 /*if all the cells of NewerDep[] are true then*/)
                    {
                        (*PseudoPeriodEnd) = PETSC_TRUE;
                    }
                }
            }
        }
    }
    else if ((*state) == WAIT4VERIFICATION)
    {
        if (UnderThreashold == PETSC_FALSE)
        {
            (*LocalCV) = PETSC_FALSE;
        }
    }
    else if ((*state) == VERIFICATION)
    {
        if ((*ElectedNode) == PETSC_TRUE)
        {
            if (UnderThreashold == PETSC_FALSE || (*LocalCV) == PETSC_FALSE /*or at least one cell of Resps[] is negative then*/)
            {
                (*PhaseTag) = (*PhaseTag) + 1;
                // Broadcast a negative verdict message to all its neighbors
                PetscCall(initialize_state(ACTUAL_PARAMS));
            }
            else
            {
                if ((*PseudoPeriodEnd) == PETSC_TRUE)
                {
                    if (1 /*there are no more 0 in Resps[]*/)
                    {
                        if (1 /*all the cells of Resps[] are positive*/)
                        {
                            // Broadcast a positive verdict message to all its neighbors
                            (*state) = FINISHED;
                        }
                        else
                        {
                            (*PhaseTag) = (*PhaseTag) + 1;
                            // Broadcast a negative verdict message to all its neighbors
                            PetscCall(initialize_state(ACTUAL_PARAMS));
                        }
                    }
                }
                else
                {
                    if (1 /*all the cells of NewerDep[] are true*/)
                    {
                        (*PseudoPeriodEnd) = PETSC_TRUE;
                    }
                }
            }
        }
        else
        {
            if ((*ResponseSent) == PETSC_FALSE)
            {
                if (UnderThreashold == PETSC_FALSE || (*LocalCV) == PETSC_FALSE /*at least one cell of Resps[] is negative*/)
                {
                    // Send a negative response to the asking neighbor
                    // by construction, that is the neighbor to which has been sent
                    // the last PartialCV message ⇔ false cell of RecvdPCV[]
                    (*ResponseSent) = PETSC_TRUE;
                }
                else
                {
                    if ((*PseudoPeriodEnd) == PETSC_TRUE)
                    {
                        if (1 /*there remains only one 0 in Resps[]*/)
                        {
                            // that last 0 is located in the cell of the asking neighbor
                            if (1 /*the other cells of Resps[] are all positive*/)
                            {
                                // Send a positive response to the asking neighbor
                            }
                            else
                            {
                                // Send a negative response to the asking neighbor
                            }
                            (*ResponseSent) = PETSC_TRUE;
                        }
                    }
                    else
                    {
                        if (1 /*all the cells of NewerDep[] are true*/)
                        {
                            (*PseudoPeriodEnd) = PETSC_TRUE;
                        }
                    }
                }
            }
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode initialize_state(PARAMS)
{
    PetscFunctionBeginUser;

    (*NbNotRecvd) = NbNeighbors;
    for (PetscInt idx = 0; idx < NbNeighbors; idx++)
    {
        ReceivedPartialCV[idx] = PETSC_FALSE;
    }
    (*ElectedNode) = PETSC_FALSE;
    (*LocalCV) = PETSC_FALSE;
    (*PartialCVSent) = PETSC_FALSE;
    PetscCall(reinitialize_pseudo_period(ACTUAL_PARAMS));
    (*state) = NORMAL;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode reinitialize_pseudo_period(PARAMS)
{
    PetscFunctionBeginUser;
    (*PseudoPeriodBegin) = PETSC_FALSE;
    (*PseudoPeriodEnd) = PETSC_FALSE;
    for (PetscInt idx = 0; idx < NbDependencies; idx++)
    {
        NewerDependencies[idx] = PETSC_FALSE;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode initialize_verification(PARAMS)
{
    PetscFunctionBeginUser;
    PetscCall(reinitialize_pseudo_period(ACTUAL_PARAMS));
    (*PhaseTag) = (*PhaseTag) + 1;
    for (PetscInt idx = 0; idx < NbNeighbors; idx++)
    {
        Responses[idx] = 0;
    }
    (*ResponseSent) = PETSC_FALSE;

    PetscFunctionReturn(PETSC_SUCCESS);
}

// Receive one information
PetscErrorCode receive_verification(PARAMS)
{
    PetscFunctionBeginUser;

    PetscMPIInt flag = 0;
    MPI_Status status;
    PetscInt SrcTag = 0;
    PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_RCV_VERIFICATION, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
    if (flag)
    {

        PetscCallMPI(MPI_Recv(&SrcTag, 1, MPIU_INT, MPI_ANY_SOURCE, TAG_SEND_RCV_VERIFICATION, MPI_COMM_WORLD, &status));
        // PetscInt SrcNode = status.MPI_SOURCE;

        if (SrcTag == ((*PhaseTag) + 1))
        {
            PetscCall(initialize_verification(ACTUAL_PARAMS));
            (*state) = VERIFICATION;
            // TODO: Broadcast the verification message to all its neighbors but SrcNode
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// Receive two informations
PetscErrorCode receive_response(PARAMS)
{
    PetscFunctionBeginUser;
    // Extract SrcNode, SrcTag and SrcResp from the message
    // PetscInt SrcNode = 0;
    PetscInt SrcTag = 0;
    PetscInt SrcResponse = 0;
    PetscInt SrcIndexNeighbor = 0; /*corresponding index of SrcNode in the list of neighbors of the current node (−1 if not in the list)*/
    if (SrcIndexNeighbor >= 0 && (*PhaseTag) == SrcTag)
    {
        Responses[SrcIndexNeighbor] = SrcResponse;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// Receive two informations
PetscErrorCode receive_verdict(PARAMS)
{
    PetscFunctionBeginUser;
    // Extract SrcNode, SrcTag and SrcVerdict from the message
    // PetscInt SrcNode = 0;
    PetscInt SrcTag = 0;
    // PetscInt SrcVerdict = 0;
    if (1 /*SrcVerdict is positive*/)
    {
        (*state) = FINISHED;
    }
    else
    {
        PetscCall(initialize_state(ACTUAL_PARAMS));
        (*PhaseTag) = SrcTag;
    }

    // Broadcast the verdict message to all its neighbors but SrcNode

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode receive_data_dependency(PARAMS, PetscInt *LastIteration)
{
    PetscFunctionBeginUser;

    // Extract SrcNode, SrcIter and SrcTag from the message
    // PetscInt SrcNode = 0;
    PetscInt SrcTag = 0;
    PetscInt SrcIteration = 0;
    PetscInt SrcIndexDependency = -1; // corresponding index of SrcNode in the list of dependencies of the current node (−1 if not in the list)

    if (SrcIndexDependency >= 0)
    {
        if (LastIteration[SrcIndexDependency] < SrcIteration && ((*state) != VERIFICATION || SrcTag == (*PhaseTag)))
        {
            // Put the data in the message at their corresponding place according to SrcIndDep in the local data array used for the computations
            LastIteration[SrcIndexDependency] = SrcIteration;
            NewerDependencies[SrcIndexDependency] = PETSC_TRUE;
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode receive_partial_CV(PARAMS, PetscInt proc_global_rank)
{
    PetscFunctionBeginUser;
    // TODO:Extract SrcNode and SrcTag from the message
    // TODO:SrcIndNeig = corresponding index of SrcNode in the list of neighbors of the current node (−1 if not in the list)
    PetscMPIInt flag = 0;
    MPI_Status status;
    PetscInt SrcNode = 0;
    PetscInt SrcTag = 0;
    PetscInt SrcIndexNeighbor = 0;

    PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_RCV_PARTIAL_CV, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
    if (flag)
    {
        PetscCallMPI(MPI_Recv(&SrcTag, 1, MPIU_INT, MPI_ANY_SOURCE, TAG_SEND_RCV_PARTIAL_CV, MPI_COMM_WORLD, &status));
        SrcNode = status.MPI_SOURCE;

        if (SrcIndexNeighbor >= 0 && SrcTag == (*PhaseTag))
        {
            ReceivedPartialCV[SrcIndexNeighbor] = PETSC_TRUE;
            (*NbNotRecvd) = (*NbNotRecvd) - 1;
            PetscInt leader = SrcNode;
            PetscCall(choose_leader(proc_global_rank, SrcNode, &leader));
            if ((*NbNotRecvd) == 0 && (*PartialCVSent) == PETSC_TRUE && leader == proc_global_rank)
            {
                (*ElectedNode) = PETSC_TRUE;
                PetscCall(initialize_verification(ACTUAL_PARAMS));
                // TODO: Broadcast a verification message to all its neighbors
                (*state) = VERIFICATION;
            }
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode choose_leader(PetscInt CurrentNode, PetscInt SrcNode, PetscInt *leader)
{
    PetscFunctionBeginUser;
    // TODO: choose the leader
    (*leader) = CurrentNode;
    PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscErrorCode pack_data(PetscScalar *send_buffer, PetscMPIInt data_size, PetscInt *version, char **pack_buffer, PetscMPIInt *position)
// {

//     PetscFunctionBeginUser;

//     PetscFunctionReturn(PETSC_SUCCESS);
// }

PetscErrorCode pack_convergence_data(PetscInt *first_place, PetscInt *second_place, char **pack_buffer, PetscMPIInt *position)
{
    PetscFunctionBeginUser;

    PetscMPIInt first_place_size;
    PetscMPIInt second_place_size;
    (*position) = 0;

    PetscCallMPI(MPI_Pack_size(1, MPIU_INT, MPI_COMM_WORLD, &first_place_size));
    PetscCallMPI(MPI_Pack_size(1, MPIU_INT, MPI_COMM_WORLD, &second_place_size));

    PetscMPIInt total_pack_size;
    total_pack_size = first_place_size + second_place_size;
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "size: %d\n", total_pack_size));

    if ((*pack_buffer) == NULL)
    {
        PetscCall(PetscMalloc1(total_pack_size, &(*pack_buffer)));
    }

    PetscCallMPI(MPI_Pack(first_place, 1, MPIU_INT, (*pack_buffer), total_pack_size, position, MPI_COMM_WORLD));
    PetscCallMPI(MPI_Pack(second_place, 1, MPIU_INT, (*pack_buffer), total_pack_size, position, MPI_COMM_WORLD));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode unpack_convergence_data(PetscInt *first_place, PetscInt *second_place, char **pack_buffer, PetscMPIInt pack_size)
{

    PetscFunctionBeginUser;

    PetscMPIInt position = 0;
    PetscCallMPI(MPI_Unpack((*pack_buffer), pack_size, &position, first_place, 1, MPIU_INT, MPI_COMM_WORLD));
    PetscCallMPI(MPI_Unpack((*pack_buffer), pack_size, &position, second_place, 1, MPIU_INT, MPI_COMM_WORLD));

    PetscFunctionReturn(PETSC_SUCCESS);
}

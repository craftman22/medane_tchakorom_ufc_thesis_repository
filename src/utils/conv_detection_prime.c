#include "constants.h"
#include "utils.h"
#include "conv_detection_prime.h"

/*
    Algorithm 5.15 (Practical version of convergence detection)
    from the book : Parallel Iterative Algorithms: from Sequential to Grid Computing

*/

// TODO: il n'ya pas moyen de passer de wait4verification à l'etat NORMAL ???? : peut etre le verdict

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
                        PetscInt proc;
                        PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc));
                        PetscCall(PetscPrintf(MPI_COMM_SELF, "LE LEADER EST (convDetection): %d \n", proc));

                        (*ElectedNode) = PETSC_TRUE;
                        PetscCall(initialize_verification(ACTUAL_PARAMS));
                        // XXX: Broadcast a verification message to all its neighbors
                        send_verification_buffer[0] = (*PhaseTag);
                        for (PetscInt idx = 0; idx < NbNeighbors; idx++)
                        {
                            PetscCallMPI(MPI_Isend(send_verification_buffer, 1, MPIU_INT, neighbors[idx], TAG_SEND_RCV_VERIFICATION, MPI_COMM_WORLD, send_verification_request));
                        }
                        (*state) = VERIFICATION;
                    }
                    else
                    {
                        if ((*NbNotRecvd) == 1)
                        {
                            // XXX: Send a PartialCV message to the neighbor corresponding to the unique cell of RecvdPCV[] being false
                            send_partialCV_buffer[0] = (*PhaseTag);
                            for (PetscInt idx = 0; idx < NbNeighbors; idx++)
                            {
                                if (ReceivedPartialCV[idx] == PETSC_FALSE)
                                {
                                    PetscCallMPI(MPI_Isend(send_partialCV_buffer, 1, MPIU_INT, neighbors[idx], TAG_SEND_RCV_PARTIAL_CV, MPI_COMM_WORLD, send_CV_request));
                                    break;
                                }
                            }

                            (*PartialCVSent) = PETSC_TRUE;
                            (*state) = WAIT4VERIFICATION;
                        }
                    }
                }
                else
                {
                    /*if all the cells of NewerDep[] are true then*/
                    PetscBool all_cells_true = PETSC_FALSE;
                    PetscCall(VecEqual(NewerDependencies_local, UNITARY_VECTOR, &all_cells_true));
                    // PetscCall(VecView(NewerDependencies_local, PETSC_VIEWER_STDOUT_SELF));
                    /* if new data from all dependency arrived*/
                    if (all_cells_true == PETSC_TRUE)
                    {
                        // PetscCall(PetscPrintf(MPI_COMM_SELF,"they are equal\n"));
                        // PetscCall(PetscPrintf(MPI_COMM_SELF, "pseudo period end!\n"));
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
            // XXX: passe ici
            // PetscCall(PetscSleep(100));
            // or at least one cell of Resps[] is negative
            PetscBool found_negative_response = PETSC_FALSE;
            PetscCall(find_value_in_responses_array(ACTUAL_PARAMS, RESPONSE_NEGATIVE, &found_negative_response, NULL));
            if (UnderThreashold == PETSC_FALSE || (*LocalCV) == PETSC_FALSE || found_negative_response == PETSC_TRUE)
            {

                // XXX: ne passe pas ici
                (*PhaseTag) = (*PhaseTag) + 1;
                // Broadcast a negative verdict message to all its neighbors
                send_verdict_buffer[0] = (*PhaseTag);
                send_verdict_buffer[1] = VERDICT_NEGATIVE; // negative verdict
                for (PetscInt idx = 0; idx < NbNeighbors; idx++)
                {
                    PetscCallMPI(MPI_Isend(send_verdict_buffer, 2, MPIU_INT, neighbors[idx], TAG_SEND_RCV_VERDICT, MPI_COMM_WORLD, send_verdict_request));
                }
                PetscCall(initialize_state(ACTUAL_PARAMS));
            }
            else
            {
                // XXX: passe par ici

                if ((*PseudoPeriodEnd) == PETSC_TRUE)
                {

                    // XXX: passe par ici
                    /*there are no more 0 in Resps[]*/
                    PetscBool found_neutral_response = PETSC_TRUE;
                    PetscCall(find_value_in_responses_array(ACTUAL_PARAMS, RESPONSE_NEUTRAL, &found_neutral_response, NULL));

                    if (found_neutral_response == PETSC_FALSE)
                    {
                        /*all the cells of Resps[] are positive*/
                        PetscBool found_negative_response = PETSC_TRUE;
                        PetscCall(find_value_in_responses_array(ACTUAL_PARAMS, RESPONSE_NEGATIVE, &found_negative_response, NULL));
                        if ((found_neutral_response == PETSC_FALSE) && (found_negative_response == PETSC_FALSE))
                        {

                            // XXX: Broadcast a positive verdict message to all its neighbors
                            send_verdict_buffer[0] = (*PhaseTag);
                            send_verdict_buffer[1] = VERDICT_POSITIVE; // positive verdict
                            for (PetscInt idx = 0; idx < NbNeighbors; idx++)
                            {
                                PetscCallMPI(MPI_Isend(send_verdict_buffer, 2, MPIU_INT, neighbors[idx], TAG_SEND_RCV_VERDICT, MPI_COMM_WORLD, send_verdict_request));
                            }
                            (*state) = FINISHED;
                        }
                        else
                        {

                            (*PhaseTag) = (*PhaseTag) + 1;
                            // Broadcast a negative verdict message to all its neighbors
                            send_verdict_buffer[0] = (*PhaseTag);
                            send_verdict_buffer[1] = VERDICT_NEGATIVE; // negative verdict
                            for (PetscInt idx = 0; idx < NbNeighbors; idx++)
                            {
                                PetscCallMPI(MPI_Isend(send_verdict_buffer, 2, MPIU_INT, neighbors[idx], TAG_SEND_RCV_VERDICT, MPI_COMM_WORLD, send_verdict_request));
                            }
                            PetscCall(initialize_state(ACTUAL_PARAMS));
                        }
                    }
                }
                else
                {
                    /*all the cells of NewerDep[] are true*/
                    PetscBool all_cells_true = PETSC_FALSE;
                    PetscCall(VecEqual(NewerDependencies_local, UNITARY_VECTOR, &all_cells_true));
                    // PetscCall(VecView(NewerDependencies_local, PETSC_VIEWER_STDOUT_SELF));
                    if (all_cells_true == PETSC_TRUE)
                    {
                        (*PseudoPeriodEnd) = PETSC_TRUE;
                    }
                }
            }
        }
        else
        {
            // XXX: ne passe pas par ici
            // PetscInt proc;
            // PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc));

            // PetscCall(PetscPrintf(MPI_COMM_SELF, "this is rank %d \n", proc));
            // PetscCall(PetscSleep(100));

            if ((*ResponseSent) == PETSC_FALSE)
            {
                PetscBool found_negative_response = PETSC_FALSE;
                PetscCall(find_value_in_responses_array(ACTUAL_PARAMS, RESPONSE_NEGATIVE, &found_negative_response, NULL));
                /* or at least one cell of Resps[] is negative*/
                if (UnderThreashold == PETSC_FALSE || (*LocalCV) == PETSC_FALSE || found_negative_response == PETSC_TRUE)
                {
                    // Send a negative response to the asking neighbor
                    // by construction, that is the neighbor to which has been sent
                    // the last PartialCV message ⇔ false cell of RecvdPCV[]
                    send_response_buffer[0] = (*PhaseTag);
                    send_response_buffer[1] = RESPONSE_NEGATIVE;
                    for (PetscInt idx = 0; idx < NbNeighbors; idx++)
                    {
                        if (ReceivedPartialCV[idx] == PETSC_FALSE)
                        {
                            PetscCallMPI(MPI_Isend(send_response_buffer, 2, MPIU_INT, neighbors[idx], TAG_SEND_RCV_RESPONSE, MPI_COMM_WORLD, send_response_request));
                            break;
                        }
                    }
                    (*ResponseSent) = PETSC_TRUE;
                }
                else
                {
                    if ((*PseudoPeriodEnd) == PETSC_TRUE)
                    {

                        /*there remains only one 0 in Resps[]*/
                        PetscInt nb_neutral_responses_occurences = 0;
                        PetscCall(find_value_in_responses_array(ACTUAL_PARAMS, RESPONSE_NEUTRAL, NULL, &nb_neutral_responses_occurences));
                        if (nb_neutral_responses_occurences == 1)
                        {
                            // that last 0 is located in the cell of the asking neighbor
                            PetscInt AskingNeigbor = -1;
                            for (PetscInt idx = 0; idx < NbNeighbors; idx++)
                            {
                                if (Responses[idx] == RESPONSE_NEUTRAL)
                                {
                                    AskingNeigbor = neighbors[idx];
                                    break;
                                }
                            }

                            PetscInt nb_positive_responses_occurences = 0;
                            PetscCall(find_value_in_responses_array(ACTUAL_PARAMS, RESPONSE_POSITIVE, NULL, &nb_positive_responses_occurences));
                            /*the other cells of Resps[] are all positive*/
                            if (nb_positive_responses_occurences == (NbNeighbors - 1))
                            {
                                // Send a positive response to the asking neighbor
                                send_response_buffer[0] = (*PhaseTag);
                                send_response_buffer[1] = RESPONSE_POSITIVE;
                                PetscCallMPI(MPI_Isend(send_response_buffer, 2, MPIU_INT, AskingNeigbor, TAG_SEND_RCV_RESPONSE, MPI_COMM_WORLD, send_response_request));
                            }
                            else
                            {
                                // Send a negative response to the asking neighbor
                                send_response_buffer[0] = (*PhaseTag);
                                send_response_buffer[1] = RESPONSE_NEGATIVE;
                                PetscCallMPI(MPI_Isend(send_response_buffer, 2, MPIU_INT, AskingNeigbor, TAG_SEND_RCV_RESPONSE, MPI_COMM_WORLD, send_response_request));
                            }
                            (*ResponseSent) = PETSC_TRUE;
                        }
                    }
                    else
                    {
                        PetscBool all_cells_true = PETSC_FALSE;
                        PetscCall(VecEqual(NewerDependencies_local, UNITARY_VECTOR, &all_cells_true));
                        /*if all the cells of NewerDep[] are true*/
                        if (all_cells_true == PETSC_TRUE)
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

/*
 * @brief Re-initialize pseudo period by setting.
 *
 * PseudoPeriodBeing -> False
 * PseudoPeriodEnd -> False
 * NewerDependencies -> False
 */
PetscErrorCode reinitialize_pseudo_period(PARAMS)
{
    PetscFunctionBeginUser;
    (*PseudoPeriodBegin) = PETSC_FALSE;
    (*PseudoPeriodEnd) = PETSC_FALSE;
    for (PetscInt idx = 0; idx < NbDependencies; idx++)
    {
        PetscCall(VecSet(NewerDependencies_local, PETSC_FALSE));
        PetscCall(VecSetValueLocal(NewerDependencies_global, 0, PETSC_FALSE, INSERT_VALUES));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * @brief initialize verification by.
 *
 * reinitalizing pseudo-period
 *
 * Increasing PhaseTag by value 1: PhaseTag = PhaseTag + 1
 *
 * Initializing all responses from neighbors to 0: neutral response
 *
 * Setting ResponseSent to FALSE
 */
PetscErrorCode initialize_verification(PARAMS)
{
    PetscFunctionBeginUser;
    PetscCall(reinitialize_pseudo_period(ACTUAL_PARAMS));
    (*PhaseTag) = (*PhaseTag) + 1;
    for (PetscInt idx = 0; idx < NbNeighbors; idx++)
    {
        Responses[idx] = RESPONSE_NEUTRAL;
    }
    (*ResponseSent) = PETSC_FALSE;

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
    PetscInt SrcIndexNeighbor = -1;

    PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_RCV_PARTIAL_CV, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
    if (flag)
    {
        // do
        // {
        PetscCallMPI(MPI_Recv(rcv_partialCV_buffer, 1, MPIU_INT, MPI_ANY_SOURCE, TAG_SEND_RCV_PARTIAL_CV, MPI_COMM_WORLD, &status));
        //     PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_RCV_PARTIAL_CV, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
        // } while (flag);

        SrcNode = status.MPI_SOURCE;
        SrcTag = rcv_partialCV_buffer[0];
        SrcIndexNeighbor = -1;
        for (PetscInt idx = 0; idx < NbNeighbors; idx++)
        {
            if (neighbors[idx] == SrcNode)
            {
                SrcIndexNeighbor = idx;
                break;
            }
        }

        if (SrcIndexNeighbor >= 0 && SrcTag == (*PhaseTag))
        {
            ReceivedPartialCV[SrcIndexNeighbor] = PETSC_TRUE;
            (*NbNotRecvd) = (*NbNotRecvd) - 1;
            PetscInt leader = SrcNode;
            PetscCall(choose_leader(proc_global_rank, SrcNode, &leader));
            if ((*NbNotRecvd) == 0 && (*PartialCVSent) == PETSC_TRUE && leader == proc_global_rank)
            {
                PetscCall(PetscPrintf(MPI_COMM_SELF, "LE LEADER EST (function rcv partial cv): %d \n", proc_global_rank));
                (*ElectedNode) = PETSC_TRUE;
                PetscCall(initialize_verification(ACTUAL_PARAMS));
                // TODO: Broadcast a verification message to all its neighbors
                send_verification_buffer[0] = (*PhaseTag);
                for (PetscInt idx = 0; idx < NbNeighbors; idx++)
                {
                    PetscCallMPI(MPI_Isend(send_verification_buffer, 1, MPIU_INT, neighbors[idx], TAG_SEND_RCV_VERIFICATION, MPI_COMM_WORLD, send_verification_request));
                }

                (*state) = VERIFICATION;
            }
        }
    }

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
        // do
        // {
        // TODO: changer le buffer si necessaire
        PetscCallMPI(MPI_Recv(rcv_verification_buffer, 1, MPIU_INT, MPI_ANY_SOURCE, TAG_SEND_RCV_VERIFICATION, MPI_COMM_WORLD, &status));
        //     PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_RCV_VERIFICATION, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
        // } while (flag);

        PetscInt SrcNode = status.MPI_SOURCE;
        SrcTag = rcv_verification_buffer[0];

        if (SrcTag == ((*PhaseTag) + 1))
        {
            PetscCall(initialize_verification(ACTUAL_PARAMS));
            (*state) = VERIFICATION;
            // TODO: Broadcast the verification message to all its neighbors but SrcNode
            for (PetscInt idx = 0; idx < NbNeighbors; idx++)
            {
                if (neighbors[idx] != SrcNode)
                {
                    // TODO: peut etre doit on augmenter la valeur de phaseTag avant de l'envoyer ?????
                    send_verification_buffer[0] = (*PhaseTag);
                    PetscCallMPI(MPI_Isend(send_verification_buffer, 1, MPIU_INT, neighbors[idx], TAG_SEND_RCV_VERIFICATION, MPI_COMM_WORLD, send_verification_request));
                }
            }
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// Receive two informations
PetscErrorCode receive_response(PARAMS)
{
    PetscFunctionBeginUser;
    PetscMPIInt flag = 0;
    PetscInt SrcNode = 0;
    PetscInt SrcTag = 0;
    PetscInt SrcResponse = 0;
    PetscInt SrcIndexNeighbor = 0; /*corresponding index of SrcNode in the list of neighbors of the current node (−1 if not in the list)*/
    MPI_Status status;
    // Extract SrcNode, SrcTag and SrcResp from the message

    PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_RCV_RESPONSE, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));

    if (flag)
    {
        // do
        // {
        PetscCallMPI(MPI_Recv(rcv_response_buffer, 2, MPIU_INT, MPI_ANY_SOURCE, TAG_SEND_RCV_RESPONSE, MPI_COMM_WORLD, &status));
        // PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_RCV_RESPONSE, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));

        // } while (flag);

        SrcNode = status.MPI_SOURCE;
        SrcTag = rcv_response_buffer[0];
        SrcResponse = rcv_response_buffer[1];
        PetscCall(get_neighbor_node_index(ACTUAL_PARAMS, SrcNode, &SrcIndexNeighbor));

        if (SrcIndexNeighbor >= 0 && (*PhaseTag) == SrcTag)
        {
            Responses[SrcIndexNeighbor] = SrcResponse;
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// Receive two informations
PetscErrorCode receive_verdict(PARAMS)
{
    PetscFunctionBeginUser;
    // Extract SrcNode, SrcTag and SrcVerdict from the message
    PetscInt SrcNode = 0;
    PetscInt SrcTag = 0;
    PetscInt SrcVerdict = 0;
    PetscMPIInt flag = 0;
    MPI_Status status;

    PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_RCV_VERDICT, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
    if (flag)
    {
        // XXX: ne passe pas ici
        // PetscCall(PetscSleep(1000));
        // do
        // {
        PetscCallMPI(MPI_Recv(rcv_verdict_buffer, 2, MPIU_INT, MPI_ANY_SOURCE, TAG_SEND_RCV_VERDICT, MPI_COMM_WORLD, &status));
        //     PetscCall(MPI_Iprobe(MPI_ANY_SOURCE, TAG_SEND_RCV_VERDICT, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
        // } while (flag);

        SrcNode = status.MPI_SOURCE;
        SrcTag = rcv_verdict_buffer[0];
        SrcVerdict = rcv_verdict_buffer[1];

        if (SrcVerdict == VERDICT_POSITIVE)
        {
            (*state) = FINISHED;
        }
        else
        {
            PetscCall(initialize_state(ACTUAL_PARAMS));
            (*PhaseTag) = SrcTag;
        }
        // Broadcast the verdict message to all its neighbors but SrcNode
        send_verdict_buffer[0] = (*PhaseTag);
        send_verdict_buffer[1] = rcv_verdict_buffer[1]; // TODO: which verdict message ???? think about initialization
        for (PetscInt idx = 0; idx < NbNeighbors; idx++)
        {
            if (neighbors[idx] != SrcNode)
            {
                PetscCallMPI(MPI_Isend(send_verdict_buffer, 2, MPIU_INT, neighbors[idx], TAG_SEND_RCV_VERDICT, MPI_COMM_WORLD, send_verdict_request));
            }
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode choose_leader(PetscInt CurrentNode, PetscInt SrcNode, PetscInt *leader)
{
    PetscFunctionBeginUser;
    // TODO: choose the leader

    (*leader) = PetscMax(CurrentNode, SrcNode);
    // (*leader) = SrcNode;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode get_neighbor_node_index(PARAMS, const PetscInt node_rank, PetscInt *idx_found)
{
    PetscFunctionBeginUser;
    (*idx_found) = -1;
    for (PetscInt idx = 0; idx < NbNeighbors; idx++)
    {
        if (neighbors[idx] == node_rank)
        {
            (*idx_found) = idx;
            break;
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode find_value_in_responses_array(PARAMS, PetscInt value_to_search, PetscBool *found, PetscInt *nb_occurences_found)
{
    PetscFunctionBeginUser;
    PetscBool _found = PETSC_FALSE;
    PetscInt _nb_occurences_found = 0;
    for (PetscInt idx = 0; idx < NbNeighbors; idx++)
    {
        if (Responses[idx] == value_to_search)
        {
            _found = PETSC_TRUE;
            _nb_occurences_found = _nb_occurences_found + 1;
        }
    }

    if (found != NULL)
    {
        (*found) = _found;
    }

    if (nb_occurences_found != NULL)
    {
        (*nb_occurences_found) = _nb_occurences_found;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode pack_computed_data_dependency(PetscInt *PhaseTag_param, PetscInt *NumIteration_param, const PetscInt data_count, PetscScalar *data, char **pack_buffer, PetscMPIInt *position)
{
    PetscFunctionBeginUser;
    PetscMPIInt size_1 = 0;
    PetscMPIInt size_2 = 0;
    PetscMPIInt size_3 = 0;
    (*position) = 0;

    PetscCallMPI(MPI_Pack_size(1, MPIU_INT, MPI_COMM_WORLD, &size_1));
    PetscCallMPI(MPI_Pack_size(1, MPIU_INT, MPI_COMM_WORLD, &size_2));
    PetscCallMPI(MPI_Pack_size(data_count, MPIU_SCALAR, MPI_COMM_WORLD, &size_3));

    PetscMPIInt total_pack_size;
    total_pack_size = size_1 + size_2 + size_3;
    // PetscCall(PetscPrintf(PETSC_COMM_SELF, "size: %d\n", total_pack_size));

    if ((*pack_buffer) == NULL)
    {
        PetscCall(PetscMalloc1(total_pack_size, &(*pack_buffer)));
    }

    PetscCallMPI(MPI_Pack(PhaseTag_param, 1, MPIU_INT, (*pack_buffer), total_pack_size, position, MPI_COMM_WORLD));
    PetscCallMPI(MPI_Pack(NumIteration_param, 1, MPIU_INT, (*pack_buffer), total_pack_size, position, MPI_COMM_WORLD));
    PetscCallMPI(MPI_Pack(data, data_count, MPIU_SCALAR, (*pack_buffer), total_pack_size, position, MPI_COMM_WORLD));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode unpack_computed_data_dependency(PetscInt *SrcPhaseTag, PetscInt *SrcCurrentIteration, const PetscInt data_count, PetscScalar *data, char **pack_buffer, PetscMPIInt pack_size)
{

    PetscFunctionBeginUser;

    PetscMPIInt position = 0;
    PetscCallMPI(MPI_Unpack((*pack_buffer), pack_size, &position, SrcPhaseTag, 1, MPIU_INT, MPI_COMM_WORLD));
    PetscCallMPI(MPI_Unpack((*pack_buffer), pack_size, &position, SrcCurrentIteration, 1, MPIU_INT, MPI_COMM_WORLD));
    PetscCallMPI(MPI_Unpack((*pack_buffer), pack_size, &position, data, data_count, MPIU_SCALAR, MPI_COMM_WORLD));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscErrorCode pack_data(PetscScalar *send_buffer, PetscMPIInt data_size, PetscInt *version, char **pack_buffer, PetscMPIInt *position)
// {

//     PetscFunctionBeginUser;

//     PetscFunctionReturn(PETSC_SUCCESS);
// }

// called by all rank in the sub_communicator
PetscErrorCode receive_data_dependency(Vec NewerDependencies_global, Vec LastIteration_global, const State state, const PetscInt PhaseTag, const PetscInt SrcPhaseTag, const PetscInt SrcCurrentIteration)
{
    PetscFunctionBeginUser;

    // Extract SrcNode, SrcIter and SrcTag from the message
    // PetscInt SrcNode = SrcNode_param;
    PetscInt SrcTag = SrcPhaseTag;
    PetscInt SrcIteration = SrcCurrentIteration;
    // corresponding index of SrcNode in the list of dependencies of the current node (−1 if not in the list)
    PetscInt SrcIndexDependency = 1;

    if (SrcIndexDependency >= 0)
    {
        PetscScalar LastIterationRegistered = SrcIteration;
        PetscInt idx = 0;
        PetscCall(VecGetValues(LastIteration_global, 1, &idx, &LastIterationRegistered));

        if (LastIterationRegistered < SrcIteration && ((state) != VERIFICATION || SrcTag == (PhaseTag)))
        {
            // Put the data in the message at their corresponding place according to SrcIndDep in the local data array used for the computations
            // LastIteration[SrcIndexDependency] = SrcIteration;
            // NewerDependencies[SrcIndexDependency] = PETSC_TRUE;
            PetscCall(VecSetValueLocal(LastIteration_global, 0, SrcIteration, INSERT_VALUES));
            PetscCall(VecSetValueLocal(NewerDependencies_global, 0, PETSC_TRUE, INSERT_VALUES));
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}
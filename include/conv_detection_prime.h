
/*
    Algorithm 5.15 (Practical version of convergence detection)
    from the book : Parallel Iterative Algorithms: from Sequential to Grid Computing

*/

#ifndef SHARED_CONV_DETECTION_PRIME_FUNCTIONS_H
#define SHARED_CONV_DETECTION_PRIME_FUNCTIONS_H

#define PARAMS                                                                                                         \
    const PetscInt proc_local_rank, PetscInt NbNeighbors, PetscBool *UnderThreashold,                                  \
        PetscInt NbDependencies, PetscInt *Responses,                                                                  \
        Vec NewerDependencies_global, PetscBool *PseudoPeriodBegin,                                                    \
        Vec LastIteration_global, Vec LastIteration_local,                                                             \
        Vec NewerDependencies_local, PetscBool *PseudoPeriodEnd, PetscBool *ReceivedPartialCV,                         \
        PetscBool *ElectedNode, PetscInt *PhaseTag, PetscBool *ResponseSent, State *state,                             \
        PetscBool *LocalCV, PetscInt *NbNotRecvd, PetscBool *PartialCVSent,                                            \
        PetscInt *neighbors, Vec UNITARY_VECTOR, Vec NULL_VECTOR,                                                      \
        PetscInt *send_verdict_buffer, PetscInt *rcv_verdict_buffer,                                                   \
        PetscInt *send_response_buffer, PetscInt *rcv_response_buffer,                                                 \
        PetscInt *send_verification_buffer, PetscInt *rcv_verification_buffer,                                         \
        PetscInt *send_partialCV_buffer, PetscInt *rcv_partialCV_buffer,                                               \
        MPI_Request *send_verdict_request, MPI_Request *send_response_request, MPI_Request *send_verification_request, \
        MPI_Request *send_CV_request

#define ACTUAL_PARAMS                                                           \
    proc_local_rank, NbNeighbors, UnderThreashold,                              \
        NbDependencies, Responses,                                              \
        NewerDependencies_global, PseudoPeriodBegin,                            \
        LastIteration_global, LastIteration_local,                              \
        NewerDependencies_local, PseudoPeriodEnd, ReceivedPartialCV,            \
        ElectedNode, PhaseTag, ResponseSent, state,                             \
        LocalCV, NbNotRecvd, PartialCVSent,                                     \
        neighbors, UNITARY_VECTOR, NULL_VECTOR,                                 \
        send_verdict_buffer, rcv_verdict_buffer,                                \
        send_response_buffer, rcv_response_buffer,                              \
        send_verification_buffer, rcv_verification_buffer,                      \
        send_partialCV_buffer, rcv_partialCV_buffer,                            \
        send_verdict_request, send_response_request, send_verification_request, \
        send_CV_request

// Use in main program
#define ACTUAL_PARAMS_POINTERS                                                     \
    proc_local_rank, NbNeighbors, &UnderThreashold,                                \
        NbDependencies, Responses,                                                 \
        NewerDependencies_global, &PseudoPeriodBegin,                              \
        LastIteration_global, LastIteration_local,                                 \
        NewerDependencies_local, &PseudoPeriodEnd, ReceivedPartialCV,              \
        &ElectedNode, &PhaseTag, &ResponseSent, &state,                            \
        &LocalCV, &NbNotRecvd, &PartialCVSent,                                     \
        neighbors, UNITARY_VECTOR, NULL_VECTOR,                                    \
        send_verdict_buffer, rcv_verdict_buffer,                                   \
        send_response_buffer, rcv_response_buffer,                                 \
        send_verification_buffer, rcv_verification_buffer,                         \
        send_partialCV_buffer, rcv_partialCV_buffer,                               \
        &send_verdict_request, &send_response_request, &send_verification_request, \
        &send_CV_request

#include "constants.h"
#include "utils.h"

PetscErrorCode comm_async_convDetection_prime(PARAMS);

PetscErrorCode initialize_state(PARAMS);

PetscErrorCode reinitialize_pseudo_period(PARAMS);

PetscErrorCode initialize_verification(PARAMS);

// PetscErrorCode receive_data_dependency(PARAMS);
PetscErrorCode receive_data_dependency(Vec NewerDependencies_global, const PetscInt proc_local_rank, Vec LastIteration_global, const State state, const PetscInt PhaseTag, const PetscInt SrcPhaseTag, const PetscInt SrcCurrentIteration);

PetscErrorCode receive_verification(PARAMS);

PetscErrorCode receive_response(PARAMS);

PetscErrorCode receive_verdict(PARAMS);

PetscErrorCode receive_partial_CV(PARAMS, PetscInt proc_global_rank);

PetscErrorCode choose_leader(PetscInt CurrentNode, PetscInt SrcNode, PetscInt *leader);

PetscErrorCode get_neighbor_node_index(PARAMS, const PetscInt node_rank, PetscInt *idx_found);

PetscErrorCode find_value_in_responses_array(PARAMS, PetscInt value_to_search, PetscBool *found, PetscInt *nb_occurences_found);

// PetscErrorCode pack_computed_data_dependency(PetscInt *first_place, PetscInt *second_place, char **pack_buffer, PetscMPIInt *position);
// PetscErrorCode unpack_computed_data_dependency(PetscInt *first_place, PetscInt *second_place, char **pack_buffer, PetscMPIInt pack_size);

PetscErrorCode pack_computed_data_dependency(PetscInt *PhaseTag_param, PetscInt *NumIteration_param, const PetscInt data_count, PetscScalar *data, char **pack_buffer, PetscMPIInt *position);
PetscErrorCode unpack_computed_data_dependency(PetscInt *PhaseTag_param, PetscInt *NumIteration_param, const PetscInt data_count, PetscScalar *data, char **pack_buffer, PetscMPIInt pack_size);

#endif // SHARED_CONV_DETECTION_PRIME_FUNCTIONS_H

/*
    Algorithm 5.15 (Practical version of convergence detection)
    from the book : Parallel Iterative Algorithms: from Sequential to Grid Computing

*/

#ifndef SHARED_CONV_DETECTION_PRIME_FUNCTIONS_H
#define SHARED_CONV_DETECTION_PRIME_FUNCTIONS_H

#define PARAMS                                                                             \
    PetscInt NbNeighbors, PetscBool *UnderThreashold,                                      \
        PetscInt NbDependencies, PetscInt *Responses,                                      \
        PetscBool *NewerDependencies, PetscBool *PseudoPeriodBegin,                        \
        PetscBool *PseudoPeriodEnd, PetscBool *ReceivedPartialCV,                          \
        PetscBool *ElectedNode, PetscInt *PhaseTag, PetscBool *ResponseSent, State *state, \
        PetscBool *LocalCV, PetscInt *NbNotRecvd, PetscBool *PartialCVSent

#define ACTUAL_PARAMS                               \
    NbNeighbors, UnderThreashold,                   \
        NbDependencies, Responses,                  \
        NewerDependencies, PseudoPeriodBegin,       \
        PseudoPeriodEnd, ReceivedPartialCV,         \
        ElectedNode, PhaseTag, ResponseSent, state, \
        LocalCV, NbNotRecvd, PartialCVSent

#include "constants.h"
#include "utils.h"

PetscErrorCode comm_async_convDetection_prime(PARAMS);

PetscErrorCode initialize_state(PARAMS);

PetscErrorCode reinitialize_pseudo_period(PARAMS);

PetscErrorCode initialize_verification(PARAMS);


PetscErrorCode receive_data_dependency(PARAMS, PetscInt *LastIteration);


PetscErrorCode receive_verification(PARAMS);

PetscErrorCode receive_response(PARAMS);

PetscErrorCode receive_verdict(PARAMS);

PetscErrorCode receive_partial_CV(PARAMS, PetscInt proc_global_rank);

PetscErrorCode choose_leader(PetscInt CurrentNode, PetscInt SrcNode, PetscInt *leader);

PetscErrorCode pack_convergence_data(PetscInt *first_place, PetscInt *second_place, char **pack_buffer, PetscMPIInt *position);

PetscErrorCode unpack_convergence_data(PetscInt *first_place, PetscInt *second_place, char **pack_buffer, PetscMPIInt pack_size);

#endif // SHARED_CONV_DETECTION_FUNCTIONS_H
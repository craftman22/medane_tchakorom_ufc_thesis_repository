#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"
#include "constants.h"
#include "utils.h"
#include "comm.h"
#include "conv_detection.h"
#include "conv_detection_prime.h"

int main(int argc, char **argv)
{

    Mat A_block_jacobi = NULL; // Operator matrix
    Vec x = NULL;              // approximation solution at iteration (k)
    Vec b = NULL;              // right hand side vector
    Vec u = NULL;

    PetscMPIInt nprocs;
    PetscMPIInt proc_global_rank;
    PetscInt n_mesh_lines = 4;
    PetscInt n_mesh_columns = 4;
    PetscInt njacobi_blocks;
    PetscMPIInt rank_jacobi_block;
    PetscInt proc_local_rank;
    PetscInt n_mesh_points;
    PetscInt jacobi_block_size;
    PetscInt s;
    PetscScalar relative_tolerance = 1e-5;
    PetscScalar absolute_tolerance = 1e-100;
    PetscMPIInt nprocs_per_jacobi_block = 1;

    Vec local_right_side_vector = NULL;
    Vec mat_mult_vec_result = NULL;

    PetscInt MIN_CONVERGENCE_COUNT = 3;
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

    // Getting applications arguments
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &n_mesh_lines, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n_mesh_columns, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-s", &s, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-min_convergence_count", &MIN_CONVERGENCE_COUNT, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-npb", &nprocs_per_jacobi_block, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &relative_tolerance, NULL));

    // PetscPrintf(PETSC_COMM_WORLD, " =====> Total number of processes: %d \n =====>s : %d\n =====>nprocessor_per_jacobi_block : %d \n ====> Grid lines: %d \n ====> Grid columns : %d ====> Relative tolerance : %f\n", nprocs, s, nprocs_per_jacobi_block, n_mesh_lines, n_mesh_columns, relative_tolerance);

    PetscCall(computeDimensionRelatedVariables(nprocs, nprocs_per_jacobi_block, proc_global_rank, n_mesh_lines, n_mesh_columns, &njacobi_blocks, &rank_jacobi_block, &proc_local_rank, &n_mesh_points, &jacobi_block_size));

    PetscAssert((n_mesh_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");

    // Creating the sub communicator for each jacobi block
    PetscSubcomm sub_comm_context = NULL;
    MPI_Comm dcomm;
    PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD, &dcomm, NULL));

    PetscCall(PetscSubcommCreate(dcomm, &sub_comm_context));
    PetscCall(PetscSubcommSetNumber(sub_comm_context, njacobi_blocks));
    PetscCall(PetscSubcommSetType(sub_comm_context, PETSC_SUBCOMM_CONTIGUOUS));
    MPI_Comm comm_jacobi_block = PetscSubcommChild(sub_comm_context);

    MPI_Comm comm_local_roots; // communicator only for local root procs
    int color = (proc_global_rank == 0 || proc_global_rank == nprocs_per_jacobi_block) ? 0 : MPI_UNDEFINED;
    PetscCallMPI(MPI_Comm_split(MPI_COMM_WORLD, color, 0, &comm_local_roots));

    KSP inner_ksp = NULL;
    PetscInt number_of_iterations = ZERO;
    PetscMPIInt idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;
    PetscMPIInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
    PetscMPIInt message_dest = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
    Mat A_block_jacobi_subMat[njacobi_blocks];
    IS is_cols_block_jacobi[njacobi_blocks];
    Vec b_block_jacobi[njacobi_blocks];
    Vec x_block_jacobi[njacobi_blocks];
    VecScatter scatter_jacobi_vec_part_to_merged_vec[njacobi_blocks];
    IS is_jacobi_vec_parts;
    IS is_merged_vec[njacobi_blocks];
    PetscMPIInt vec_local_size = ZERO;
    PetscScalar *send_buffer = NULL;
    PetscScalar *rcv_buffer = NULL;
    MPI_Request send_data_request = MPI_REQUEST_NULL;
    Vec local_iterates_difference;
    Vec x_block_jacobi_previous_iterate = NULL;
    // PetscScalar *temp_buffer = NULL;
    // PetscMPIInt send_data_flag = ZERO;
    // PetscMPIInt rcv_data_flag = ZERO;

    // PetscInt *LastIteration = NULL;
    // PetscBool *NewerDependencies = NULL;
    Vec LastIteration_global = NULL;
    Vec NewerDependencies_global = NULL;
    Vec LastIteration_local = NULL;
    Vec NewerDependencies_local = NULL;
    Vec UNITARY_VECTOR = NULL;
    Vec NULL_VECTOR = NULL;

    PetscInt NbNeighbors = 0;
    PetscBool UnderThreashold = PETSC_FALSE;
    PetscInt NbDependencies = 0;
    PetscInt *Responses = NULL;
    PetscBool *ReceivedPartialCV = NULL;
    PetscBool PseudoPeriodBegin = PETSC_FALSE;
    PetscBool PseudoPeriodEnd = PETSC_FALSE;
    PetscBool ElectedNode = PETSC_FALSE;
    PetscInt PhaseTag = 0;
    PetscBool ResponseSent = PETSC_FALSE;
    State state = NORMAL;
    PetscBool LocalCV = PETSC_FALSE;
    PetscInt NbNotRecvd = 0;
    PetscBool PartialCVSent = PETSC_FALSE;
    PetscInt *neighbors = NULL; /* array containing all the node neighbors */
    PetscInt *send_verdict_buffer = NULL;
    PetscInt *rcv_verdict_buffer = NULL;
    PetscInt *send_response_buffer = NULL;
    PetscInt *rcv_response_buffer = NULL;
    PetscInt *send_verification_buffer = NULL;
    PetscInt *rcv_verification_buffer = NULL;
    PetscInt *send_partialCV_buffer = NULL;
    PetscInt *rcv_partialCV_buffer = NULL;
    MPI_Request send_verdict_request = MPI_REQUEST_NULL;
    MPI_Request send_response_request = MPI_REQUEST_NULL;
    MPI_Request send_verification_request = MPI_REQUEST_NULL;
    MPI_Request send_CV_request = MPI_REQUEST_NULL;
    PetscInt dependency_received = 0;
    PetscInt neighbor_current_iteration = 0;
    char *send_pack_buffer = NULL;
    char *rcv_pack_buffer = NULL;

    NbDependencies = nprocs / 2;
    PetscCall(create_vector(comm_jacobi_block, &NewerDependencies_global, NbDependencies, VECMPI));
    PetscCall(VecSet(NewerDependencies_global, PETSC_FALSE)); // Meaning petsc_false
    PetscCall(create_vector(comm_jacobi_block, &LastIteration_global, NbDependencies, VECMPI));
    PetscCall(VecSet(LastIteration_global, -1));

    VecScatter scatter_ctx;
    PetscCall(VecScatterCreateToZero(NewerDependencies_global, &scatter_ctx, &NewerDependencies_local));
    PetscCall(VecSet(NewerDependencies_local, PETSC_FALSE)); // Meaning petsc_false

    if (proc_local_rank == 0)
    {

        NbNeighbors = 1;

        PetscCall(PetscMalloc1(NbNeighbors, &neighbors));
        PetscCall(PetscArrayfill_custom_int(neighbors, -11, NbNeighbors));

        PetscCall(PetscMalloc1(NbNeighbors, &Responses));
        PetscCall(PetscArrayfill_custom_int(Responses, RESPONSE_NEUTRAL, NbNeighbors));

        // PetscCall(create_vector(comm_jacobi_block, &NewerDependencies_local, NbDependencies, VECSEQ));
        // PetscCall(VecSet(NewerDependencies_local, PETSC_FALSE)); // Meaning petsc_false

        PetscCall(create_vector(PETSC_COMM_SELF, &UNITARY_VECTOR, NbDependencies, VECSEQ));
        PetscCall(VecSet(UNITARY_VECTOR, PETSC_TRUE)); // Meaning petsc_false

        PetscCall(create_vector(PETSC_COMM_SELF, &NULL_VECTOR, NbDependencies, VECSEQ));
        PetscCall(VecSet(NULL_VECTOR, PETSC_FALSE)); // Meaning petsc_false

        PetscCall(create_vector(PETSC_COMM_SELF, &LastIteration_local, NbDependencies, VECSEQ));
        PetscCall(VecSet(LastIteration_local, -1));

        PetscCall(PetscMalloc1(2, &send_verdict_buffer));
        PetscCall(PetscArrayfill_custom_int(send_verdict_buffer, 0, 2));

        PetscCall(PetscMalloc1(2, &rcv_verdict_buffer));
        PetscCall(PetscArrayfill_custom_int(rcv_verdict_buffer, 0, 2));

        PetscCall(PetscMalloc1(2, &send_response_buffer));
        PetscCall(PetscArrayfill_custom_int(send_response_buffer, 0, 2));

        PetscCall(PetscMalloc1(2, &rcv_response_buffer));
        PetscCall(PetscArrayfill_custom_int(rcv_response_buffer, 0, 2));

        PetscCall(PetscMalloc1(1, &send_verification_buffer));
        PetscCall(PetscArrayfill_custom_int(send_verification_buffer, 0, 1));

        PetscCall(PetscMalloc1(1, &rcv_verification_buffer));
        PetscCall(PetscArrayfill_custom_int(rcv_verification_buffer, 0, 1));

        PetscCall(PetscMalloc1(1, &send_partialCV_buffer));
        PetscCall(PetscArrayfill_custom_int(send_partialCV_buffer, 0, 1));

        PetscCall(PetscMalloc1(1, &rcv_partialCV_buffer));
        PetscCall(PetscArrayfill_custom_int(rcv_partialCV_buffer, 0, 1));

        PetscCall(PetscMalloc1(NbNeighbors, &ReceivedPartialCV));
        PetscCall(PetscArrayfill_custom_bool(ReceivedPartialCV, PETSC_FALSE, NbNeighbors));

        PetscCall(build_spanning_tree(rank_jacobi_block, neighbors, &NbNeighbors, proc_local_rank, nprocs_per_jacobi_block));
        NbNotRecvd = NbNeighbors;

        PetscCall(initialize_state(ACTUAL_PARAMS_POINTERS));
        UnderThreashold = PETSC_FALSE;
        PhaseTag = 0;
    }

    PetscCall(PetscPrintf(PETSC_COMM_SELF, "this is rank %d\n", proc_global_rank));
    
    // PetscScalar LastIterationRegistered = -11;
    // PetscInt idx = proc_local_rank;
    // PetscCall(VecSetValueLocal(LastIteration_global, proc_local_rank, 89 * proc_local_rank, INSERT_VALUES));
    
    // PetscCall(VecAssemblyBegin(LastIteration_global));
    // PetscCall(VecAssemblyEnd(LastIteration_global));
    
    // PetscCall(VecGetValues(LastIteration_global, 1, &idx, &LastIterationRegistered));
    // if (rank_jacobi_block == 0)
    // {
    //     PetscCall(PetscPrintf(PETSC_COMM_SELF, "this is rank %d last iteration %e \n",proc_local_rank, LastIterationRegistered));
    //     PetscCall(VecView(LastIteration_global, PETSC_VIEWER_STDOUT_(comm_jacobi_block)));
    // }
    // PetscCall(PetscBarrier(NULL));
    // PetscCall(PetscFinalize());
    // return 0;

    PetscCall(PetscBarrier(NULL));

    PetscCall(VecCreate(comm_jacobi_block, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, n_mesh_points));
    PetscCall(VecSetType(x, VECMPI));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecSetUp(x));

    PetscCall(VecDuplicate(x, &b));

    PetscCall(VecDuplicate(x, &u));
    PetscScalar initial_scalar_value = 1.0;
    PetscCall(VecSet(u, initial_scalar_value));

    PetscCall(create_matrix_sparse(comm_jacobi_block, &A_block_jacobi, n_mesh_points / njacobi_blocks, n_mesh_points, MATMPIAIJ, 5, 5));

    PetscCall(poisson2DMatrix(&A_block_jacobi, n_mesh_lines, n_mesh_columns, rank_jacobi_block, njacobi_blocks));

    PetscCall(divideSubDomainIntoBlockMatrices(comm_jacobi_block, A_block_jacobi, A_block_jacobi_subMat, is_cols_block_jacobi, rank_jacobi_block, njacobi_blocks, proc_local_rank, nprocs_per_jacobi_block));

    for (PetscInt i = 0; i < njacobi_blocks; i++)
    {
        PetscCall(create_vector(comm_jacobi_block, &b_block_jacobi[i], jacobi_block_size, VECMPI));
        PetscCall(create_vector(comm_jacobi_block, &x_block_jacobi[i], jacobi_block_size, VECMPI));
    }

    PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, ZERO, ONE, &is_jacobi_vec_parts));
    for (PetscInt i = 0; i < njacobi_blocks; i++)
    {
        PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, (i * (jacobi_block_size)), ONE, &is_merged_vec[i]));
        PetscCall(VecScatterCreate(b_block_jacobi[i], is_jacobi_vec_parts, b, is_merged_vec[i], &scatter_jacobi_vec_part_to_merged_vec[i]));
    }

    // PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, &b, b_block_jacobi, u, rank_jacobi_block, jacobi_block_size, nprocs_per_jacobi_block, proc_local_rank));
    PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, b, b_block_jacobi, u, rank_jacobi_block, message_source, message_dest));

    // PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, INNER_KSP_PREFIX, INNER_PC_PREFIX));

    const char *ksp_prefix;
    const char *pc_prefix;
    if (rank_jacobi_block == 0)
    {
        ksp_prefix = "inner1_";
        pc_prefix = "inner1_";
    }

    if (rank_jacobi_block == 1)
    {
        ksp_prefix = "inner2_";
        pc_prefix = "inner2_";
    }

    PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, ksp_prefix, pc_prefix));
    PetscCall(offloadJunk_00001(comm_jacobi_block, rank_jacobi_block, 1));

    PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
    PetscCall(PetscMalloc1(vec_local_size, &send_buffer));
    PetscCall(PetscMalloc1(vec_local_size, &rcv_buffer));

    PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &local_iterates_difference));

    PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &x_block_jacobi_previous_iterate));
    PetscCall(VecCopy(x_block_jacobi[rank_jacobi_block], x_block_jacobi_previous_iterate));
    PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
    PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

    PetscScalar approximation_residual_infinity_norm_iter_zero __attribute__((unused)) = PETSC_MAX_REAL;
    PetscInt inner_solver_iterations __attribute__((unused)) = ZERO;
    PetscInt message_received __attribute__((unused)) = 0;
    PetscInt last_message_received_iter_number __attribute__((unused)) = 0;
    // char *send_pack_buffer = NULL;
    // char *rcv_pack_buffer = NULL;
    // PetscMPIInt other_block_current_iteration = -1;
    // PetscMPIInt current_number_of_iterations = -1;

    // PetscScalar val;
    // PetscCall(VecNorm(b, NORM_2, &val));
    // printf("Norm de b %e \n", val);

    Vec local_residual = NULL;
    PetscCall(VecDuplicate(x_block_jacobi[idx_non_current_block], &local_residual));
    PetscScalar local_norm = PETSC_MAX_REAL;
    PetscScalar local_norm_0 = 0.0;
    PetscCall(MatResidual(A_block_jacobi, b_block_jacobi[rank_jacobi_block], x, local_residual));
    PetscCall(VecNorm(local_residual, NORM_2, &local_norm_0));
    // PetscCall(updateLocalRHS(A_block_jacobi_subMat[idx_non_current_block], x_block_jacobi[idx_non_current_block], b_block_jacobi[rank_jacobi_block], local_right_side_vector));
    // PetscCall(MatResidual(A_block_jacobi_subMat[rank_jacobi_block], local_right_side_vector, x_block_jacobi[rank_jacobi_block], local_residual)); // r_i = b_i - (A_i * x_i)
    // PetscCall(VecNorm(local_residual, NORM_2, &local_norm_0));

    PetscCall(PetscPrintf(comm_jacobi_block, "Rank block %d [local] b norm : %e \n", rank_jacobi_block, local_norm_0));

    PetscScalar global_norm_0 = 0.0;
    PetscCall(computeFinalResidualNorm(comm_jacobi_block, comm_local_roots, A_block_jacobi, x, b_block_jacobi, local_residual, rank_jacobi_block, proc_local_rank, &global_norm_0));
    PetscCall(PetscPrintf(comm_jacobi_block, "Rank block %d [global] b norm : %e \n", rank_jacobi_block, global_norm_0));

    PetscCall(PetscBarrier(NULL));
    double start_time, end_time;
    start_time = MPI_Wtime();

    do
    {

        message_received = 0;
        inner_solver_iterations = 0;

        PetscCall(comm_async_probe_and_receive_prime(x_block_jacobi,
                                                     rcv_buffer, vec_local_size, message_source, idx_non_current_block,
                                                     &dependency_received, &neighbor_current_iteration,
                                                     &rcv_pack_buffer, NewerDependencies_global, LastIteration_global,
                                                     state, PhaseTag, &scatter_ctx, NewerDependencies_local, proc_local_rank));

        // PetscCall(PetscPrintf(MPI_COMM_SELF, "[rank %d] Another Another hot spot!\n", proc_global_rank));

        PetscCall(updateLocalRHS(A_block_jacobi_subMat[idx_non_current_block], x_block_jacobi[idx_non_current_block], b_block_jacobi[rank_jacobi_block], local_right_side_vector));

        PetscCall(inner_solver(comm_jacobi_block, inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, local_right_side_vector, rank_jacobi_block, &inner_solver_iterations, number_of_iterations));

        PetscCall(comm_async_test_and_send_prime(PhaseTag, number_of_iterations, x_block_jacobi, send_buffer,
                                                 &send_data_request, vec_local_size, message_dest,
                                                 rank_jacobi_block, &send_pack_buffer));

        PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

        PetscCall(MatResidual(A_block_jacobi_subMat[rank_jacobi_block], local_right_side_vector, x_block_jacobi[rank_jacobi_block], local_residual));
        PetscCall(VecNorm(local_residual, NORM_2, &local_norm));

        PetscCall(PetscPrintf(comm_jacobi_block, "[Rank %d] Local norm_2 block  = %e \n", rank_jacobi_block, local_norm));

        if (proc_local_rank == 0) // ONLY root node from each block check for convergence
        {

            if (local_norm <= PetscMax(absolute_tolerance, (relative_tolerance / PetscSqrtScalar(2.0)) * 0.7 * global_norm_0))
            {

                UnderThreashold = PETSC_TRUE;
            }
            else
            {
                UnderThreashold = PETSC_FALSE;
                // UnderThreashold = PETSC_TRUE;
            }

            // PseudoPeriodBegin = PETSC_TRUE;
            // PseudoPeriodEnd = PETSC_TRUE;

            // Main convergence detection mechanism
            PetscCall(comm_async_convDetection_prime(ACTUAL_PARAMS_POINTERS));

            // local convergence
            PetscCall(receive_partial_CV(ACTUAL_PARAMS_POINTERS, proc_global_rank));

            // receive verificication phase
            PetscCall(receive_verification(ACTUAL_PARAMS_POINTERS));

            // response after verificiation
            PetscCall(receive_response(ACTUAL_PARAMS_POINTERS));

            // receive final verdict
            PetscCall(receive_verdict(ACTUAL_PARAMS_POINTERS));
        }

        PetscCallMPI(MPI_Bcast(&state, 1, MPIU_INT, LOCAL_ROOT_NODE, comm_jacobi_block));
        PetscCallMPI(MPI_Bcast(&PhaseTag, 1, MPIU_INT, LOCAL_ROOT_NODE, comm_jacobi_block));

        PetscCall(PetscPrintf(comm_jacobi_block, "[rank %d]   STATE %d ELECTED %d\n", proc_global_rank, state, ElectedNode));

        // PetscCall(PetscPrintf(comm_jacobi_block, "[rank %d]   hot spot!\n", proc_global_rank));
        // if (state == VERIFICATION )
        // {

        //     PetscScalar LastIterationRegistered = -2;
        //     PetscInt idx = 0;
        //     PetscCall(VecGetValues(LastIteration_global, 1, &idx, &LastIterationRegistered));
        //     PetscCall(PetscPrintf(MPI_COMM_SELF, "[rank %d iteration %d] last new depedency received at iteration: %e\n", proc_global_rank, number_of_iterations, LastIterationRegistered));
        //     PetscCall(PetscSleep(1000));
        // }
        // PetscCall(PetscPrintf(comm_jacobi_block, "[rank %d]  Another hot spot!\n", proc_global_rank));

        number_of_iterations = number_of_iterations + 1;
        // if (number_of_iterations >= 10)
        //     PetscCall(PetscSleep(100));
        // if (number_of_iterations >= 20)
        // {

        //     PetscScalar LastIterationRegistered = -2;
        //     PetscInt idx = 0;
        //     PetscCall(VecGetValues(LastIteration_global, 1, &idx, &LastIterationRegistered));
        //     PetscCall(PetscPrintf(MPI_COMM_SELF, "[rank %d iteration %d] last new depedency received at iteration: %e\n", proc_global_rank,number_of_iterations, LastIterationRegistered));
        //     PetscCall(PetscSleep(1000));
        // }

    } while (state != FINISHED);

    // MPI_Request requests[nbNeighbors];
    // // for (PetscInt i = 0; i < nbNeighbors; i++)
    // // {
    // //     requests[i] = MPI_REQUEST_NULL;
    // // }
    // PetscCall(comm_async_sendGlobalCV(rank_jacobi_block, nbNeighbors, neighbors, &globalCV, requests));
    // PetscCallMPI(MPI_Waitall(nbNeighbors, requests, MPI_STATUSES_IGNORE));

    PetscCall(PetscPrintf(comm_jacobi_block, "Rank %d: PROGRAMME TERMINE\n", rank_jacobi_block));
    PetscCall(PetscBarrier(NULL));

    end_time = MPI_Wtime();
    PetscCall(printElapsedTime(start_time, end_time));
    PetscCall(printTotalNumberOfIterations(comm_jacobi_block, rank_jacobi_block, number_of_iterations));

    PetscCall(PetscBarrier(NULL));

    PetscCall(MatResidual(A_block_jacobi, b_block_jacobi[rank_jacobi_block], x, local_residual));
    PetscCall(VecNorm(local_residual, NORM_2, &local_norm));
    PetscCall(PetscPrintf(comm_jacobi_block, "[Rank %d] Local norm_2 FINISHED  = %e \n", rank_jacobi_block, local_norm));

    PetscCall(PetscBarrier(NULL));

    PetscCall(comm_sync_send_and_receive_final(x_block_jacobi, vec_local_size, message_dest, message_source, rank_jacobi_block, idx_non_current_block));
    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

    PetscScalar norm;
    PetscCall(computeFinalResidualNorm(comm_jacobi_block, comm_local_roots, A_block_jacobi, x, b_block_jacobi, local_residual, rank_jacobi_block, proc_local_rank, &norm));
    PetscCall(printFinalResidualNorm(norm));

    PetscScalar error;
    PetscCall(computeError(x, u, &error));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Erreur : %e \n", error));

    // END OF PROGRAM  - FREE MEMORY
    // Discard any pending message
    PetscCall(comm_discard_pending_messages());

    // for (PetscInt i = 0; i < nbNeighbors; i++)
    // {
    //     if (requests[i] != MPI_REQUEST_NULL)
    //     {
    //         PetscCallMPI(MPI_Cancel(&requests[i]));
    //         PetscCallMPI(MPI_Request_free(&requests[i]));
    //     }
    // }

    // if (cancelSPartialRequest != MPI_REQUEST_NULL)
    // {

    //     PetscCallMPI(MPI_Cancel(&cancelSPartialRequest));
    //     PetscCallMPI(MPI_Request_free(&cancelSPartialRequest));
    // }

    // if (sendSPartialRequest != MPI_REQUEST_NULL)
    // {
    //     PetscCallMPI(MPI_Cancel(&sendSPartialRequest));
    //     PetscCallMPI(MPI_Request_free(&sendSPartialRequest));
    // }

    if (send_data_request != MPI_REQUEST_NULL)
    {
        PetscCallMPI(MPI_Cancel(&send_data_request));
        PetscCallMPI(MPI_Request_free(&send_data_request));
    }

    PetscCall(PetscBarrier(NULL));

    PetscCall(ISDestroy(&is_jacobi_vec_parts));
    for (PetscInt i = 0; i < njacobi_blocks; i++)
    {
        PetscCall(ISDestroy(&is_merged_vec[i]));
        PetscCall(ISDestroy(&is_cols_block_jacobi[i]));
        PetscCall(VecDestroy(&x_block_jacobi[i]));
        PetscCall(VecDestroy(&b_block_jacobi[i]));
        PetscCall(MatDestroy(&A_block_jacobi_subMat[i]));
        PetscCall(VecScatterDestroy(&scatter_jacobi_vec_part_to_merged_vec[i]));
    }

    PetscCall(PetscFree(rcv_buffer));
    PetscCall(PetscFree(send_buffer));
    PetscCall(VecDestroy(&x_block_jacobi_previous_iterate));
    PetscCall(VecDestroy(&local_iterates_difference));
    PetscCall(VecDestroy(&local_right_side_vector));
    PetscCall(VecDestroy(&mat_mult_vec_result));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&local_residual));
    PetscCall(MatDestroy(&A_block_jacobi));
    PetscCall(KSPDestroy(&inner_ksp));
    // PetscCall(PetscFree(prevIterNumS));
    // PetscCall(PetscFree(prevIterNumC));
    // PetscCall(PetscFree(neighbors));

    PetscCall(PetscSubcommDestroy(&sub_comm_context));
    PetscCall(PetscCommDestroy(&dcomm));
    PetscCall(PetscBarrier(NULL));
    PetscCall(PetscFinalize());
    return 0;
}

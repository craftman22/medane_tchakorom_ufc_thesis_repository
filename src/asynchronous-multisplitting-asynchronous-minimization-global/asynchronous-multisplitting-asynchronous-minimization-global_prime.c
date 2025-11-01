#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"
#include "constants.h"
#include "comm.h"
#include "utils.h"
#include "petscdraw.h"
#include "petscviewer.h"
#include "comm.h"
#include "conv_detection.h"
#include "conv_detection_prime.h"

// #ifdef VERSION_1_0

int main(int argc, char **argv)
{

    Mat A_block_jacobi = NULL;
    Mat A_block_jacobi_resdistributed = NULL;
    Vec x = NULL; // vector of unknows
    Vec b = NULL; // right hand side vector
    Vec u = NULL;
    PetscInt s;
    PetscMPIInt nprocs;
    PetscMPIInt proc_global_rank;
    PetscInt n_mesh_lines = 4;
    PetscInt n_mesh_columns = 4;
    PetscMPIInt njacobi_blocks;
    PetscMPIInt rank_jacobi_block;
    PetscMPIInt proc_local_rank;
    PetscInt n_mesh_points;
    PetscInt jacobi_block_size;
    PetscInt nprocs_per_jacobi_block = 1;
    PetscScalar relative_tolerance = 1e-5;
    PetscScalar absolute_tolerance = 1e-100;
    PetscSubcomm sub_comm_context;
    MPI_Comm dcomm;
    MPI_Comm comm_jacobi_block;

    IS is_jacobi_vec_parts;
    PetscInt number_of_iterations = ZERO;
    PetscMPIInt idx_non_current_block;
    KSP inner_ksp = NULL;
    KSP outer_ksp = NULL;
    PetscInt vec_local_size = 0;
    PetscScalar *send_minimization_data_buffer = NULL;
    PetscScalar *rcv_minimization_data_buffer = NULL;
    PetscScalar *temp_minimization_data_buffer = NULL;

    MPI_Request send_minimization_data_request = MPI_REQUEST_NULL;
    PetscMPIInt rcv_minimization_data_flag = 0;

    PetscInt *vec_local_idx = NULL;
    PetscInt x_local_size;
    PetscScalar *vector_to_insert_into_S;

    PetscInt MIN_CONVERGENCE_COUNT = 5;

    MPI_Status status __attribute__((unused));

    // Minimization variables

    Mat R = NULL;
    Mat S = NULL;
    PetscInt n_vectors_inserted;
    Vec x_minimized = NULL;
    Vec global_iterates_difference = NULL;
    Vec local_right_side_vector = NULL;
    Vec mat_mult_vec_result = NULL;

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

    PetscCall(computeDimensionRelatedVariables(nprocs, nprocs_per_jacobi_block, proc_global_rank, n_mesh_lines, n_mesh_columns, &njacobi_blocks, &rank_jacobi_block, &proc_local_rank, &n_mesh_points, &jacobi_block_size));
    PetscAssert((n_mesh_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");
    // Creating the sub communicator for each jacobi block
    sub_comm_context = NULL;
    PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD, &dcomm, NULL));
    PetscCall(PetscSubcommCreate(dcomm, &sub_comm_context));
    PetscCall(PetscSubcommSetNumber(sub_comm_context, njacobi_blocks));
    PetscCall(PetscSubcommSetType(sub_comm_context, PETSC_SUBCOMM_CONTIGUOUS));
    comm_jacobi_block = PetscSubcommChild(sub_comm_context);

    MPI_Comm comm_local_roots; // communicator only for local root procs
    int color = (proc_global_rank == 0 || proc_global_rank == nprocs_per_jacobi_block) ? 0 : MPI_UNDEFINED;
    PetscCallMPI(MPI_Comm_split(MPI_COMM_WORLD, color, 0, &comm_local_roots));

    idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;
    IS is_cols_block_jacobi[njacobi_blocks];
    Mat A_block_jacobi_subMat[njacobi_blocks];
    Mat R_block_jacobi_subMat[njacobi_blocks];
    Vec b_block_jacobi[njacobi_blocks];
    Vec x_block_jacobi[njacobi_blocks];
    VecScatter scatter_jacobi_vec_part_to_merged_vec[njacobi_blocks];
    IS is_merged_vec[njacobi_blocks];
    Mat R_transpose_R = NULL;
    Vec vec_R_transpose_b_block_jacobi = NULL;
    Vec alpha = NULL;
    PetscInt lda;
    PetscMPIInt R_local_values_count;
    PetscMPIInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
    PetscMPIInt message_dest = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
    PetscInt rstart, rend;

    // PetscInt convergence_count = ZERO;
    // PetscMPIInt broadcast_message = NO_MESSAGE;
    // PetscInt *LastIteration = NULL;
    // PetscBool *NewerDependencies = NULL;

    PetscScalar *send_buffer = NULL;
    PetscScalar *rcv_buffer = NULL;
    MPI_Request send_data_request = MPI_REQUEST_NULL;
    Vec LastIteration_global = NULL;
    Vec NewerDependencies_global = NULL;
    Vec LastIteration_local = NULL;
    Vec NewerDependencies_local = NULL;
    Vec UNITARY_VECTOR = NULL;
    Vec NULL_VECTOR = NULL;

    PetscInt NbNeighbors = 0;
    PetscBool UnderThreashold = PETSC_FALSE;
    PetscInt NbDependencies = 0;
    PetscBool PseudoPeriodBegin = PETSC_FALSE;
    PetscBool PseudoPeriodEnd = PETSC_FALSE;
    PetscBool ElectedNode = PETSC_FALSE;
    PetscInt PhaseTag = 0;
    PetscBool ResponseSent = PETSC_FALSE;
    State state = NORMAL;
    PetscBool LocalCV = PETSC_FALSE;
    PetscInt NbNotRecvd = 0;
    PetscBool PartialCVSent = PETSC_FALSE;
    PetscInt *Responses = NULL;
    PetscBool *ReceivedPartialCV = NULL;
    PetscInt *neighbors = NULL; /* array containing all the node neighbors */
    PetscInt *send_verdict_buffer = NULL;
    PetscInt *rcv_verdict_buffer = NULL;
    PetscInt *send_response_buffer = NULL;
    PetscInt *rcv_response_buffer = NULL;
    PetscInt *send_verification_buffer = NULL;
    PetscInt *rcv_verification_buffer = NULL;
    PetscInt *send_partialCV_buffer = NULL;
    PetscInt *rcv_partialCV_buffer = NULL;
    char *send_pack_buffer = NULL;
    char *rcv_pack_buffer = NULL;
    MPI_Request send_verdict_request = MPI_REQUEST_NULL;
    MPI_Request send_response_request = MPI_REQUEST_NULL;
    MPI_Request send_verification_request = MPI_REQUEST_NULL;
    MPI_Request send_CV_request = MPI_REQUEST_NULL;
    PetscInt dependency_received = 0;
    PetscInt neighbor_current_iteration = 0;

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

    PetscCall(PetscBarrier(NULL));
    for (PetscMPIInt i = 0; i < njacobi_blocks; i++)
    {
        PetscCall(create_vector(comm_jacobi_block, &b_block_jacobi[i], jacobi_block_size, VECMPI));
        PetscCall(create_vector(comm_jacobi_block, &x_block_jacobi[i], jacobi_block_size, VECMPI));
    }

    PetscCall(create_matrix_sparse(comm_jacobi_block, &A_block_jacobi, n_mesh_points / njacobi_blocks, n_mesh_points, MATMPIAIJ, 5, 5));
    // Insert non-zeros entries into the operator matrix
    PetscCall(poisson2DMatrix(&A_block_jacobi, n_mesh_lines, n_mesh_columns, rank_jacobi_block, njacobi_blocks));

    PetscCall(create_matrix_dense(comm_jacobi_block, &R, n_mesh_points, s, MATMPIDENSE));
    PetscCall(MatZeroEntries(R));
    PetscCall(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));

    PetscCall(getHalfSubMatrixFromR(R, R_block_jacobi_subMat, n_mesh_lines, n_mesh_columns, rank_jacobi_block));

    PetscInt redistributed_local_size;
    PetscInt first_row_owned;
    PetscCall(MatGetLocalSize(R_block_jacobi_subMat[rank_jacobi_block], &redistributed_local_size, NULL));
    PetscCall(MatGetOwnershipRange(R_block_jacobi_subMat[rank_jacobi_block], &first_row_owned, NULL));
    PetscCall(create_redistributed_A_block_jacobi(comm_jacobi_block, A_block_jacobi, &A_block_jacobi_resdistributed, nprocs_per_jacobi_block, proc_local_rank, redistributed_local_size, first_row_owned));
    PetscCall(restoreHalfSubMatrixToR(R, R_block_jacobi_subMat, rank_jacobi_block));

    PetscCall(MatDenseGetLDA(R, &lda));
    R_local_values_count = s * lda;
    PetscCall(PetscMalloc1(R_local_values_count, &send_minimization_data_buffer));
    PetscCall(PetscArrayzero(send_minimization_data_buffer, R_local_values_count));
    PetscCall(PetscMalloc1(R_local_values_count, &rcv_minimization_data_buffer));
    PetscCall(PetscArrayzero(rcv_minimization_data_buffer, R_local_values_count));
    PetscCall(MatGetOwnershipRange(R, &rstart, &rend));

    PetscCall(create_matrix_dense(comm_jacobi_block, &S, n_mesh_points, s, MATMPIDENSE));

    PetscCall(create_matrix_dense(comm_jacobi_block, &R_transpose_R, s, s, MATMPIDENSE));

    PetscCall(create_vector(comm_jacobi_block, &vec_R_transpose_b_block_jacobi, s, VECMPI));
    PetscCall(create_vector(comm_jacobi_block, &alpha, s, VECMPI));
    PetscCall(create_vector(comm_jacobi_block, &x, n_mesh_points, VECMPI));

    PetscCall(VecDuplicate(x, &b));
    PetscCall(VecDuplicate(x, &u));
    PetscCall(VecSet(u, ONE));

    PetscCall(divideSubDomainIntoBlockMatrices(comm_jacobi_block, A_block_jacobi, A_block_jacobi_subMat, is_cols_block_jacobi, rank_jacobi_block, njacobi_blocks, proc_local_rank, nprocs_per_jacobi_block));

    PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, ZERO, ONE, &is_jacobi_vec_parts));

    for (PetscMPIInt i = 0; i < njacobi_blocks; i++)
    {
        PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, (i * (jacobi_block_size)), ONE, &is_merged_vec[i]));
        PetscCall(VecScatterCreate(b_block_jacobi[i], is_jacobi_vec_parts, b, is_merged_vec[i], &scatter_jacobi_vec_part_to_merged_vec[i]));
    }

    // PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, &b, b_block_jacobi, u, rank_jacobi_block, jacobi_block_size, nprocs_per_jacobi_block, proc_local_rank));

    PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, b, b_block_jacobi, u, rank_jacobi_block, message_source, message_dest));

    // PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, INNER_KSP_PREFIX, INNER_PC_PREFIX));
    // PetscCall(initializeKSP(comm_jacobi_block, &outer_ksp, NULL, rank_jacobi_block, PETSC_TRUE, OUTER_KSP_PREFIX, OUTER_PC_PREFIX));

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

    if (rank_jacobi_block == 0)
    {
        ksp_prefix = "outer1_";
        pc_prefix = "outer1_";
    }

    if (rank_jacobi_block == 1)
    {
        ksp_prefix = "outer2_";
        pc_prefix = "outer2_";
    }

    PetscCall(initializeKSP(comm_jacobi_block, &outer_ksp, NULL, rank_jacobi_block, PETSC_TRUE, ksp_prefix, pc_prefix));
    PetscCall(offloadJunk_00001(comm_jacobi_block, rank_jacobi_block, 2));

    PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
    PetscCall(PetscMalloc1(vec_local_size, &send_buffer));
    PetscCall(PetscArrayzero(send_buffer, vec_local_size));
    PetscCall(PetscMalloc1(vec_local_size, &rcv_buffer));
    PetscCall(PetscArrayzero(rcv_buffer, vec_local_size));

    PetscCall(create_vector(comm_jacobi_block, &x_minimized, n_mesh_points, VECMPI));
    PetscCall(VecSet(x_minimized, ZERO));

    PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
    PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

    PetscCall(VecGetLocalSize(x, &x_local_size));
    PetscCall(PetscMalloc1(x_local_size, &vec_local_idx));
    for (PetscMPIInt i = 0; i < (x_local_size); i++)
    {
        vec_local_idx[i] = (proc_local_rank * x_local_size) + i;
    }
    PetscCall(PetscMalloc1(x_local_size, &vector_to_insert_into_S));

    PetscCall(VecDuplicate(x_minimized, &global_iterates_difference));

    Vec local_residual = NULL;
    PetscCall(VecDuplicate(x_block_jacobi[idx_non_current_block], &local_residual));
    PetscScalar local_norm = PETSC_MAX_REAL;
    PetscScalar local_norm_0 = 0.0;
    PetscCall(MatResidual(A_block_jacobi, b_block_jacobi[rank_jacobi_block], x, local_residual));
    PetscCall(VecNorm(local_residual, NORM_2, &local_norm_0));
    // PetscCall(updateLocalRHS(A_block_jacobi_subMat[idx_non_current_block], x_block_jacobi[idx_non_current_block], b_block_jacobi[rank_jacobi_block], local_right_side_vector));
    // PetscCall(MatResidual(A_block_jacobi_subMat[rank_jacobi_block], local_right_side_vector, x_block_jacobi[rank_jacobi_block], local_residual)); // r_i = b_i - (A_i * x_i)
    // PetscCall(VecNorm(local_residual, NORM_2, &local_norm_0));

    PetscScalar global_norm_0 = 0.0;
    PetscCall(computeFinalResidualNorm(comm_jacobi_block, comm_local_roots, A_block_jacobi, x, b_block_jacobi, local_residual, rank_jacobi_block, proc_local_rank, &global_norm_0));
    PetscCall(PetscPrintf(comm_jacobi_block, "Rank block %d [global] b norm : %e \n", rank_jacobi_block, global_norm_0));

    PetscInt message_received __attribute__((unused)) = 0;
    PetscInt inner_solver_iterations __attribute__((unused)) = ZERO;

    PetscLogEvent USER_EVENT;
    PetscCall(PetscLogEventRegister("outer_solve", 0, &USER_EVENT));

    PetscCall(PetscBarrier(NULL));
    double start_time, end_time;
    start_time = MPI_Wtime();

    do
    {

        n_vectors_inserted = 0;
        message_received = 0;
        inner_solver_iterations = 0;

        while (n_vectors_inserted < s)
        {

            PetscCall(comm_async_probe_and_receive_prime(x_block_jacobi,
                                                         rcv_buffer, vec_local_size, message_source, idx_non_current_block,
                                                         &dependency_received, &neighbor_current_iteration,
                                                         &rcv_pack_buffer, NewerDependencies_global, LastIteration_global,
                                                         state, PhaseTag, &scatter_ctx, NewerDependencies_local, proc_local_rank));

            PetscCall(updateLocalRHS(A_block_jacobi_subMat[idx_non_current_block], x_block_jacobi[idx_non_current_block],
                                     b_block_jacobi[rank_jacobi_block], local_right_side_vector));

            PetscCall(inner_solver(comm_jacobi_block, inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi,
                                   local_right_side_vector, rank_jacobi_block, &inner_solver_iterations, number_of_iterations));

            PetscCall(comm_async_test_and_send_prime(PhaseTag, number_of_iterations, x_block_jacobi, send_buffer,
                                                     &send_data_request, vec_local_size, message_dest,
                                                     rank_jacobi_block, &send_pack_buffer));

            PetscCall(comm_async_probe_and_receive_prime(x_block_jacobi,
                                                         rcv_buffer, vec_local_size, message_source, idx_non_current_block,
                                                         &dependency_received, &neighbor_current_iteration,
                                                         &rcv_pack_buffer, NewerDependencies_global, LastIteration_global,
                                                         state, PhaseTag, &scatter_ctx, NewerDependencies_local, proc_local_rank));

            // TODO: ajouter ici une possible reception de données afin que cela soit cohérent

            PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
            PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
            PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
            PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

            PetscCall(VecGetValues(x, x_local_size, vec_local_idx, vector_to_insert_into_S));

            PetscCall(MatSetValuesLocal(S, x_local_size, vec_local_idx, ONE, &n_vectors_inserted, vector_to_insert_into_S, INSERT_VALUES));

            n_vectors_inserted++;
        }

        PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));

        PetscCall(getHalfSubMatrixFromR(R, R_block_jacobi_subMat, n_mesh_lines, n_mesh_columns, rank_jacobi_block));
        PetscCall(MatMatMult(A_block_jacobi_resdistributed, S, MAT_REUSE_MATRIX, PETSC_DETERMINE, &R_block_jacobi_subMat[rank_jacobi_block]));
        PetscCall(restoreHalfSubMatrixToR(R, R_block_jacobi_subMat, rank_jacobi_block));

        PetscCall(comm_async_test_and_send_min(R, send_minimization_data_buffer, temp_minimization_data_buffer,
                                               &send_minimization_data_request, R_local_values_count, message_dest,
                                               rank_jacobi_block));

        PetscCall(comm_async_probe_and_receive_min(R, rcv_minimization_data_buffer, temp_minimization_data_buffer,
                                                   R_local_values_count, rcv_minimization_data_flag, message_source, rank_jacobi_block, idx_non_current_block,
                                                   n_mesh_points, rstart, rend, lda, s));

        PetscCall(PetscBarrier((PetscObject)S));
        PetscCall(outer_solver_norm_equation(comm_jacobi_block, outer_ksp, x_minimized, R, S, alpha, b, rank_jacobi_block, number_of_iterations));

        PetscCall(MatResidual(A_block_jacobi, b_block_jacobi[rank_jacobi_block], x_minimized, local_residual));
        PetscCall(VecNorm(local_residual, NORM_2, &local_norm));

        PetscCall(PetscPrintf(comm_jacobi_block, "[Rank %d] Local norm_2 block  = %e \n", rank_jacobi_block, local_norm));

        PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_minimized, x_block_jacobi[rank_jacobi_block], INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_minimized, x_block_jacobi[rank_jacobi_block], INSERT_VALUES, SCATTER_REVERSE));

        if (proc_local_rank == 0) // ONLY root node from each block check for convergence
        {

            if (local_norm <= PetscMax(absolute_tolerance, (relative_tolerance / PetscSqrtScalar(2.0)) * 1.0 * global_norm_0))
            {
                PetscCall(PetscPrintf(comm_jacobi_block, "[Rank %d] Under the threashold  = %e \n", rank_jacobi_block, local_norm));
                UnderThreashold = PETSC_TRUE;
            }
            else
            {
                UnderThreashold = PETSC_FALSE;
            }

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

        number_of_iterations = number_of_iterations + 1;

    } while (state != FINISHED);

    PetscCall(PetscPrintf(comm_jacobi_block, "Rank %d: PROGRAMME TERMINE\n", rank_jacobi_block));
    PetscCall(PetscBarrier(NULL));

    end_time = MPI_Wtime();
    PetscCall(printElapsedTime(start_time, end_time));
    PetscCall(printTotalNumberOfIterations_2(comm_jacobi_block, rank_jacobi_block, number_of_iterations, s));

    PetscCall(PetscBarrier(NULL));

    /////
    PetscCall(MatResidual(A_block_jacobi, b_block_jacobi[rank_jacobi_block], x_minimized, local_residual));
    PetscCall(VecNorm(local_residual, NORM_2, &local_norm));
    PetscCall(PetscPrintf(comm_jacobi_block, "[Block Rank %d] Local norm_2 with x before exchange or average = %e \n", rank_jacobi_block, local_norm));

    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_minimized, x_block_jacobi[rank_jacobi_block], INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_minimized, x_block_jacobi[rank_jacobi_block], INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(comm_sync_send_and_receive_final(x_block_jacobi, vec_local_size, message_dest, message_source, rank_jacobi_block, idx_non_current_block));
    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(MatResidual(A_block_jacobi, b_block_jacobi[rank_jacobi_block], x, local_residual));
    PetscCall(VecNorm(local_residual, NORM_2, &local_norm));
    PetscCall(PetscPrintf(comm_jacobi_block, "[Block Rank %d] Local norm_2 with x after exchange = %e \n", rank_jacobi_block, local_norm));

    ///////////
    Vec x_minimized_opposite_block;
    PetscCall(VecDuplicate(x_minimized, &x_minimized_opposite_block));
    PetscCall(VecCopy(x_minimized, x_minimized_opposite_block));
    PetscInt local_size;
    PetscCall(VecGetLocalSize(x_minimized, &local_size));

    PetscScalar *send_buffer_c = NULL;
    PetscScalar *rcv_buffer_c = NULL;

    PetscCall(VecGetArray(x_minimized, &send_buffer_c));
    PetscCall(VecGetArray(x_minimized_opposite_block, &rcv_buffer_c));

    PetscCallMPI(MPI_Sendrecv(send_buffer_c, local_size, MPIU_SCALAR, message_dest, (TAG_FINAL_DATA_EXCHANGE + rank_jacobi_block), rcv_buffer_c, local_size, MPIU_SCALAR, message_source, (TAG_FINAL_DATA_EXCHANGE + idx_non_current_block), MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    PetscCall(VecRestoreArray(x_minimized, &send_buffer_c));
    PetscCall(VecRestoreArray(x_minimized_opposite_block, &rcv_buffer_c));

    PetscCall(PetscFree(send_buffer_c));
    PetscCall(PetscFree(rcv_buffer_c));

    PetscCall(VecAXPY(x_minimized, 1.0, x_minimized_opposite_block));
    PetscCall(VecScale(x_minimized, 0.5));

    PetscCall(MatResidual(A_block_jacobi, b_block_jacobi[rank_jacobi_block], x_minimized, local_residual));
    PetscCall(VecNorm(local_residual, NORM_2, &local_norm));
    PetscCall(PetscPrintf(comm_jacobi_block, "[Block Rank %d] Local norm_2 with x after average  = %e \n", rank_jacobi_block, local_norm));

    PetscScalar norm;
    PetscCall(computeFinalResidualNorm(comm_jacobi_block, comm_local_roots, A_block_jacobi, x_minimized, b_block_jacobi, local_residual, rank_jacobi_block, proc_local_rank, &norm));
    PetscCall(printFinalResidualNorm(norm));

    PetscScalar error;
    PetscCall(computeError(x_minimized, u, &error));
    PetscCall(PetscPrintf(comm_jacobi_block, "[BLOCK %d] Erreur  : %e  \n", rank_jacobi_block, error));

    // END OF PROGRAM  - FREE MEMORY

    // Discard any pending message
    PetscCall(comm_discard_pending_messages());

    PetscCall(PetscBarrier(NULL));

    PetscCall(comm_discard_pending_messages());

    if (send_verdict_request != MPI_REQUEST_NULL)
    {

        PetscCallMPI(MPI_Cancel(&send_verdict_request));
        PetscCallMPI(MPI_Request_free(&send_verdict_request));
    }

    if (send_response_request != MPI_REQUEST_NULL)
    {
        PetscCallMPI(MPI_Cancel(&send_response_request));
        PetscCallMPI(MPI_Request_free(&send_response_request));
    }

    if (send_verification_request != MPI_REQUEST_NULL)
    {
        PetscCallMPI(MPI_Cancel(&send_verification_request));
        PetscCallMPI(MPI_Request_free(&send_verification_request));
    }

    if (send_CV_request != MPI_REQUEST_NULL)
    {
        PetscCallMPI(MPI_Cancel(&send_CV_request));
        PetscCallMPI(MPI_Request_free(&send_CV_request));
    }

    if (send_data_request != MPI_REQUEST_NULL)
    {
        PetscCallMPI(MPI_Cancel(&send_data_request));
        PetscCallMPI(MPI_Request_free(&send_data_request));
    }

    if (send_minimization_data_request != MPI_REQUEST_NULL)
    {
        PetscCallMPI(MPI_Cancel(&send_minimization_data_request));
        PetscCallMPI(MPI_Request_free(&send_minimization_data_request));
    }

    PetscCall(PetscBarrier(NULL));

    for (PetscInt i = 0; i < njacobi_blocks; i++)
    {
        PetscCall(ISDestroy(&is_cols_block_jacobi[i]));
        PetscCall(VecDestroy(&x_block_jacobi[i]));
        PetscCall(VecDestroy(&b_block_jacobi[i]));
        PetscCall(MatDestroy(&A_block_jacobi_subMat[i]));
        PetscCall(VecScatterDestroy(&scatter_jacobi_vec_part_to_merged_vec[i]));
        PetscCall(ISDestroy(&is_merged_vec[i]));
    }

    PetscCall(PetscFree(Responses));
    PetscCall(PetscFree(ReceivedPartialCV));
    PetscCall(PetscFree(neighbors));
    PetscCall(PetscFree(send_verdict_buffer));
    PetscCall(PetscFree(rcv_verdict_buffer));
    PetscCall(PetscFree(send_response_buffer));
    PetscCall(PetscFree(rcv_response_buffer));

    PetscCall(PetscFree(send_verification_buffer));
    PetscCall(PetscFree(rcv_verification_buffer));

    PetscCall(PetscFree(send_partialCV_buffer));
    PetscCall(PetscFree(rcv_partialCV_buffer));
    PetscCall(PetscFree(send_pack_buffer));
    PetscCall(PetscFree(rcv_pack_buffer));

    PetscCall(VecScatterDestroy(&scatter_ctx));
    PetscCall(VecDestroy(&LastIteration_global));
    PetscCall(VecDestroy(&NewerDependencies_global));
    PetscCall(VecDestroy(&NewerDependencies_local));
    PetscCall(VecDestroy(&LastIteration_local));
    PetscCall(VecDestroy(&UNITARY_VECTOR));
    PetscCall(VecDestroy(&NULL_VECTOR));

    PetscCall(PetscFree(rcv_buffer));
    PetscCall(PetscFree(send_buffer));
    PetscCall(PetscFree(send_minimization_data_buffer));
    PetscCall(PetscFree(rcv_minimization_data_buffer));
    PetscCall(PetscFree(vec_local_idx));
    PetscCall(PetscFree(vector_to_insert_into_S));
    PetscCall(VecDestroy(&global_iterates_difference));
    PetscCall(VecDestroy(&local_right_side_vector));
    PetscCall(VecDestroy(&local_residual));
    PetscCall(VecDestroy(&mat_mult_vec_result));
    PetscCall(ISDestroy(&is_jacobi_vec_parts));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&x_minimized));
    PetscCall(VecDestroy(&x_minimized_opposite_block));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&A_block_jacobi));
    PetscCall(MatDestroy(&A_block_jacobi_resdistributed));
    PetscCall(MatDestroy(&S));
    PetscCall(MatDestroy(&R));

    PetscCall(KSPDestroy(&inner_ksp));
    PetscCall(KSPDestroy(&outer_ksp));
    PetscCall(MatDestroy(&R_transpose_R));
    PetscCall(VecDestroy(&vec_R_transpose_b_block_jacobi));
    PetscCall(VecDestroy(&alpha));

    PetscCall(PetscSubcommDestroy(&sub_comm_context));
    PetscCall(PetscCommDestroy(&dcomm));
    PetscCall(PetscFinalize());
    return 0;
}

// PetscCall(printResidualNorm(comm_jacobi_block, rank_jacobi_block, global_iterates_difference_norm_inf, number_of_iterations));
// if (global_iterates_difference_norm_inf <= PetscMax(absolute_tolerance, relative_tolerance * current_iterate_norm_inf))
// {
//     preLocalCV = PETSC_TRUE;
// }
// else
// {
//     preLocalCV = PETSC_FALSE;
// }

// PetscCall(VecWAXPY(global_iterates_difference, -1.0, x_minimized_prev_iteration, x_minimized));
// PetscCall(VecNorm(global_iterates_difference, NORM_INFINITY, &global_iterates_difference_norm_inf));
// PetscCall(VecNorm(x_minimized, NORM_INFINITY, &current_iterate_norm_inf));

// PetscScalar direct_residual_norm;
// PetscCall(computeFinalResidualNorm(A_block_jacobi, x_minimized, b_block_jacobi, rank_jacobi_block, proc_local_rank, &direct_residual_norm));
// PetscCall(printFinalResidualNorm(direct_residual_norm));
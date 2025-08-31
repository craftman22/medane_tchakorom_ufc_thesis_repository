#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"
#include "constants.h"
#include "utils.h"
#include "comm.h"
#include "petscdraw.h"
#include "petscviewer.h"
#include "conv_detection.h"

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
    PetscMPIInt nprocs_per_jacobi_block = 1;
    PetscScalar relative_tolerance = 1e-5;
    PetscScalar absolute_tolerance = 1e-100;
    PetscSubcomm sub_comm_context;
    MPI_Comm dcomm;
    MPI_Comm comm_jacobi_block;

    IS is_jacobi_vec_parts;
    PetscInt number_of_iterations = ZERO;
    PetscMPIInt idx_non_current_block;
    // PetscScalar local_iterates_difference_norm_inf = PETSC_MAX_REAL;
    // PetscScalar current_iterate_norm_inf = PETSC_MAX_REAL;
    KSP inner_ksp = NULL;
    KSP outer_ksp = NULL;
    PetscMPIInt vec_local_size = 0;
    PetscScalar *send_multisplitting_data_buffer = NULL;
    PetscScalar *rcv_multisplitting_data_buffer = NULL;
    PetscScalar *temp_multisplitting_data_buffer = NULL;

    Vec local_right_side_vector = NULL;
    Vec mat_mult_vec_result = NULL;

    MPI_Request send_multisplitting_data_request = MPI_REQUEST_NULL;
    // MPI_Request rcv_multisplitting_data_request = MPI_REQUEST_NULL;

    PetscMPIInt send_multisplitting_data_flag = 0;
    PetscMPIInt rcv_multisplitting_data_flag = 0;

    PetscInt *vec_local_idx = NULL;
    PetscInt x_local_size;
    PetscScalar *vector_to_insert_into_S;

    // MPI_Request send_signal_request = MPI_REQUEST_NULL;
    // MPI_Request rcv_signal_request = MPI_REQUEST_NULL;

    // MPI_Status status;

    // Minimization variables

    Mat R = NULL;
    Mat S = NULL;
    PetscInt n_vectors_inserted;
    Vec x_minimized = NULL;
    // Vec x_minimized_prev_iteration = NULL;
    // Vec local_iterates_difference = NULL;

    PetscInt MIN_CONVERGENCE_COUNT = 5;

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

    idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;
    IS is_cols_block_jacobi[njacobi_blocks];
    Mat A_block_jacobi_subMat[njacobi_blocks];
    Vec b_block_jacobi[njacobi_blocks];
    Vec x_block_jacobi[njacobi_blocks];
    VecScatter scatter_jacobi_vec_part_to_merged_vec[njacobi_blocks];
    IS is_merged_vec[njacobi_blocks];
    Mat R_transpose_R = NULL;
    Vec vec_R_transpose_b_block_jacobi = NULL;
    Vec alpha = NULL;
    PetscMPIInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
    PetscMPIInt message_dest = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
    // PetscInt convergence_count = ZERO;
    // PetscMPIInt broadcast_message = NO_MESSAGE;
    // PetscInt inner_solver_iterations;

    PetscInt nbNeigNotLCV = ZERO;
    PetscInt nbIterPreLocalCV = ZERO;
    PetscBool preLocalCV = PETSC_FALSE;
    PetscBool sLocalCV = PETSC_FALSE;
    PetscBool globalCV = PETSC_FALSE;
    PetscInt THRESHOLD_SLCV = MIN_CONVERGENCE_COUNT;
    PetscInt *neighbors = NULL;    /* array containing all the node neighbors */
    PetscInt nbNeighbors = ZERO;   /* number of node neighbors */
    PetscInt *prevIterNumS = NULL; /* array containing previous iteration at witch node "i" notify convergence */
    PetscInt *prevIterNumC = NULL; /* array containing previous iteration at witch node "i" notify convergence CANCELING */
    PetscMPIInt dest_node = -1;
    PetscInt cancelSPartialBuffer;
    MPI_Request cancelSPartialRequest;
    PetscInt sendSPartialBuffer;
    MPI_Request sendSPartialRequest;
    PetscLogDouble time_period_with_globalCV __attribute__((unused)) = 0.0;
    PetscLogDouble globalCV_timer = 0.0;
    PetscLogDouble MAX_TRAVERSAL_TIME __attribute__((unused)) = 0.0; // 13.21 ms
    PetscInt MAX_NEIGHBORS = ONE;

    // FIXME: latency between all local root node

    PetscCall(PetscBarrier(NULL));
    if (proc_local_rank == 0)
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Starting latency checking .... \n"));
        PetscMPIInt proc_rank_node_1 = 0;
        PetscMPIInt proc_rank_node_2 = nprocs_per_jacobi_block;
        PetscCall(comm_sync_measure_latency_between_two_nodes(proc_rank_node_1, proc_rank_node_2, proc_global_rank, &MAX_TRAVERSAL_TIME));
    }

    PetscCallMPI(MPI_Bcast(&MAX_TRAVERSAL_TIME, 1, MPI_DOUBLE, ROOT_NODE, PETSC_COMM_WORLD));

    PetscCall(PetscPrintf(PETSC_COMM_SELF, "MAX_TRAVERSAL_TIME = %e \n", MAX_TRAVERSAL_TIME));

    if (proc_local_rank == 0)
    {
        PetscCall(PetscMalloc1(MAX_NEIGHBORS, &neighbors));
        PetscCall(PetscArrayfill_custom(neighbors, -11, MAX_NEIGHBORS));

        PetscCall(PetscMalloc1(MAX_NEIGHBORS, &prevIterNumS));
        PetscCall(PetscArrayfill_custom(prevIterNumS, -1, MAX_NEIGHBORS));

        PetscCall(PetscMalloc1(MAX_NEIGHBORS, &prevIterNumC));
        PetscCall(PetscArrayfill_custom(prevIterNumC, 0, MAX_NEIGHBORS));

        PetscCall(build_spanning_tree(rank_jacobi_block, neighbors, &nbNeighbors, proc_local_rank, nprocs_per_jacobi_block));
        nbNeigNotLCV = nbNeighbors;
        nbIterPreLocalCV = 0;
        preLocalCV = PETSC_FALSE;
        sLocalCV = PETSC_FALSE;
    }
    globalCV = PETSC_FALSE;

    PetscCall(PetscBarrier(NULL));

    for (PetscInt i = 0; i < njacobi_blocks; i++)
    {
        PetscCall(create_vector(comm_jacobi_block, &b_block_jacobi[i], jacobi_block_size, VECMPI));
        PetscCall(create_vector(comm_jacobi_block, &x_block_jacobi[i], jacobi_block_size, VECMPI));
    }

    PetscCall(create_matrix_sparse(comm_jacobi_block, &A_block_jacobi, n_mesh_points / njacobi_blocks, n_mesh_points, MATMPIAIJ, 5, 5));
    PetscCall(poisson2DMatrix(&A_block_jacobi, n_mesh_lines, n_mesh_columns, rank_jacobi_block, njacobi_blocks));

    PetscCall(create_matrix_dense(comm_jacobi_block, &R, jacobi_block_size, s, MATMPIDENSE));
    PetscCall(MatZeroEntries(R));
    PetscCall(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));

    PetscCall(create_matrix_dense(comm_jacobi_block, &S, n_mesh_points, s, MATMPIDENSE));

    PetscCall(create_matrix_dense(comm_jacobi_block, &R_transpose_R, s, s, MATMPIDENSE));

    PetscCall(create_vector(comm_jacobi_block, &vec_R_transpose_b_block_jacobi, s, VECMPI));
    PetscCall(create_vector(comm_jacobi_block, &alpha, s, VECMPI));
    PetscCall(create_vector(comm_jacobi_block, &x, n_mesh_points, VECMPI));

    PetscCall(VecDuplicate(x, &b));
    PetscCall(VecDuplicate(x, &u));
    PetscCall(VecSet(u, ONE));

    PetscCall(divideSubDomainIntoBlockMatrices(comm_jacobi_block, A_block_jacobi, A_block_jacobi_subMat, is_cols_block_jacobi, rank_jacobi_block, njacobi_blocks, proc_local_rank, nprocs_per_jacobi_block));

    // creation of a scatter context to manage data transfert between complete b or x , and their part x_block_jacobi[..] and b_block_jacobi[...]

    PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, ZERO, ONE, &is_jacobi_vec_parts));

    for (PetscMPIInt i = 0; i < njacobi_blocks; i++)
    {
        PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, (i * (jacobi_block_size)), ONE, &is_merged_vec[i]));
        PetscCall(VecScatterCreate(b_block_jacobi[i], is_jacobi_vec_parts, b, is_merged_vec[i], &scatter_jacobi_vec_part_to_merged_vec[i]));
    }

    PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, &b, b_block_jacobi, u, rank_jacobi_block, jacobi_block_size, nprocs_per_jacobi_block, proc_local_rank));

    PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, INNER_KSP_PREFIX, INNER_PC_PREFIX));
    PetscCall(initializeKSP(comm_jacobi_block, &outer_ksp, NULL, rank_jacobi_block, PETSC_TRUE, OUTER_KSP_PREFIX, OUTER_PC_PREFIX));

    PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
    PetscCall(PetscMalloc1(vec_local_size, &send_multisplitting_data_buffer));
    PetscCall(PetscMalloc1(vec_local_size, &rcv_multisplitting_data_buffer));

    PetscCall(create_vector(comm_jacobi_block, &x_minimized, n_mesh_points, VECMPI));
    PetscCall(VecSet(x_minimized, ZERO));

    PetscCall(VecGetLocalSize(x, &x_local_size));
    PetscCall(PetscMalloc1(x_local_size, &vec_local_idx));
    for (PetscMPIInt i = 0; i < (x_local_size); i++)
    {
        vec_local_idx[i] = (proc_local_rank * x_local_size) + i;
    }
    PetscCall(PetscMalloc1(x_local_size, &vector_to_insert_into_S));

    PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
    PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

    PetscInt inner_solver_iterations __attribute__((unused)) = ZERO;
    // PetscInt message_received __attribute__((unused)) = 0;
    // PetscInt inner_solver_iterations_count __attribute__((unused)) = 0;
    char *send_pack_buffer = NULL;
    char *rcv_pack_buffer = NULL;
    PetscMPIInt other_block_current_iteration = -1;
    PetscMPIInt current_number_of_iterations = -1;

    PetscLogEvent USER_EVENT;
    PetscLogEventRegister("outer_solve", 0, &USER_EVENT);

    Vec local_residual = NULL;
    PetscCall(VecDuplicate(x_block_jacobi[idx_non_current_block], &local_residual));
    PetscScalar local_norm = PETSC_MAX_REAL;
    PetscScalar local_norm_0 = 0.0;
    PetscCall(updateLocalRHS(A_block_jacobi_subMat[idx_non_current_block], x_block_jacobi[idx_non_current_block], b_block_jacobi[rank_jacobi_block], local_right_side_vector));
    PetscCall(MatResidual(A_block_jacobi_subMat[rank_jacobi_block], local_right_side_vector, x_block_jacobi[rank_jacobi_block], local_residual)); // r_i = b_i - (A_i * x_i)
    PetscCall(VecNorm(local_residual, NORM_2, &local_norm_0));
    PetscCall(PetscPrintf(comm_jacobi_block, "rank block %d local b norm %e \n", rank_jacobi_block, local_norm_0));

    PetscCall(PetscBarrier(NULL));
    PetscCall(PetscTime(&globalCV_timer));
    double start_time, end_time;
    start_time = MPI_Wtime();

    do
    {

        n_vectors_inserted = 0;

        while (n_vectors_inserted < s)
        {

            PetscCall(comm_async_probe_and_receive(x_block_jacobi, rcv_multisplitting_data_buffer, vec_local_size, rcv_multisplitting_data_flag, message_source, idx_non_current_block, NULL, &other_block_current_iteration, &rcv_pack_buffer));

            PetscCall(updateLocalRHS(A_block_jacobi_subMat[idx_non_current_block], x_block_jacobi[idx_non_current_block], b_block_jacobi[rank_jacobi_block], local_right_side_vector));

            PetscCall(inner_solver(comm_jacobi_block, inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, local_right_side_vector, rank_jacobi_block, &inner_solver_iterations, number_of_iterations));

            current_number_of_iterations = number_of_iterations;
            PetscCall(comm_async_test_and_send(x_block_jacobi, send_multisplitting_data_buffer, temp_multisplitting_data_buffer, &send_multisplitting_data_request, vec_local_size, send_multisplitting_data_flag, message_dest, rank_jacobi_block, &current_number_of_iterations, &send_pack_buffer));

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

        PetscCall(MatMatMult(A_block_jacobi, S, MAT_REUSE_MATRIX, PETSC_DETERMINE, &R));

        PetscLogEventBegin(USER_EVENT, 0, 0, 0, 0);
        PetscCall(outer_solver_norm_equation_modify(comm_jacobi_block, outer_ksp, x_minimized, R, S, alpha, b_block_jacobi[rank_jacobi_block], rank_jacobi_block, number_of_iterations, message_dest, message_source));
        PetscLogEventEnd(USER_EVENT, 0, 0, 0, 0);

        PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));

        PetscCall(comm_async_test_and_send(x_block_jacobi, send_multisplitting_data_buffer, temp_multisplitting_data_buffer, &send_multisplitting_data_request, vec_local_size, send_multisplitting_data_flag, message_dest, rank_jacobi_block, &current_number_of_iterations, &send_pack_buffer));

        PetscCall(MatResidual(A_block_jacobi, b_block_jacobi[rank_jacobi_block], x_minimized, local_residual));
        PetscCall(VecNorm(local_residual, NORM_2, &local_norm));

        PetscCall(PetscPrintf(comm_jacobi_block, "Local norm_2 block rank %d = %e \n", rank_jacobi_block, local_norm));

        if (proc_local_rank == 0) // XXX: ONLY root node from each block check for convergence
        {

            if (local_norm <= PetscMax(absolute_tolerance, relative_tolerance * local_norm_0))
            {
                preLocalCV = PETSC_TRUE;
            }
            else
            {
                preLocalCV = PETSC_FALSE;
            }

            cancelSPartialBuffer = number_of_iterations;
            sendSPartialBuffer = number_of_iterations;

            PetscCall(comm_async_convDetection(rank_jacobi_block, nbNeighbors, &nbNeigNotLCV, neighbors, prevIterNumS, prevIterNumC, &nbIterPreLocalCV, &preLocalCV, &sLocalCV, &globalCV, &dest_node, THRESHOLD_SLCV, number_of_iterations, &cancelSPartialBuffer, &cancelSPartialRequest, &sendSPartialBuffer, &sendSPartialRequest));

            PetscCall(comm_async_recvSPartialCV(rank_jacobi_block, &nbNeigNotLCV, prevIterNumS, prevIterNumC));

            PetscCall(comm_async_recvCancelSPartialCV(rank_jacobi_block, &nbNeigNotLCV, nbNeighbors, prevIterNumS, prevIterNumC, &globalCV));

            PetscCall(comm_async_recvGlobalCV(rank_jacobi_block, &globalCV));
        }

        PetscCallMPI(MPI_Bcast(&globalCV, 1, MPIU_BOOL, LOCAL_ROOT_NODE, comm_jacobi_block));

        if (globalCV == PETSC_FALSE)
        {
            PetscCall(PetscTime(&globalCV_timer));
        }
        else
        {
            time_period_with_globalCV = globalCV_timer;
            PetscCall(PetscTimeSubtract(&time_period_with_globalCV));
            time_period_with_globalCV = PetscAbs(time_period_with_globalCV);
        }

        number_of_iterations = number_of_iterations + 1;

        PetscCall(comm_async_probe_and_receive(x_block_jacobi, rcv_multisplitting_data_buffer, vec_local_size, rcv_multisplitting_data_flag, message_source, idx_non_current_block, NULL, &other_block_current_iteration, &rcv_pack_buffer));
        PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x_minimized, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x_minimized, INSERT_VALUES, SCATTER_FORWARD));

    } while ((time_period_with_globalCV * 1000.0) <= MAX_TRAVERSAL_TIME);

    MPI_Request requests[nbNeighbors];
    PetscCall(comm_async_sendGlobalCV(rank_jacobi_block, nbNeighbors, neighbors, &globalCV, requests));
    PetscCallMPI(MPI_Waitall(nbNeighbors, requests, MPI_STATUSES_IGNORE));

    PetscCall(PetscPrintf(comm_jacobi_block, "Rank %d: PROGRAMME TERMINE\n", rank_jacobi_block));

    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    end_time = MPI_Wtime();
    PetscCall(printElapsedTime(start_time, end_time));
    PetscCall(printTotalNumberOfIterations_2(comm_jacobi_block, rank_jacobi_block, number_of_iterations, s));

    PetscCall(comm_sync_send_and_receive_final(x_block_jacobi, vec_local_size, message_dest, message_source, rank_jacobi_block, idx_non_current_block));

    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

    PetscScalar direct_residual_norm;
    PetscCall(computeFinalResidualNorm(A_block_jacobi, x, b_block_jacobi, rank_jacobi_block, proc_global_rank, &direct_residual_norm));

    PetscCall(printFinalResidualNorm(direct_residual_norm));

    PetscScalar error;
    PetscCall(computeError(x, u, &error));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Erreur : %e \n", error));

    for (PetscInt i = 0; i < njacobi_blocks; i++)
    {
        PetscCall(ISDestroy(&is_cols_block_jacobi[i]));
        PetscCall(VecDestroy(&x_block_jacobi[i]));
        PetscCall(VecDestroy(&b_block_jacobi[i]));
        PetscCall(MatDestroy(&A_block_jacobi_subMat[i]));
        PetscCall(VecScatterDestroy(&scatter_jacobi_vec_part_to_merged_vec[i]));
        PetscCall(ISDestroy(&is_merged_vec[i]));
    }

    PetscFree(vec_local_idx);
    PetscFree(vector_to_insert_into_S);
    PetscCall(ISDestroy(&is_jacobi_vec_parts));
    PetscCall(VecDestroy(&local_right_side_vector));
    PetscCall(VecDestroy(&mat_mult_vec_result));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&x_minimized));
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

    // Maybe delete the rest of this code, not necessary
    // PetscInt message = ZERO;
    // PetscInt count;

    // do
    // {
    //     MPI_Datatype data_type = MPIU_INT;
    //     PetscCallMPI(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &message, &status));
    //     if (message)
    //     {
    //         if (status.MPI_TAG == (TAG_MULTISPLITTING_DATA + idx_non_current_block))
    //         {
    //             data_type = MPIU_SCALAR;
    //             PetscCall(MPI_Get_count(&status, data_type, &count));
    //             PetscScalar *buffer;
    //             PetscCall(PetscMalloc1(count, &buffer));
    //             PetscCallMPI(MPI_Recv(buffer, count, data_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    //             PetscCall(PetscFree(buffer));
    //         }
    //         else
    //         {
    //             data_type = MPIU_INT;
    //             PetscCall(MPI_Get_count(&status, data_type, &count));
    //             PetscInt *buffer;
    //             PetscCall(PetscMalloc1(count, &buffer));
    //             PetscCallMPI(MPI_Recv(buffer, count, data_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    //             PetscCall(PetscFree(buffer));
    //         }
    //     }
    // } while (message);

    // PetscCallMPI(MPI_Wait(&send_multisplitting_data_request, MPI_STATUS_IGNORE));
    // PetscCallMPI(MPI_Wait(&send_signal_request, MPI_STATUS_IGNORE));
    // PetscCall(PetscFree(send_multisplitting_data_buffer));
    // PetscCall(PetscFree(rcv_multisplitting_data_buffer));

    PetscCall(PetscSubcommDestroy(&sub_comm_context));
    PetscCall(PetscCommDestroy(&dcomm));
    PetscCall(PetscFinalize());
    return 0;
}

// #endif

// #ifdef VERSION_1_1

// int main(int argc, char **argv)
// {

//     printf("Code : Rien à faire ....Fin \n");

//     return 0;
// }

// #endif

// PetscScalar b_norm;
// PetscCall(VecNorm(b, NORM_2, &b_norm));
// PetscPrintf(MPI_COMM_SELF, "rank block %d b norm %e \n", rank_jacobi_block, b_norm);
// PetscScalar norm;
// PetscScalar global_norm_0 = 0.0;
// PetscCall(computeFinalResidualNorm(A_block_jacobi, x_minimized, b_block_jacobi, rank_jacobi_block, proc_local_rank, &global_norm_0));

// PetscCall(VecWAXPY(local_iterates_difference, -1.0, x_minimized_prev_iteration, x_minimized));
// PetscCall(VecNorm(local_iterates_difference, NORM_INFINITY, &local_iterates_difference_norm_inf));
// PetscCall(VecNorm(x_minimized, NORM_INFINITY, &current_iterate_norm_inf));
// PetscCall(printResidualNorm(comm_jacobi_block, rank_jacobi_block, local_iterates_difference_norm_inf, number_of_iterations));
// if (local_iterates_difference_norm_inf <= PetscMax(absolute_tolerance, relative_tolerance * current_iterate_norm_inf))
// {
//     preLocalCV = PETSC_TRUE;
// }
// else
// {
//     preLocalCV = PETSC_FALSE;
// }
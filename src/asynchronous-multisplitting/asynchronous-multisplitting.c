#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"
#include "constants.h"
#include "utils.h"
#include "comm.h"
#include "conv_detection.h"

// #ifdef VERSION_1_0

// int main(int argc, char **argv)
// {

//   Mat A_block_jacobi = NULL; // Operator matrix
//   Vec x = NULL;              // approximation solution at iteration (k)
//   Vec b = NULL;              // right hand side vector
//   Vec x_initial_guess = NULL;

//   PetscMPIInt nprocs;
//   PetscMPIInt proc_global_rank;
//   PetscInt n_mesh_lines = 4;
//   PetscInt n_mesh_columns = 4;
//   PetscInt njacobi_blocks;
//   PetscMPIInt rank_jacobi_block;
//   PetscInt proc_local_rank;
//   PetscInt n_mesh_points;
//   PetscInt jacobi_block_size;
//   PetscInt s;
//   PetscScalar relative_tolerance = 1e-5;
//   PetscMPIInt nprocs_per_jacobi_block = 1;

//   Vec local_right_side_vector = NULL;
//   Vec mat_mult_vec_result = NULL;

//   PetscInt MIN_CONVERGENCE_COUNT = 5;
//   PetscFunctionBeginUser;
//   PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
//   PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
//   PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

//   // Getting applications arguments
//   PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &n_mesh_lines, NULL));
//   PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n_mesh_columns, NULL));
//   PetscCall(PetscOptionsGetInt(NULL, NULL, "-s", &s, NULL));
//   PetscCall(PetscOptionsGetInt(NULL, NULL, "-min_convergence_count", &MIN_CONVERGENCE_COUNT, NULL));
//   PetscCall(PetscOptionsGetInt(NULL, NULL, "-npb", &nprocs_per_jacobi_block, NULL));
//   PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &relative_tolerance, NULL));

//   // PetscPrintf(PETSC_COMM_WORLD, " =====> Total number of processes: %d \n =====>s : %d\n =====>nprocessor_per_jacobi_block : %d \n ====> Grid lines: %d \n ====> Grid columns : %d ====> Relative tolerance : %f\n", nprocs, s, nprocs_per_jacobi_block, n_mesh_lines, n_mesh_columns, relative_tolerance);

//   PetscCall(computeDimensionRelatedVariables(nprocs, nprocs_per_jacobi_block, proc_global_rank, n_mesh_lines, n_mesh_columns, &njacobi_blocks, &rank_jacobi_block, &proc_local_rank, &n_mesh_points, &jacobi_block_size));

//   PetscAssert((n_mesh_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");

//   // Creating the sub communicator for each jacobi block
//   PetscSubcomm sub_comm_context = NULL;
//   MPI_Comm dcomm;
//   PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD, &dcomm, NULL));

//   PetscCall(PetscSubcommCreate(dcomm, &sub_comm_context));
//   PetscCall(PetscSubcommSetNumber(sub_comm_context, njacobi_blocks));
//   PetscCall(PetscSubcommSetType(sub_comm_context, PETSC_SUBCOMM_CONTIGUOUS));
//   MPI_Comm comm_jacobi_block = PetscSubcommChild(sub_comm_context);

//   KSP inner_ksp = NULL;
//   PetscInt number_of_iterations = ZERO;
//   PetscMPIInt idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;
//   PetscScalar approximation_residual_infinity_norm = PETSC_MAX_REAL;
//   PetscMPIInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
//   PetscMPIInt message_dest = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
//   Mat A_block_jacobi_subMat[njacobi_blocks];
//   IS is_cols_block_jacobi[njacobi_blocks];
//   Vec b_block_jacobi[njacobi_blocks];
//   Vec x_block_jacobi[njacobi_blocks];
//   PetscMPIInt send_signal = NO_SIGNAL;
//   PetscMPIInt rcv_signal = NO_SIGNAL;
//   VecScatter scatter_jacobi_vec_part_to_merged_vec[njacobi_blocks];
//   IS is_jacobi_vec_parts;
//   IS is_merged_vec[njacobi_blocks];
//   PetscMPIInt vec_local_size = ZERO;
//   PetscScalar *send_buffer = NULL;
//   PetscScalar *rcv_buffer = NULL;
//   PetscScalar *temp_buffer = NULL;
//   MPI_Request send_data_request = MPI_REQUEST_NULL;
//   MPI_Request send_signal_request = MPI_REQUEST_NULL;
//   MPI_Status status;
//   PetscMPIInt broadcast_message = NO_MESSAGE;
//   PetscMPIInt send_data_flag = ZERO;
//   PetscMPIInt rcv_data_flag = ZERO;
//   // PetscInt inner_solver_iterations = ZERO;
//   PetscInt convergence_count = ZERO;
//   Vec approximation_residual;
//   Vec x_block_jacobi_previous_iteration = NULL;

//   PetscCall(VecCreate(comm_jacobi_block, &x));
//   PetscCall(VecSetSizes(x, PETSC_DECIDE, n_mesh_points));
//   PetscCall(VecSetType(x, VECMPI));
//   PetscCall(VecSetFromOptions(x));
//   PetscCall(VecSetUp(x));

//   PetscCall(VecDuplicate(x, &b));

//   PetscCall(VecDuplicate(x, &x_initial_guess));
//   PetscScalar initial_scalar_value = 1.0;
//   PetscCall(VecSet(x_initial_guess, initial_scalar_value));

//   PetscCall(create_matrix_sparse(comm_jacobi_block, &A_block_jacobi, n_mesh_points / njacobi_blocks, n_mesh_points, MATMPIAIJ, 5, 5));

//   PetscCall(poisson2DMatrix(&A_block_jacobi, n_mesh_lines, n_mesh_columns, rank_jacobi_block, njacobi_blocks));

//   PetscCall(divideSubDomainIntoBlockMatrices(comm_jacobi_block, A_block_jacobi, A_block_jacobi_subMat, is_cols_block_jacobi, rank_jacobi_block, njacobi_blocks, proc_local_rank, nprocs_per_jacobi_block));

//   for (PetscInt i = 0; i < njacobi_blocks; i++)
//   {
//     PetscCall(create_vector(comm_jacobi_block, &b_block_jacobi[i], jacobi_block_size, VECMPI));
//     PetscCall(create_vector(comm_jacobi_block, &x_block_jacobi[i], jacobi_block_size, VECMPI));
//   }

//   PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, ZERO, ONE, &is_jacobi_vec_parts));
//   for (PetscInt i = 0; i < njacobi_blocks; i++)
//   {
//     PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, (i * (jacobi_block_size)), ONE, &is_merged_vec[i]));
//     PetscCall(VecScatterCreate(b_block_jacobi[i], is_jacobi_vec_parts, b, is_merged_vec[i], &scatter_jacobi_vec_part_to_merged_vec[i]));
//   }

//   PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, &b, b_block_jacobi, x_initial_guess, rank_jacobi_block, jacobi_block_size, nprocs_per_jacobi_block, proc_local_rank));

//   PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, INNER_KSP_PREFIX, INNER_PC_PREFIX));

//   PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
//   PetscCall(PetscMalloc1(vec_local_size, &send_buffer));
//   PetscCall(PetscMalloc1(vec_local_size, &rcv_buffer));

//   PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &approximation_residual));

//   PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &x_block_jacobi_previous_iteration));
//   PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
//   PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

//   PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
//   double start_time, end_time;
//   start_time = MPI_Wtime();

//   do
//   {

//     PetscCall(comm_async_probe_and_receive(x_block_jacobi, rcv_buffer, vec_local_size, rcv_data_flag, message_source, idx_non_current_block, NULL));

//     PetscCall(updateLocalRHS(local_right_side_vector, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, mat_mult_vec_result, rank_jacobi_block));
//     PetscCall(inner_solver(comm_jacobi_block, inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, local_right_side_vector, rank_jacobi_block, NULL, number_of_iterations));

//     PetscCall(comm_async_test_and_send(x_block_jacobi, send_buffer, temp_buffer, &send_data_request, vec_local_size, send_data_flag, message_dest, rank_jacobi_block));

//     PetscCall(comm_async_probe_and_receive(x_block_jacobi, rcv_buffer, vec_local_size, rcv_data_flag, message_source, idx_non_current_block, NULL));

//     PetscCall(VecWAXPY(approximation_residual, -1.0, x_block_jacobi_previous_iteration, x_block_jacobi[rank_jacobi_block]));
//     PetscCall(VecNorm(approximation_residual, NORM_INFINITY, &approximation_residual_infinity_norm));
//     PetscCall(VecCopy(x_block_jacobi[rank_jacobi_block], x_block_jacobi_previous_iteration));
//     PetscCall(printResidualNorm(comm_jacobi_block, rank_jacobi_block, approximation_residual_infinity_norm, number_of_iterations));

//     if (PetscApproximateLTE(approximation_residual_infinity_norm, relative_tolerance))
//       convergence_count++;
//     else
//       convergence_count = ZERO;

//     PetscCall(comm_async_convergence_detection(&broadcast_message, convergence_count, MIN_CONVERGENCE_COUNT, &send_signal, &send_signal_request, &rcv_signal, message_dest, message_source, rank_jacobi_block, idx_non_current_block, proc_local_rank));

//     PetscCallMPI(MPI_Bcast(&broadcast_message, ONE, MPIU_INT, proc_local_rank, comm_jacobi_block));

//     number_of_iterations = number_of_iterations + 1;

//   } while (broadcast_message != TERMINATE_SIGNAL);

//   PetscCall(PetscPrintf(comm_jacobi_block, "Rank %d: PROGRAMME TERMINE\n", rank_jacobi_block));
//   PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
//   end_time = MPI_Wtime();
//   PetscCall(printElapsedTime(start_time, end_time));
//   PetscCall(printTotalNumberOfIterations(comm_jacobi_block, rank_jacobi_block, number_of_iterations));

//   PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));

//   PetscCall(comm_sync_send_and_receive_final(x_block_jacobi, vec_local_size, message_dest, message_source, rank_jacobi_block, idx_non_current_block));

//   PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
//   PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
//   PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
//   PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

//   PetscScalar direct_residual_norm;
//   PetscCall(computeFinalResidualNorm(A_block_jacobi, &x, b_block_jacobi, rank_jacobi_block, proc_global_rank, &direct_residual_norm));
//   PetscCall(printFinalResidualNorm(direct_residual_norm));

//   // END OF PROGRAM  - FREE MEMORY

//   PetscCall(ISDestroy(&is_jacobi_vec_parts));
//   for (PetscInt i = 0; i < njacobi_blocks; i++)
//   {
//     PetscCall(ISDestroy(&is_merged_vec[i]));
//     PetscCall(ISDestroy(&is_cols_block_jacobi[i]));
//     PetscCall(VecDestroy(&x_block_jacobi[i]));
//     PetscCall(VecDestroy(&b_block_jacobi[i]));
//     PetscCall(MatDestroy(&A_block_jacobi_subMat[i]));
//     PetscCall(VecScatterDestroy(&scatter_jacobi_vec_part_to_merged_vec[i]));
//   }

//   PetscCall(VecDestroy(&x_block_jacobi_previous_iteration));
//   PetscCall(VecDestroy(&approximation_residual));
//   PetscCall(VecDestroy(&local_right_side_vector));
//   PetscCall(VecDestroy(&mat_mult_vec_result));
//   PetscCall(VecDestroy(&x));
//   PetscCall(VecDestroy(&b));
//   PetscCall(VecDestroy(&x_initial_guess));
//   PetscCall(MatDestroy(&A_block_jacobi));
//   PetscCall(KSPDestroy(&inner_ksp));

//   // Discard any pending message
//   PetscInt count;
//   PetscInt message = NO_MESSAGE;

//   do
//   {
//     MPI_Datatype data_type = MPIU_INT;
//     PetscCallMPI(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &message, &status));
//     if (message)
//     {
//       if (status.MPI_TAG == (TAG_MULTISPLITTING_DATA + idx_non_current_block))
//       {
//         data_type = MPIU_SCALAR;
//         PetscCall(MPI_Get_count(&status, data_type, &count));
//         PetscScalar *buffer;
//         PetscCall(PetscMalloc1(count, &buffer));
//         PetscCallMPI(MPI_Recv(buffer, count, data_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
//         PetscCall(PetscFree(buffer));
//       }
//       else
//       {
//         data_type = MPIU_INT;
//         PetscCall(MPI_Get_count(&status, data_type, &count));
//         PetscInt *buffer;
//         PetscCall(PetscMalloc1(count, &buffer));
//         PetscCallMPI(MPI_Recv(buffer, count, data_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
//         PetscCall(PetscFree(buffer));
//       }
//     }
//   } while (message);

//   PetscCallMPI(MPI_Wait(&send_data_request, MPI_STATUS_IGNORE));
//   PetscCallMPI(MPI_Wait(&send_signal_request, MPI_STATUS_IGNORE));
//   PetscCall(PetscFree(send_buffer));
//   PetscCall(PetscFree(rcv_buffer));

//   PetscCall(PetscSubcommDestroy(&sub_comm_context));
//   PetscCall(PetscCommDestroy(&dcomm));
//   PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
//   PetscCall(PetscFinalize());

//   return 0;
// }
// #endif

// #ifdef VERSION_1_1
// int main(int argc, char **argv)
// {

//   Mat A_block_jacobi = NULL; // Operator matrix
//   Vec x = NULL;              // approximation solution at iteration (k)
//   Vec b = NULL;              // right hand side vector
//   Vec x_initial_guess = NULL;

//   PetscMPIInt nprocs;
//   PetscMPIInt proc_global_rank;
//   PetscInt n_mesh_lines = 4;
//   PetscInt n_mesh_columns = 4;
//   PetscInt njacobi_blocks;
//   PetscMPIInt rank_jacobi_block;
//   PetscInt proc_local_rank;
//   PetscInt n_mesh_points;
//   PetscInt jacobi_block_size;
//   PetscInt s;
//   PetscScalar relative_tolerance = 1e-5;
//   PetscMPIInt nprocs_per_jacobi_block = 1;

//   Vec local_right_side_vector = NULL;
//   Vec mat_mult_vec_result = NULL;

//   PetscInt MIN_CONVERGENCE_COUNT = 5;
//   PetscFunctionBeginUser;
//   PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

//   PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
//   PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

//   // Getting applications arguments
//   PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &n_mesh_lines, NULL));
//   PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n_mesh_columns, NULL));
//   PetscCall(PetscOptionsGetInt(NULL, NULL, "-s", &s, NULL));
//   PetscCall(PetscOptionsGetInt(NULL, NULL, "-min_convergence_count", &MIN_CONVERGENCE_COUNT, NULL));
//   PetscCall(PetscOptionsGetInt(NULL, NULL, "-npb", &nprocs_per_jacobi_block, NULL));
//   PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &relative_tolerance, NULL));

//   // PetscPrintf(PETSC_COMM_WORLD, " =====> Total number of processes: %d \n =====>s : %d\n =====>nprocessor_per_jacobi_block : %d \n ====> Grid lines: %d \n ====> Grid columns : %d ====> Relative tolerance : %f\n", nprocs, s, nprocs_per_jacobi_block, n_mesh_lines, n_mesh_columns, relative_tolerance);

//   PetscCall(computeDimensionRelatedVariables(nprocs, nprocs_per_jacobi_block, proc_global_rank, n_mesh_lines, n_mesh_columns, &njacobi_blocks, &rank_jacobi_block, &proc_local_rank, &n_mesh_points, &jacobi_block_size));

//   PetscAssert((n_mesh_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");

//   // Creating the sub communicator for each jacobi block
//   PetscSubcomm sub_comm_context = NULL;
//   MPI_Comm dcomm;
//   PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD, &dcomm, NULL));

//   PetscCall(PetscSubcommCreate(dcomm, &sub_comm_context));
//   PetscCall(PetscSubcommSetNumber(sub_comm_context, njacobi_blocks));
//   PetscCall(PetscSubcommSetType(sub_comm_context, PETSC_SUBCOMM_CONTIGUOUS));
//   MPI_Comm comm_jacobi_block = PetscSubcommChild(sub_comm_context);

//   KSP inner_ksp = NULL;
//   PetscInt number_of_iterations = ZERO;
//   PetscMPIInt idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;
//   PetscScalar approximation_residual_infinity_norm = PETSC_MAX_REAL;
//   PetscMPIInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
//   PetscMPIInt message_dest = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
//   Mat A_block_jacobi_subMat[njacobi_blocks];
//   IS is_cols_block_jacobi[njacobi_blocks];
//   Vec b_block_jacobi[njacobi_blocks];
//   Vec x_block_jacobi[njacobi_blocks];
//   PetscMPIInt send_signal = NO_SIGNAL;
//   PetscMPIInt rcv_signal = NO_SIGNAL;
//   VecScatter scatter_jacobi_vec_part_to_merged_vec[njacobi_blocks];
//   IS is_jacobi_vec_parts;
//   IS is_merged_vec[njacobi_blocks];
//   PetscMPIInt vec_local_size = ZERO;
//   PetscScalar *send_buffer = NULL;
//   PetscScalar *rcv_buffer = NULL;
//   PetscScalar *temp_buffer = NULL;
//   MPI_Request send_data_request = MPI_REQUEST_NULL;
//   MPI_Request send_signal_request = MPI_REQUEST_NULL;
//   MPI_Status status;
//   PetscMPIInt broadcast_message = NO_MESSAGE;
//   PetscMPIInt send_data_flag = ZERO;
//   PetscMPIInt rcv_data_flag = ZERO;
//   // PetscInt inner_solver_iterations = ZERO;
//   PetscInt convergence_count = ZERO;
//   Vec approximation_residual;
//   Vec x_block_jacobi_previous_iteration = NULL;

//   PetscCall(VecCreate(comm_jacobi_block, &x));
//   PetscCall(VecSetSizes(x, PETSC_DECIDE, n_mesh_points));
//   PetscCall(VecSetType(x, VECMPI));
//   PetscCall(VecSetFromOptions(x));
//   PetscCall(VecSetUp(x));

//   PetscCall(VecDuplicate(x, &b));

//   PetscCall(VecDuplicate(x, &x_initial_guess));
//   PetscScalar initial_scalar_value = 1.0;
//   PetscCall(VecSet(x_initial_guess, initial_scalar_value));

//   PetscCall(create_matrix_sparse(comm_jacobi_block, &A_block_jacobi, n_mesh_points / njacobi_blocks, n_mesh_points, MATMPIAIJ, 5, 5));

//   PetscCall(poisson2DMatrix(&A_block_jacobi, n_mesh_lines, n_mesh_columns, rank_jacobi_block, njacobi_blocks));

//   PetscCall(divideSubDomainIntoBlockMatrices(comm_jacobi_block, A_block_jacobi, A_block_jacobi_subMat, is_cols_block_jacobi, rank_jacobi_block, njacobi_blocks, proc_local_rank, nprocs_per_jacobi_block));

//   for (PetscInt i = 0; i < njacobi_blocks; i++)
//   {
//     PetscCall(create_vector(comm_jacobi_block, &b_block_jacobi[i], jacobi_block_size, VECMPI));
//     PetscCall(create_vector(comm_jacobi_block, &x_block_jacobi[i], jacobi_block_size, VECMPI));
//   }

//   PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, ZERO, ONE, &is_jacobi_vec_parts));
//   for (PetscInt i = 0; i < njacobi_blocks; i++)
//   {
//     PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, (i * (jacobi_block_size)), ONE, &is_merged_vec[i]));
//     PetscCall(VecScatterCreate(b_block_jacobi[i], is_jacobi_vec_parts, b, is_merged_vec[i], &scatter_jacobi_vec_part_to_merged_vec[i]));
//   }

//   PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, &b, b_block_jacobi, x_initial_guess, rank_jacobi_block, jacobi_block_size, nprocs_per_jacobi_block, proc_local_rank));

//   PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, INNER_KSP_PREFIX, INNER_PC_PREFIX));

//   PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
//   PetscCall(PetscMalloc1(vec_local_size, &send_buffer));
//   PetscCall(PetscMalloc1(vec_local_size, &rcv_buffer));

//   PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &approximation_residual));

//   PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &x_block_jacobi_previous_iteration));
//   PetscCall(VecCopy(x_block_jacobi[rank_jacobi_block], x_block_jacobi_previous_iteration));
//   PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
//   PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

//   PetscScalar approximation_residual_infinity_norm_iter_zero __attribute__((unused)) = PETSC_MAX_REAL;
//   PetscInt inner_solver_iterations __attribute__((unused)) = ZERO;
//   PetscInt message_received __attribute__((unused)) = 0;
//   PetscInt last_message_received_iter_number __attribute__((unused)) = 0;
//   PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
//   double start_time, end_time;
//   start_time = MPI_Wtime();

//   do
//   {

//     message_received = 0;
//     inner_solver_iterations = 0;

//     PetscCall(comm_async_probe_and_receive(x_block_jacobi, rcv_buffer, vec_local_size, rcv_data_flag, message_source, idx_non_current_block, &message_received));

//     PetscCall(updateLocalRHS(local_right_side_vector, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, mat_mult_vec_result, rank_jacobi_block));
//     PetscCall(inner_solver(comm_jacobi_block, inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, local_right_side_vector, rank_jacobi_block, &inner_solver_iterations, number_of_iterations));

//     if (inner_solver_iterations > 0)
//     {
//       PetscCall(comm_async_test_and_send(x_block_jacobi, send_buffer, temp_buffer, &send_data_request, vec_local_size, send_data_flag, message_dest, rank_jacobi_block));
//     }

//     PetscCall(comm_async_probe_and_receive(x_block_jacobi, rcv_buffer, vec_local_size, rcv_data_flag, message_source, idx_non_current_block, &message_received));

//     PetscCall(VecWAXPY(approximation_residual, -1.0, x_block_jacobi_previous_iteration, x_block_jacobi[rank_jacobi_block]));
//     PetscCall(VecNorm(approximation_residual, NORM_INFINITY, &approximation_residual_infinity_norm));
//     PetscCall(VecCopy(x_block_jacobi[rank_jacobi_block], x_block_jacobi_previous_iteration));

//     PetscCall(printResidualNorm(comm_jacobi_block, rank_jacobi_block, approximation_residual_infinity_norm, number_of_iterations));

//     if (message_received && inner_solver_iterations > 0)
//     {
//       if (PetscApproximateLTE(approximation_residual_infinity_norm, relative_tolerance))
//         convergence_count++;
//       else
//         convergence_count = ZERO;

//       PetscCall(PetscPrintf(comm_jacobi_block, "Rank %d: CONVERGENCE COUNT %d \n", rank_jacobi_block, convergence_count));
//     }

//     PetscCall(comm_async_convergence_detection(&broadcast_message, convergence_count, MIN_CONVERGENCE_COUNT, &send_signal, &send_signal_request, &rcv_signal, message_dest, message_source, rank_jacobi_block, idx_non_current_block, proc_local_rank));

//     PetscCallMPI(MPI_Bcast(&broadcast_message, ONE, MPIU_INT, proc_local_rank, comm_jacobi_block));

//     number_of_iterations = number_of_iterations + 1;

//   } while (broadcast_message != TERMINATE_SIGNAL);

//   PetscCall(PetscPrintf(comm_jacobi_block, "Rank %d: PROGRAMME TERMINE\n", rank_jacobi_block));
//   PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
//   end_time = MPI_Wtime();
//   PetscCall(printElapsedTime(start_time, end_time));
//   PetscCall(printTotalNumberOfIterations(comm_jacobi_block, rank_jacobi_block, number_of_iterations));

//   PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));

//   PetscCall(comm_sync_send_and_receive_final(x_block_jacobi, vec_local_size, message_dest, message_source, rank_jacobi_block, idx_non_current_block));

//   PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
//   PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
//   PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
//   PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

//   PetscScalar direct_residual_norm;
//   PetscCall(computeFinalResidualNorm(A_block_jacobi, &x, b_block_jacobi, rank_jacobi_block, proc_global_rank, &direct_residual_norm));
//   PetscCall(printFinalResidualNorm(direct_residual_norm));

//   // END OF PROGRAM  - FREE MEMORY

//   PetscCall(ISDestroy(&is_jacobi_vec_parts));
//   for (PetscInt i = 0; i < njacobi_blocks; i++)
//   {
//     PetscCall(ISDestroy(&is_merged_vec[i]));
//     PetscCall(ISDestroy(&is_cols_block_jacobi[i]));
//     PetscCall(VecDestroy(&x_block_jacobi[i]));
//     PetscCall(VecDestroy(&b_block_jacobi[i]));
//     PetscCall(MatDestroy(&A_block_jacobi_subMat[i]));
//     PetscCall(VecScatterDestroy(&scatter_jacobi_vec_part_to_merged_vec[i]));
//   }

//   PetscCall(VecDestroy(&x_block_jacobi_previous_iteration));
//   PetscCall(VecDestroy(&approximation_residual));
//   PetscCall(VecDestroy(&local_right_side_vector));
//   PetscCall(VecDestroy(&mat_mult_vec_result));
//   PetscCall(VecDestroy(&x));
//   PetscCall(VecDestroy(&b));
//   PetscCall(VecDestroy(&x_initial_guess));
//   PetscCall(MatDestroy(&A_block_jacobi));
//   PetscCall(KSPDestroy(&inner_ksp));

//   // Discard any pending message
//   PetscInt count;
//   PetscInt message = NO_MESSAGE;

//   do
//   {
//     MPI_Datatype data_type = MPIU_INT;
//     PetscCallMPI(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &message, &status));
//     if (message)
//     {
//       if (status.MPI_TAG == (TAG_MULTISPLITTING_DATA + idx_non_current_block))
//       {
//         data_type = MPIU_SCALAR;
//         PetscCall(MPI_Get_count(&status, data_type, &count));
//         PetscScalar *buffer;
//         PetscCall(PetscMalloc1(count, &buffer));
//         PetscCallMPI(MPI_Recv(buffer, count, data_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
//         PetscCall(PetscFree(buffer));
//       }
//       else
//       {
//         data_type = MPIU_INT;
//         PetscCall(MPI_Get_count(&status, data_type, &count));
//         PetscInt *buffer;
//         PetscCall(PetscMalloc1(count, &buffer));
//         PetscCallMPI(MPI_Recv(buffer, count, data_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
//         PetscCall(PetscFree(buffer));
//       }
//     }
//   } while (message);

//   PetscCallMPI(MPI_Wait(&send_data_request, MPI_STATUS_IGNORE));
//   PetscCallMPI(MPI_Wait(&send_signal_request, MPI_STATUS_IGNORE));
//   PetscCall(PetscFree(send_buffer));
//   PetscCall(PetscFree(rcv_buffer));

//   PetscCall(PetscSubcommDestroy(&sub_comm_context));
//   PetscCall(PetscCommDestroy(&dcomm));
//   PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
//   PetscCall(PetscFinalize());
//   return 0;
// }
// #endif

// #ifdef VERSION_2_0
int main(int argc, char **argv)
{

  Mat A_block_jacobi = NULL; // Operator matrix
  Vec x = NULL;              // approximation solution at iteration (k)
  Vec b = NULL;              // right hand side vector
  Vec x_initial_guess = NULL;

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
  PetscMPIInt nprocs_per_jacobi_block = 1;

  Vec local_right_side_vector = NULL;
  Vec mat_mult_vec_result = NULL;

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

  KSP inner_ksp = NULL;
  PetscInt number_of_iterations = ZERO;
  PetscMPIInt idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;
  PetscScalar approximation_residual_infinity_norm = PETSC_MAX_REAL;
  PetscMPIInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
  PetscMPIInt message_dest = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
  Mat A_block_jacobi_subMat[njacobi_blocks];
  IS is_cols_block_jacobi[njacobi_blocks];
  Vec b_block_jacobi[njacobi_blocks];
  Vec x_block_jacobi[njacobi_blocks];
  // PetscMPIInt send_signal = NO_SIGNAL;
  // PetscMPIInt rcv_signal = NO_SIGNAL;
  // PetscInt convergence_count = ZERO;
  VecScatter scatter_jacobi_vec_part_to_merged_vec[njacobi_blocks];
  IS is_jacobi_vec_parts;
  IS is_merged_vec[njacobi_blocks];
  PetscMPIInt vec_local_size = ZERO;
  PetscScalar *send_buffer = NULL;
  PetscScalar *rcv_buffer = NULL;
  PetscScalar *temp_buffer = NULL;
  MPI_Request send_data_request = MPI_REQUEST_NULL;
  MPI_Request send_signal_request = MPI_REQUEST_NULL;
  MPI_Status status;
  // PetscMPIInt broadcast_message = NO_MESSAGE;
  PetscMPIInt send_data_flag = ZERO;
  PetscMPIInt rcv_data_flag = ZERO;
  // PetscInt inner_solver_iterations = ZERO;
  Vec approximation_residual;
  Vec x_block_jacobi_previous_iteration = NULL;

  PetscInt nbNeigNotLCV = 0;
  PetscInt nbIterPreLocalCV = 0;
  PetscBool preLocalCV = PETSC_FALSE;
  PetscBool sLocalCV = PETSC_FALSE;
  PetscBool globalCV = PETSC_FALSE;
  PetscInt THRESHOLD_SLCV = MIN_CONVERGENCE_COUNT;
  PetscInt *neighbors = NULL;
  PetscInt nbNeighbors = 0;
  PetscInt *prevIterNumS = NULL;
  PetscInt *prevIterNumC = NULL;
  PetscMPIInt dest_node = -1;
  PetscInt cancelSPartialBuffer;
  MPI_Request cancelSPartialRequest;
  PetscInt sendSPartialBuffer;
  MPI_Request sendSPartialRequest;
  PetscLogDouble time_period_with_globalCV __attribute__((unused)) = 0.0;
  PetscLogDouble globalCV_timer = 0.0;
  PetscLogDouble MAX_TRAVERSAL_TIME __attribute__((unused)) = 0.0;

  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  PetscCall(PetscPrintf( PETSC_COMM_WORLD ,"Starting latency checking .... \n"));
  PetscMPIInt proc_rank_node_1 = 0;
  PetscMPIInt proc_rank_node_2 = 1;
  PetscCall(comm_sync_measure_latency_between_two_nodes(proc_rank_node_1, proc_rank_node_2, proc_global_rank));

  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  PetscCall(PetscFinalize());
  return 0;

  PetscCall(PetscTime(&globalCV_timer));

  PetscCall(PetscMalloc1(MAX_NEIGHBORS, &neighbors));
  PetscCall(PetscArrayfill(neighbors, -11, MAX_NEIGHBORS));

  PetscCall(PetscMalloc1(MAX_NEIGHBORS, &prevIterNumS));
  PetscCall(PetscArrayfill(prevIterNumS, -1, MAX_NEIGHBORS));

  PetscCall(PetscMalloc1(MAX_NEIGHBORS, &prevIterNumC));
  PetscCall(PetscArrayfill(prevIterNumC, 0, MAX_NEIGHBORS));

  PetscCall(build_spanning_tree(rank_jacobi_block, neighbors, &nbNeighbors, proc_local_rank, proc_global_rank, nprocs_per_jacobi_block));

  nbNeigNotLCV = nbNeighbors;
  nbIterPreLocalCV = 0;
  preLocalCV = PETSC_FALSE;
  sLocalCV = PETSC_FALSE;
  globalCV = PETSC_FALSE;

  // printf("proc %d : \n", proc_global_rank);
  // printf("neighbors : ");
  // for (int i = 0; i < nbNeighbors; i++)
  // {
  //   printf(" %d", neighbors[i]);
  // }
  // printf("\n");
  // PetscCall(PetscFinalize());
  // return 0;

  PetscCall(VecCreate(comm_jacobi_block, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n_mesh_points));
  PetscCall(VecSetType(x, VECMPI));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetUp(x));

  PetscCall(VecDuplicate(x, &b));

  PetscCall(VecDuplicate(x, &x_initial_guess));
  PetscScalar initial_scalar_value = 1.0;
  PetscCall(VecSet(x_initial_guess, initial_scalar_value));

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

  PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, &b, b_block_jacobi, x_initial_guess, rank_jacobi_block, jacobi_block_size, nprocs_per_jacobi_block, proc_local_rank));

  PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, INNER_KSP_PREFIX, INNER_PC_PREFIX));

  PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
  PetscCall(PetscMalloc1(vec_local_size, &send_buffer));
  PetscCall(PetscMalloc1(vec_local_size, &rcv_buffer));

  PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &approximation_residual));

  PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &x_block_jacobi_previous_iteration));
  PetscCall(VecCopy(x_block_jacobi[rank_jacobi_block], x_block_jacobi_previous_iteration));
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

  PetscScalar approximation_residual_infinity_norm_iter_zero __attribute__((unused)) = PETSC_MAX_REAL;
  PetscInt inner_solver_iterations __attribute__((unused)) = ZERO;
  PetscInt message_received __attribute__((unused)) = 0;
  PetscInt last_message_received_iter_number __attribute__((unused)) = 0;
  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  double start_time, end_time;
  start_time = MPI_Wtime();

  do
  {

    message_received = 0;
    inner_solver_iterations = 0;

    PetscCall(comm_async_probe_and_receive(x_block_jacobi, rcv_buffer, vec_local_size, rcv_data_flag, message_source, idx_non_current_block, &message_received));

    PetscCall(updateLocalRHS(local_right_side_vector, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, mat_mult_vec_result, rank_jacobi_block));
    PetscCall(inner_solver(comm_jacobi_block, inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, local_right_side_vector, rank_jacobi_block, &inner_solver_iterations, number_of_iterations));

    PetscCall(comm_async_test_and_send(x_block_jacobi, send_buffer, temp_buffer, &send_data_request, vec_local_size, send_data_flag, message_dest, rank_jacobi_block));

    PetscCall(comm_async_probe_and_receive(x_block_jacobi, rcv_buffer, vec_local_size, rcv_data_flag, message_source, idx_non_current_block, &message_received));

    PetscCall(VecWAXPY(approximation_residual, -1.0, x_block_jacobi_previous_iteration, x_block_jacobi[rank_jacobi_block]));
    PetscCall(VecNorm(approximation_residual, NORM_INFINITY, &approximation_residual_infinity_norm));
    PetscCall(VecCopy(x_block_jacobi[rank_jacobi_block], x_block_jacobi_previous_iteration));

    PetscCall(printResidualNorm(comm_jacobi_block, rank_jacobi_block, approximation_residual_infinity_norm, number_of_iterations));

    if (PetscApproximateLTE(approximation_residual_infinity_norm, relative_tolerance) && inner_solver_iterations > 0)
    {
      preLocalCV = PETSC_TRUE;
    }
    else
    {
      if (inner_solver_iterations != 0)
        preLocalCV = PETSC_FALSE;
    }

    cancelSPartialBuffer = number_of_iterations;
    sendSPartialBuffer = number_of_iterations;

    PetscCall(comm_async_convDetection(rank_jacobi_block, nbNeighbors, &nbNeigNotLCV, neighbors, prevIterNumS, prevIterNumC, &nbIterPreLocalCV, &preLocalCV, &sLocalCV, &globalCV, &dest_node, THRESHOLD_SLCV, number_of_iterations, &cancelSPartialBuffer, &cancelSPartialRequest, &sendSPartialBuffer, &sendSPartialRequest));

    PetscCall(comm_async_recvSPartialCV(rank_jacobi_block, &nbNeigNotLCV, prevIterNumS, prevIterNumC));

    PetscCall(comm_async_recvCancelSPartialCV(rank_jacobi_block, &nbNeigNotLCV, prevIterNumS, prevIterNumC, &globalCV));

    PetscCall(comm_async_recvGlobalCV(rank_jacobi_block, &globalCV));

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
  } while (time_period_with_globalCV <= MAX_TRAVERSAL_TIME);
  // } while (globalCV == PETSC_FALSE);

  PetscMPIInt buff;
  MPI_Request requests[nbNeighbors];
  PetscCall(comm_async_sendGlobalCV(rank_jacobi_block, nbNeighbors, neighbors, &buff, requests));
  PetscCallMPI(MPI_Waitall(nbNeighbors, requests, MPI_STATUSES_IGNORE));

  PetscCall(PetscPrintf(comm_jacobi_block, "Rank %d: PROGRAMME TERMINE\n", rank_jacobi_block));
  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  PetscCall(PetscSleep(5));
  end_time = MPI_Wtime();
  PetscCall(printElapsedTime(start_time, end_time));
  PetscCall(printTotalNumberOfIterations(comm_jacobi_block, rank_jacobi_block, number_of_iterations));

  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));

  PetscCall(comm_sync_send_and_receive_final(x_block_jacobi, vec_local_size, message_dest, message_source, rank_jacobi_block, idx_non_current_block));

  PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

  PetscScalar direct_residual_norm;
  PetscCall(computeFinalResidualNorm(A_block_jacobi, &x, b_block_jacobi, rank_jacobi_block, proc_global_rank, &direct_residual_norm));
  PetscCall(printFinalResidualNorm(direct_residual_norm));

  // END OF PROGRAM  - FREE MEMORY

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

  PetscCall(VecDestroy(&x_block_jacobi_previous_iteration));
  PetscCall(VecDestroy(&approximation_residual));
  PetscCall(VecDestroy(&local_right_side_vector));
  PetscCall(VecDestroy(&mat_mult_vec_result));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x_initial_guess));
  PetscCall(MatDestroy(&A_block_jacobi));
  PetscCall(KSPDestroy(&inner_ksp));

  // Discard any pending message
  PetscInt count;
  PetscInt message = NO_MESSAGE;

  do
  {
    MPI_Datatype data_type = MPIU_INT;
    PetscCallMPI(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &message, &status));
    if (message)
    {
      if (status.MPI_TAG == (TAG_MULTISPLITTING_DATA + idx_non_current_block))
      {
        data_type = MPIU_SCALAR;
        PetscCall(MPI_Get_count(&status, data_type, &count));
        PetscScalar *buffer;
        PetscCall(PetscMalloc1(count, &buffer));
        PetscCallMPI(MPI_Recv(buffer, count, data_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        PetscCall(PetscFree(buffer));
      }
      else
      {
        data_type = MPIU_INT;
        PetscCall(MPI_Get_count(&status, data_type, &count));
        PetscInt *buffer;
        PetscCall(PetscMalloc1(count, &buffer));
        PetscCallMPI(MPI_Recv(buffer, count, data_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        PetscCall(PetscFree(buffer));
      }
    }
  } while (message);

  PetscCallMPI(MPI_Wait(&send_data_request, MPI_STATUS_IGNORE));
  PetscCallMPI(MPI_Wait(&send_signal_request, MPI_STATUS_IGNORE));
  PetscCall(PetscFree(send_buffer));
  PetscCall(PetscFree(rcv_buffer));

  PetscCall(PetscSubcommDestroy(&sub_comm_context));
  PetscCall(PetscCommDestroy(&dcomm));
  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  PetscCall(PetscFinalize());
  return 0;
}

// #endif
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
  PetscMPIInt send_signal = NO_SIGNAL;
  // PetscMPIInt rcv_signal = NO_SIGNAL;

  IS is_jacobi_vec_parts;
  PetscInt number_of_iterations = ZERO;
  PetscMPIInt idx_non_current_block;
  PetscScalar global_iterates_difference_norm_inf __attribute__((unused)) = PETSC_MAX_REAL;
  PetscScalar current_iterate_norm_inf __attribute__((unused)) = PETSC_MAX_REAL;
  KSP inner_ksp = NULL;
  KSP outer_ksp = NULL;
  PetscMPIInt nlocal_rows_x_block = 0;
  PetscMPIInt nlocal_rows_x = 0;
  PetscScalar *send_multisplitting_data_buffer = NULL;
  PetscScalar *rcv_multisplitting_data_buffer = NULL;
  PetscScalar *send_minimization_data_buffer = NULL;
  PetscScalar *rcv_minimization_data_buffer = NULL;
  PetscInt *vec_local_idx = NULL;
  Mat R = NULL;
  Mat S = NULL;
  PetscInt basis_vector_i = 0;
  Vec x_minimized = NULL;
  Vec local_right_side_vector = NULL;
  Vec local_residual = NULL;

  // PetscInt x_local_size;
  // PetscScalar **vector_to_insert_into_S_tmp;
  // Vec x_minimized_prev_iterate = NULL;
  // Vec global_iterates_difference = NULL;
  // Vec mat_mult_vec_result = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

  // Getting applications arguments
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &n_mesh_lines, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n_mesh_columns, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-s", &s, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npb", &nprocs_per_jacobi_block, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &relative_tolerance, NULL));

  // XXX: profiling
  PetscLogStage loading_stage;
  PetscCall(PetscLogStageRegister("Loading stage", &loading_stage));
  PetscLogStage inner_solver_stage;
  PetscLogStage outer_solver_stage;
  PetscLogStage last_stage;
  PetscCall(PetscLogStageRegister("I_Solver stage", &inner_solver_stage));
  PetscCall(PetscLogStageRegister("O_Solver stage", &outer_solver_stage));
  PetscCall(PetscLogStageRegister("Last stage", &last_stage));
  PetscCall(PetscLogStagePush(loading_stage)); // XXX: profiling
  // XXX: profiling

  PetscCall(computeDimensionRelatedVariables(nprocs, nprocs_per_jacobi_block, proc_global_rank, n_mesh_lines, n_mesh_columns, &njacobi_blocks, &rank_jacobi_block, &proc_local_rank, &n_mesh_points, &jacobi_block_size));
  PetscAssert((n_mesh_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");
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

  ISLocalToGlobalMapping rmapping;
  ISLocalToGlobalMapping cmapping;
  IS is_cols_block_jacobi[njacobi_blocks];
  Mat A_block_jacobi_subMat[njacobi_blocks];
  Mat R_block_jacobi_subMat[njacobi_blocks];
  Vec b_block_jacobi[njacobi_blocks];
  Vec x_block_jacobi[njacobi_blocks];
  VecScatter scatter_jacobi_vec_part_to_merged_vec[njacobi_blocks];
  IS is_merged_vec[njacobi_blocks];
  // Mat R_transpose_R = NULL;
  // Vec vec_R_transpose_b_block_jacobi = NULL;
  Vec alpha = NULL;
  PetscInt lda;
  PetscMPIInt R_local_values_count;
  PetscInt rstart_matrix_R, rend_matrix_R;
  PetscInt rstart_matrix_S, rend_matrix_S;
  PetscMPIInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
  PetscMPIInt message_dest = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;

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
  PetscCall(PetscMalloc1(R_local_values_count, &rcv_minimization_data_buffer));

  PetscCall(create_matrix_dense(comm_jacobi_block, &S, n_mesh_points, s, MATMPIDENSE));
  PetscCall(MatZeroEntries(S));
  PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));

  // PetscCall(create_matrix_dense(comm_jacobi_block, &R_transpose_R, s, s, MATMPIDENSE));
  // PetscCall(create_vector(comm_jacobi_block, &vec_R_transpose_b_block_jacobi, s, VECMPI));
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
  // PetscCall(initializeKSP(comm_jacobi_block, &outer_ksp, R_transpose_R, rank_jacobi_block, PETSC_TRUE, OUTER_KSP_PREFIX, OUTER_PC_PREFIX));

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
  // PetscCall(KSPSetConvergenceTest(outer_ksp, MyConvergeTest, NULL, NULL));
  PetscCall(offloadJunk_00001(comm_jacobi_block, rank_jacobi_block, 2));

  PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &nlocal_rows_x_block));
  PetscCall(VecGetLocalSize(x, &nlocal_rows_x));
  PetscCall(PetscMalloc1(nlocal_rows_x_block, &send_multisplitting_data_buffer));
  PetscCall(PetscMalloc1(nlocal_rows_x_block, &rcv_multisplitting_data_buffer));

  PetscCall(create_vector(comm_jacobi_block, &x_minimized, n_mesh_points, VECMPI));
  PetscCall(VecSet(x_minimized, ZERO));

  // PetscCall(VecDuplicate(x_minimized, &x_minimized_prev_iterate));

  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_residual));
  // PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

  // PetscCall(VecGetLocalSize(x, &x_local_size));
  // PetscCall(PetscMalloc1(x_local_size, &vec_local_idx));
  // for (PetscMPIInt i = 0; i < (x_local_size); i++)
  // {
  //   vec_local_idx[i] = (proc_local_rank * x_local_size) + i;
  // }
  // PetscCall(PetscMalloc1(x_local_size, &vector_to_insert_into_S));

  // PetscCall(VecDuplicate(x_minimized, &global_iterates_difference));
  PetscCall(MatGetOwnershipRange(R, &rstart_matrix_R, &rend_matrix_R));
  PetscCall(MatGetOwnershipRange(S, &rstart_matrix_S, &rend_matrix_S));

  PetscInt *global_cols_idx;
  PetscInt *global_rows_idx;
  PetscInt *local_row_indices;

  PetscCall(PetscMalloc1((rend_matrix_S - rstart_matrix_S), &global_rows_idx));
  PetscCall(PetscMalloc1(s, &global_cols_idx));
  PetscCall(PetscMalloc1(nlocal_rows_x, &local_row_indices));

  for (PetscInt i = 0; i < s; i++)
  {
    global_cols_idx[i] = i;
  }

  for (PetscInt row = rstart_matrix_S, i = 0; row < rend_matrix_S; row++, i++)
  {
    global_rows_idx[i] = row;
    local_row_indices[i] = i;
  }

  // PetscCall(PetscFinalize());
  // return 0;

  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, nlocal_rows_x, global_rows_idx, PETSC_COPY_VALUES, &rmapping));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, s, global_cols_idx, PETSC_COPY_VALUES, &cmapping));
  PetscCall(MatSetLocalToGlobalMapping(S, rmapping, cmapping));

  PetscScalar norm_b;
  PetscCall(VecNorm(b, NORM_2, &norm_b));
  PetscCall(PetscPrintf(comm_jacobi_block, "Norm de b %e \n", norm_b));

  PetscScalar norm = 0.0;
  PetscScalar global_norm_0 = 0.0;
  PetscCall(computeFinalResidualNorm(comm_jacobi_block, comm_local_roots, A_block_jacobi, x, b_block_jacobi, local_residual, rank_jacobi_block, proc_local_rank, &global_norm_0));

  const PetscScalar *vals = NULL;

  PetscLogEvent USER_EVENT;
  PetscCall(PetscLogEventRegister("outer_solve", 0, &USER_EVENT));

  PetscCall(PetscLogStagePop()); // XXX: profiling

  PetscCall(PetscBarrier(NULL));
  double start_time, end_time;
  start_time = MPI_Wtime();

  do
  {

    // XXX: profiling
    PetscCall(PetscLogStagePush(inner_solver_stage));
    // XXX: profiling

    basis_vector_i = 0;

    while (basis_vector_i < s)
    {
      PetscCall(updateLocalRHS(A_block_jacobi_subMat[idx_non_current_block], x_block_jacobi[idx_non_current_block], b_block_jacobi[rank_jacobi_block], local_right_side_vector));

      PetscCall(inner_solver(comm_jacobi_block, inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, local_right_side_vector, rank_jacobi_block, NULL, number_of_iterations));

      PetscCall(comm_sync_send_and_receive(x_block_jacobi, nlocal_rows_x_block, message_dest, message_source, rank_jacobi_block, idx_non_current_block));

      PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

      PetscCall(VecGetArrayRead(x, &vals));
      PetscCall(MatSetValuesLocal(S, nlocal_rows_x, local_row_indices, 1, &basis_vector_i, vals, INSERT_VALUES));
      PetscCall(VecRestoreArrayRead(x, &vals));

      basis_vector_i++;
    }

    PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));
    PetscCall(PetscLogStagePop()); // XXX: profiling

    // XXX: profiling
    PetscCall(PetscLogStagePush(outer_solver_stage));
    // XXX: profiling

    PetscCall(getHalfSubMatrixFromR(R, R_block_jacobi_subMat, n_mesh_lines, n_mesh_columns, rank_jacobi_block));
    PetscCall(MatMatMult(A_block_jacobi_resdistributed, S, MAT_REUSE_MATRIX, PETSC_DETERMINE, &R_block_jacobi_subMat[rank_jacobi_block]));
    PetscCall(restoreHalfSubMatrixToR(R, R_block_jacobi_subMat, rank_jacobi_block));

    PetscCall(comm_sync_send_and_receive_minimization(R, send_minimization_data_buffer, rcv_minimization_data_buffer, R_local_values_count, message_dest, message_source, rank_jacobi_block, idx_non_current_block, n_mesh_points, rstart_matrix_R, rend_matrix_R, lda, s));

    PetscCall(PetscLogEventBegin(USER_EVENT, 0, 0, 0, 0));
    PetscCall(outer_solver_norm_equation(comm_jacobi_block, outer_ksp, x_minimized, R, S, alpha, b, rank_jacobi_block, number_of_iterations));
    PetscCall(PetscLogEventEnd(USER_EVENT, 0, 0, 0, 0));

    /* Certain solvers, under certain conditions,
     may not compute the final residual norm
      in an iteration, in that case the previous norm
       is returned. */

       
    // PetscCall(computeFinalResidualNorm(comm_jacobi_block, comm_local_roots, A_block_jacobi, x_minimized, b_block_jacobi, local_residual, rank_jacobi_block, proc_local_rank, &norm));
    // PetscCall(printFinalResidualNorm(norm));
    // PetscScalar norm_test;
    PetscCall(KSPGetResidualNorm(outer_ksp, &norm));
    PetscCall(printFinalResidualNorm(norm));

    if (norm <= PetscMax(absolute_tolerance, relative_tolerance * global_norm_0))
    {
      send_signal = CONVERGENCE_SIGNAL;
    }

    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_minimized, x_block_jacobi[rank_jacobi_block], INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_minimized, x_block_jacobi[rank_jacobi_block], INSERT_VALUES, SCATTER_REVERSE));

    // PetscInt k;
    // PetscCall(KSPGetTolerances(inner_ksp, NULL, NULL, NULL, &k));
    // PetscCall(KSPSetTolerances(inner_ksp, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT, k + 1));

    number_of_iterations = number_of_iterations + 1;
    PetscCall(PetscLogStagePop()); // XXX: profiling

  } while (send_signal != CONVERGENCE_SIGNAL);

  PetscCall(PetscBarrier(NULL));
  end_time = MPI_Wtime();

  PetscCall(PetscBarrier(NULL));
  // XXX: profiling
  PetscCall(PetscLogStagePush(last_stage));
  // XXX: profiling

  PetscCall(printElapsedTime(start_time, end_time));
  PetscCall(printTotalNumberOfIterations_2(comm_jacobi_block, rank_jacobi_block, number_of_iterations, s));

  PetscCall(comm_sync_send_and_receive_final(x_block_jacobi, nlocal_rows_x_block, message_dest, message_source, rank_jacobi_block, idx_non_current_block));

  PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(computeFinalResidualNorm(comm_jacobi_block, comm_local_roots, A_block_jacobi, x, b_block_jacobi, local_residual, rank_jacobi_block, proc_local_rank, &norm));
  PetscCall(printFinalResidualNorm(norm));

  PetscScalar error;
  PetscCall(computeError(x, u, &error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Erreur : %e \n", error));

  // Start free of memory
  for (PetscInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(ISDestroy(&is_cols_block_jacobi[i]));
    PetscCall(VecDestroy(&x_block_jacobi[i]));
    PetscCall(VecDestroy(&b_block_jacobi[i]));
    PetscCall(MatDestroy(&A_block_jacobi_subMat[i]));
    PetscCall(VecScatterDestroy(&scatter_jacobi_vec_part_to_merged_vec[i]));
    PetscCall(ISDestroy(&is_merged_vec[i]));
  }

  PetscCall(ISLocalToGlobalMappingDestroy(&rmapping));
  PetscCall(ISLocalToGlobalMappingDestroy(&cmapping));
  PetscCall(PetscFree(global_cols_idx));
  PetscCall(PetscFree(global_rows_idx));
  PetscCall(PetscFree(local_row_indices));
  PetscCall(PetscFree(vec_local_idx));
  PetscCall(VecDestroy(&local_right_side_vector));
  PetscCall(VecDestroy(&local_residual));
  PetscCall(ISDestroy(&is_jacobi_vec_parts));
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
  PetscCall(VecDestroy(&alpha));

  // PetscFree(vector_to_insert_into_S);
  // PetscCall(VecDestroy(&x_minimized_prev_iterate));
  // PetscCall(VecDestroy(&mat_mult_vec_result));
  // PetscCall(VecDestroy(&global_iterates_difference));
  // PetscCall(MatDestroy(&R_transpose_R));
  // PetscCall(VecDestroy(&vec_R_transpose_b_block_jacobi));

  PetscCall(PetscFree(send_multisplitting_data_buffer));
  PetscCall(PetscFree(rcv_multisplitting_data_buffer));
  PetscCall(PetscFree(send_minimization_data_buffer));
  PetscCall(PetscFree(rcv_minimization_data_buffer));

  // PetscCall(comm_discard_pending_messages());

  PetscCall(PetscSubcommDestroy(&sub_comm_context));
  PetscCall(PetscCommDestroy(&dcomm));

  PetscCall(PetscLogStagePop()); // XXX: profiling
  PetscCall(PetscFinalize());
  return 0;
}

// #endif

// ////
// PetscScalar val1, local_norm_0;
// Vec res;
// PetscCall(VecNorm(local_right_side_vector, NORM_2, &val1));
// PetscCall(PetscPrintf(comm_jacobi_block, "norm of b %e \n", val1));
// PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &res));
// PetscCall(MatResidual(A_block_jacobi_subMat[rank_jacobi_block], local_right_side_vector, x_block_jacobi[rank_jacobi_block], res));
// PetscCall(VecNorm(res, NORM_2, &local_norm_0));
// PetscCall(PetscPrintf(comm_jacobi_block, "Block %d local norm 0 %e ====== inner_rtol * norm_0 %e \n",rank_jacobi_block, local_norm_0 , (1.e-3) * local_norm_0));
// ///

// PetscCall(VecWAXPY(global_iterates_difference, -1.0, x_minimized_prev_iterate, x_minimized));
// PetscCall(VecNorm(global_iterates_difference, NORM_INFINITY, &global_iterates_difference_norm_inf));
// PetscCall(VecNorm(x_minimized, NORM_INFINITY, &current_iterate_norm_inf));
// PetscCall(printResidualNorm(comm_jacobi_block, rank_jacobi_block, global_iterates_difference_norm_inf, number_of_iterations));
// if (global_iterates_difference_norm_inf <= PetscMax(absolute_tolerance, relative_tolerance * current_iterate_norm_inf))
// {
//   send_signal = CONVERGENCE_SIGNAL;
// }

// if (rank_jacobi_block == 0)
// {
//   PetscInt size_b;
//   PetscCall(VecGetSize(b_block_jacobi[0], &size_b));
//   PetscCall(PetscPrintf(comm_jacobi_block, "size of b %d\n", size_b));
// }

// PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
// PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
// PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
// PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

// // if (rank_jacobi_block == 0)
// //   PetscCall(VecView(x, PETSC_VIEWER_STDOUT_(comm_jacobi_block)));
// // PetscCall(VecGetValues(x, x_local_size, vec_local_idx, vector_to_insert_into_S));

// PetscCall(MatSetLocalToGlobalMapping(S, rmapping, cmapping));
// PetscCall(VecGetArrayRead(x_block_jacobi[rank_jacobi_block], &vals));
// PetscCall(MatSetValuesLocal(S, nlocal_rows, local_row_indices, 1, &local_cols_idx, vals, INSERT_VALUES));
// PetscCall(VecRestoreArrayRead(x_block_jacobi[rank_jacobi_block], &vals));

// PetscCall(MatSetLocalToGlobalMapping(S, rmapping, cmapping));
// PetscCall(VecGetArrayRead(x_block_jacobi[idx_non_current_block], &vals));
// PetscCall(MatSetValuesLocal(S, nlocal_rows, local_row_indices, 1, &local_cols_idx, vals, INSERT_VALUES));
// PetscCall(VecRestoreArrayRead(x_block_jacobi[idx_non_current_block], &vals));

// PetscCall(MatSetValuesLocal(S, x_local_size, vec_local_idx, ONE, &n_vectors_inserted, &vector_to_insert_into_S, INSERT_VALUES));
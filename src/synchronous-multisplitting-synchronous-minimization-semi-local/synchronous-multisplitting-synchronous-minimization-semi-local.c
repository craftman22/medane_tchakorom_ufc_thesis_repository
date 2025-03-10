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

int main(int argc, char **argv)
{

  Mat A_block_jacobi = NULL;
  Vec x = NULL; // vector of unknows
  Vec b = NULL; // right hand side vector
  Vec x_initial_guess = NULL;
  PetscInt s;
  PetscInt nprocs;
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
  PetscSubcomm sub_comm_context;
  MPI_Comm dcomm;
  MPI_Comm comm_jacobi_block;
  PetscMPIInt send_signal = NO_SIGNAL;
  PetscMPIInt rcv_signal = NO_SIGNAL;

  IS is_jacobi_vec_parts;
  PetscInt number_of_iterations = ZERO;
  PetscMPIInt idx_non_current_block;
  PetscScalar approximation_residual_infinity_norm = PETSC_MAX_REAL;
  KSP inner_ksp = NULL;
  KSP outer_ksp = NULL;
  PetscMPIInt vec_local_size = ZERO;

  PetscInt *vec_local_idx = NULL;
  PetscInt x_local_size;
  PetscScalar *vector_to_insert_into_S;

  // Minimization variables

  Mat R = NULL;
  Mat S = NULL;
  PetscInt n_vectors_inserted;
  Vec x_minimized = NULL;
  Vec x_minimized_prev_iteration = NULL;
  Vec approximate_residual = NULL;
  Vec local_right_side_vector = NULL;
  Vec mat_mult_vec_result = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &n_mesh_lines, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n_mesh_columns, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-s", &s, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npb", &nprocs_per_jacobi_block, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &relative_tolerance, NULL));

  PetscCall(computeDimensionRelatedVariables(nprocs, nprocs_per_jacobi_block, proc_global_rank, n_mesh_lines, n_mesh_columns, &njacobi_blocks, &rank_jacobi_block, &proc_local_rank, &n_mesh_points, &jacobi_block_size));
  PetscAssert((n_mesh_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");
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
  PetscMPIInt broadcast_message = NO_MESSAGE;
  PetscMPIInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
  PetscMPIInt message_dest = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;

  for (PetscMPIInt i = 0; i < njacobi_blocks; i++)
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
  PetscCall(VecDuplicate(x, &x_initial_guess));
  PetscCall(VecSet(x_initial_guess, ONE));

  PetscCall(divideSubDomainIntoBlockMatrices(comm_jacobi_block, A_block_jacobi, A_block_jacobi_subMat, is_cols_block_jacobi, rank_jacobi_block, njacobi_blocks, proc_local_rank, nprocs_per_jacobi_block));

  PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, ZERO, ONE, &is_jacobi_vec_parts));

  for (PetscMPIInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, (i * (jacobi_block_size)), ONE, &is_merged_vec[i]));
    PetscCall(VecScatterCreate(b_block_jacobi[i], is_jacobi_vec_parts, b, is_merged_vec[i], &scatter_jacobi_vec_part_to_merged_vec[i]));
  }

  PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, &b, b_block_jacobi, x_initial_guess, rank_jacobi_block, jacobi_block_size, nprocs_per_jacobi_block, proc_local_rank));

  PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, INNER_KSP_PREFIX, INNER_PC_PREFIX));
  PetscCall(initializeKSP(comm_jacobi_block, &outer_ksp, NULL, rank_jacobi_block, PETSC_TRUE, OUTER_KSP_PREFIX, OUTER_PC_PREFIX));

  PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));

  PetscCall(create_vector(comm_jacobi_block, &x_minimized, n_mesh_points, VECMPI));
  PetscCall(VecSet(x_minimized, ZERO));

  PetscCall(VecDuplicate(x_minimized, &x_minimized_prev_iteration));

  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

  PetscCall(VecGetLocalSize(x, &x_local_size));
  PetscCall(PetscMalloc1(x_local_size, &vec_local_idx));
  for (PetscMPIInt i = 0; i < (x_local_size); i++)
  {
    vec_local_idx[i] = (proc_local_rank * x_local_size) + i;
  }
  PetscCall(PetscMalloc1(x_local_size, &vector_to_insert_into_S));
  PetscCall(VecDuplicate(x_minimized, &approximate_residual));

  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  double start_time, end_time;
  start_time = MPI_Wtime();

  do
  {

    n_vectors_inserted = 0;
    PetscCall(VecCopy(x_minimized, x_minimized_prev_iteration));

    while (n_vectors_inserted < s)
    {
      PetscCall(updateLocalRHS(local_right_side_vector, A_block_jacobi_subMat,x_block_jacobi, b_block_jacobi, mat_mult_vec_result, rank_jacobi_block));
      PetscCall(inner_solver(comm_jacobi_block, inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, local_right_side_vector, rank_jacobi_block, NULL, number_of_iterations));

      PetscCall(comm_sync_send_and_receive(x_block_jacobi, vec_local_size, message_dest, message_source, rank_jacobi_block, idx_non_current_block));

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

#ifdef VERSION1
    PetscCall(outer_solver(comm_jacobi_block, outer_ksp, x_block_jacobi[rank_jacobi_block], R, S, R_transpose_R, vec_R_transpose_b_block_jacobi, alpha, b_block_jacobi[rank_jacobi_block], rank_jacobi_block, s, number_of_iterations));
#endif

#ifdef VERSION2
    PetscCall(updateLocalRHS(local_right_side_vector, A_block_jacobi_subMat,x_block_jacobi, b_block_jacobi, mat_mult_vec_result, rank_jacobi_block));
    PetscCall(outer_solver(comm_jacobi_block, outer_ksp, x_block_jacobi[rank_jacobi_block], R, S, R_transpose_R, vec_R_transpose_b_block_jacobi, alpha, local_right_side_vector, rank_jacobi_block, s, number_of_iterations));
#endif
    PetscCall(VecWAXPY(approximate_residual, -1.0, x_minimized_prev_iteration, x_minimized));

    PetscCall(VecNorm(approximate_residual, NORM_INFINITY, &approximation_residual_infinity_norm));

    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_minimized, x_block_jacobi[rank_jacobi_block], INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_minimized, x_block_jacobi[rank_jacobi_block], INSERT_VALUES, SCATTER_REVERSE));

    PetscCall(printResidualNorm(comm_jacobi_block, rank_jacobi_block, approximation_residual_infinity_norm, number_of_iterations));

    if (PetscApproximateLTE(approximation_residual_infinity_norm, relative_tolerance))
      send_signal = CONVERGENCE_SIGNAL;

    PetscCall(comm_sync_convergence_detection(&broadcast_message, send_signal, rcv_signal, message_dest, message_source, rank_jacobi_block, idx_non_current_block, proc_local_rank));

    PetscCallMPI(MPI_Bcast(&broadcast_message, ONE, MPIU_INT, proc_local_rank, comm_jacobi_block));

    number_of_iterations = number_of_iterations + 1;

  } while (broadcast_message != TERMINATE_SIGNAL);

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
  PetscCall(computeFinalResidualNorm(A_block_jacobi, &x, b_block_jacobi, rank_jacobi_block, proc_global_rank, &direct_residual_norm));

  PetscCall(printFinalResidualNorm(direct_residual_norm));

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
  PetscCall(VecDestroy(&x_minimized_prev_iteration));
  PetscCall(VecDestroy(&approximate_residual));
  PetscCall(VecDestroy( &local_right_side_vector));
  PetscCall(VecDestroy( &mat_mult_vec_result));
  PetscCall(ISDestroy(&is_jacobi_vec_parts));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x_minimized));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x_initial_guess));
  PetscCall(MatDestroy(&A_block_jacobi));
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

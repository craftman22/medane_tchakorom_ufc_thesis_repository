#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"
#include "constants.h"
#include "utils.h"
#include "comm.h"

// #ifdef VERSION_1_0

int main(int argc, char **argv)
{

  Mat A_block_jacobi = NULL;     // Operator matrix
  Vec x = NULL;                  // approximation solution at iteration (k)
  Vec x_previous_iterate = NULL; // approximation solution at iteration (k-1)
  Vec b = NULL;                  // right hand side vector
  Vec u = NULL;

  PetscMPIInt nprocs;
  PetscMPIInt proc_global_rank;
  PetscInt n_mesh_lines = 4;
  PetscInt n_mesh_columns = 4;
  PetscInt njacobi_blocks;
  PetscMPIInt rank_jacobi_block;
  PetscMPIInt proc_local_rank;
  PetscInt n_mesh_points;
  PetscInt jacobi_block_size;

  PetscInt s;
  PetscInt nprocs_per_jacobi_block = 1;
  PetscScalar relative_tolerance = 1e-2;
  PetscScalar absolute_tolerance = 1e-100;

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
  // PetscCall(PetscOptionsGetReal(NULL, NULL, "-atol", &absolute_tolerance, NULL));

  PetscCall(computeDimensionRelatedVariables(nprocs, nprocs_per_jacobi_block, proc_global_rank, n_mesh_lines, n_mesh_columns, &njacobi_blocks, &rank_jacobi_block, &proc_local_rank, &n_mesh_points, &jacobi_block_size));
  PetscAssert((n_mesh_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");
  PetscSubcomm sub_comm_context = NULL;
  MPI_Comm dcomm;
  PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD, &dcomm, NULL));
  PetscCall(PetscSubcommCreate(dcomm, &sub_comm_context));
  PetscCall(PetscSubcommSetNumber(sub_comm_context, njacobi_blocks));
  PetscCall(PetscSubcommSetType(sub_comm_context, PETSC_SUBCOMM_CONTIGUOUS));
  PetscCall(PetscSubcommSetFromOptions(sub_comm_context));
  MPI_Comm comm_jacobi_block = PetscSubcommChild(sub_comm_context);

  PetscBool stop_condition = PETSC_FALSE;
  PetscInt number_of_iterations = ZERO;
  PetscMPIInt idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;
  // PetscScalar iterate_difference_norm_inf = PETSC_MAX_REAL;
  // PetscScalar current_iterate_norm_inf = PETSC_MAX_REAL;
  PetscMPIInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
  PetscMPIInt message_dest = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
  PetscScalar initial_scalar_value = 1.0;
  Mat A_block_jacobi_subMat[njacobi_blocks];
  IS is_cols_block_jacobi[njacobi_blocks];
  Vec b_block_jacobi[njacobi_blocks];
  Vec x_block_jacobi[njacobi_blocks];
  VecScatter scatter_jacobi_vec_part_to_merged_vec[njacobi_blocks];
  IS is_jacobi_vec_parts;
  IS is_merged_vec[njacobi_blocks];
  KSP inner_ksp = NULL;
  PetscScalar *send_multisplitting_data_buffer = NULL;
  PetscScalar *rcv_multisplitting_data_buffer = NULL;
  PetscMPIInt vec_local_size = 0;
  Vec iterates_difference;

  PetscCall(create_vector(comm_jacobi_block, &x, n_mesh_points, VECMPI));
  PetscCall(VecDuplicate(x, &x_previous_iterate));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &u));
  PetscCall(VecSet(u, initial_scalar_value));

  PetscCall(create_matrix_sparse(comm_jacobi_block, &A_block_jacobi, n_mesh_points / njacobi_blocks, n_mesh_points, MATMPIAIJ, 5, 5));
  PetscCall(poisson2DMatrix(&A_block_jacobi, n_mesh_lines, n_mesh_columns, rank_jacobi_block, njacobi_blocks));
  PetscCall(divideSubDomainIntoBlockMatrices(comm_jacobi_block, A_block_jacobi, A_block_jacobi_subMat, is_cols_block_jacobi, rank_jacobi_block, njacobi_blocks, proc_local_rank, nprocs_per_jacobi_block));

  for (PetscMPIInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(create_vector(comm_jacobi_block, &x_block_jacobi[i], jacobi_block_size, VECMPI));
    PetscCall(create_vector(comm_jacobi_block, &b_block_jacobi[i], jacobi_block_size, VECMPI));
  }

  PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, ZERO, ONE, &is_jacobi_vec_parts));
  for (PetscMPIInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, (i * (jacobi_block_size)), ONE, &is_merged_vec[i]));
    PetscCall(VecScatterCreate(b_block_jacobi[i], is_jacobi_vec_parts, b, is_merged_vec[i], &scatter_jacobi_vec_part_to_merged_vec[i]));
  }

  PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, &b, b_block_jacobi, u, rank_jacobi_block, jacobi_block_size, nprocs_per_jacobi_block, proc_local_rank));

  // PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, INNER_KSP_PREFIX, INNER_PC_PREFIX));

  if (rank_jacobi_block == 0)
  {

    PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, "inner1_", "inner1_"));
  }

  if (rank_jacobi_block == 1)
  {

    PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, "inner2_", "inner2_"));
  }

  PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
  PetscCall(PetscMalloc1(vec_local_size, &send_multisplitting_data_buffer));
  PetscCall(PetscMalloc1(vec_local_size, &rcv_multisplitting_data_buffer));

  PetscCall(VecDuplicate(x, &iterates_difference));
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

  PetscScalar val;
  PetscCall(VecNorm(b, NORM_2, &val));
  printf("Norm de b %e \n", val);

  PetscScalar global_norm_0 = 0.0;
  PetscCall(computeFinalResidualNorm(A_block_jacobi, x, b_block_jacobi, rank_jacobi_block, proc_local_rank, &global_norm_0));

  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  double start_time, end_time;
  start_time = MPI_Wtime();

  do
  {

    PetscCall(updateLocalRHS(A_block_jacobi_subMat[idx_non_current_block], x_block_jacobi[idx_non_current_block], b_block_jacobi[rank_jacobi_block], local_right_side_vector));

    PetscCall(inner_solver(comm_jacobi_block, inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, local_right_side_vector, rank_jacobi_block, NULL, number_of_iterations));

    PetscCall(comm_sync_send_and_receive(x_block_jacobi, vec_local_size, message_dest, message_source, rank_jacobi_block, idx_non_current_block));

    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

    PetscScalar direct_residual_norm;
    PetscCall(computeFinalResidualNorm(A_block_jacobi, x, b_block_jacobi, rank_jacobi_block, proc_local_rank, &direct_residual_norm));
    // PetscCall(printFinalResidualNorm(direct_residual_norm));
    if (direct_residual_norm <= PetscMax(absolute_tolerance, relative_tolerance * global_norm_0))
    {
      stop_condition = CONVERGENCE_SIGNAL;
    }

    number_of_iterations = number_of_iterations + 1;

  } while (stop_condition == PETSC_FALSE);

  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  end_time = MPI_Wtime();

  PetscCall(printElapsedTime(start_time, end_time));
  PetscCall(printTotalNumberOfIterations(comm_jacobi_block, rank_jacobi_block, number_of_iterations));

  PetscScalar direct_residual_norm;
  PetscCall(computeFinalResidualNorm(A_block_jacobi, x, b_block_jacobi, rank_jacobi_block, proc_local_rank, &direct_residual_norm));

  PetscCall(printFinalResidualNorm(direct_residual_norm));

  PetscScalar error;
  PetscCall(computeError(x, u, &error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Erreur : %e \n", error));

  // PetscCall(VecView(x, PETSC_VIEWER_STDOUT_SELF));

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

  PetscCall(VecDestroy(&iterates_difference));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&local_right_side_vector));
  PetscCall(VecDestroy(&mat_mult_vec_result));
  PetscCall(VecDestroy(&x_previous_iterate));
  PetscCall(MatDestroy(&A_block_jacobi));
  PetscCall(PetscFree(send_multisplitting_data_buffer));
  PetscCall(PetscFree(rcv_multisplitting_data_buffer));
  PetscCall(KSPDestroy(&inner_ksp));
  PetscCall(PetscSubcommDestroy(&sub_comm_context));
  PetscCall(PetscCommDestroy(&dcomm));

  PetscCall(PetscFinalize());
  return 0;
}

// #endif

// PetscCall(VecWAXPY(iterates_difference, -1.0, x_previous_iterate, x));
// PetscCall(VecNorm(iterates_difference, NORM_INFINITY, &iterate_difference_norm_inf));
// PetscCall(VecNorm(x, NORM_INFINITY, &current_iterate_norm_inf));
// PetscCall(printResidualNorm(comm_jacobi_block, rank_jacobi_block, iterate_difference_norm_inf, number_of_iterations));
// if (iterate_difference_norm_inf <= PetscMax(absolute_tolerance, relative_tolerance * current_iterate_norm_inf))
// {
//   stop_condition = PETSC_TRUE;
// }
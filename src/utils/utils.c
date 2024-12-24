#include "constants.h"
#include "utils.h"
#include <petscts.h>

PetscErrorCode poisson2DMatrix(Mat *A_block_jacobi, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block, PetscInt njacobi_blocks)
{
  PetscFunctionBeginUser;

  PetscInt Idx_start = 0, Idx_end = 0;
  PetscCall(MatGetOwnershipRange(*A_block_jacobi, &Idx_start, &Idx_end));

  PetscInt rowBlockSize = (n_grid_lines * n_grid_columns) / njacobi_blocks;
  // PetscInt columnBlockSize = n_grid_lines * n_grid_columns;

  PetscInt i, j, J;
  PetscScalar v;
  PetscInt Ii_new;

  for (PetscInt Ii = (rank_jacobi_block * rowBlockSize) + Idx_start; Ii < (rank_jacobi_block * rowBlockSize) + Idx_end; Ii++)
  {
    v = -1.0, i = Ii / n_grid_columns, j = Ii - i * n_grid_columns;
    Ii_new = Ii - (rank_jacobi_block * rowBlockSize);
    if (i > 0)
    {
      J = Ii - n_grid_columns;
      PetscCall(MatSetValue(*A_block_jacobi, Ii_new, J, v, INSERT_VALUES));
    }
    if (i < n_grid_lines - 1)
    {
      J = Ii + n_grid_columns;
      PetscCall(MatSetValue(*A_block_jacobi, Ii_new, J, v, INSERT_VALUES));
    }
    if (j > 0)
    {
      J = Ii - 1;
      PetscCall(MatSetValue(*A_block_jacobi, Ii_new, J, v, INSERT_VALUES));
    }
    if (j < n_grid_columns - 1)
    {
      J = Ii + 1;
      PetscCall(MatSetValue(*A_block_jacobi, Ii_new, J, v, INSERT_VALUES));
    }
    v = 4.0;
    PetscCall(MatSetValue(*A_block_jacobi, Ii_new, Ii, v, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(*A_block_jacobi, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A_block_jacobi, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Divide the A_block_jacobi matrix into number_of_blocks matrices in the y direction. Resulting matrix has the possesses the same distribution
// on the processor on the x axis, but different distribution on y-axis
PetscErrorCode divideSubDomainIntoBlockMatrices(MPI_Comm comm_jacobi_block, Mat A_block_jacobi, Mat *A_block_jacobi_subMat, IS *is_cols_block_jacobi, PetscInt rank_jacobi_block, PetscInt njacobi_blocks, PetscInt proc_local_rank, PetscInt nprocs_per_jacobi_block)
{
  PetscFunctionBeginUser;
  PetscInt n_rows;
  PetscCall(MatGetSize(A_block_jacobi, &n_rows, NULL)); // return the number of rows and columns of the matrix

  for (PetscInt i = 0; i < njacobi_blocks; ++i)
  {
    PetscInt n = n_rows / nprocs_per_jacobi_block;                                          // length of the locally owned portion of the index set
    PetscInt first = (i * n_rows) + (proc_local_rank * (n_rows / nprocs_per_jacobi_block)); // the first element of the locally owned portion of the index set
    PetscInt step = 1;                                                                      // the change to the next index
    PetscCall(ISCreateStride(comm_jacobi_block, n, first, step, &is_cols_block_jacobi[i]));
  }

  IS is_rows_block_jacobi;
  PetscInt n = n_rows / nprocs_per_jacobi_block;                                               // length of the locally owned portion of the index set
  PetscInt first = proc_local_rank * (n_rows / nprocs_per_jacobi_block); /*+rankBlock*n_rows*/ // the first element of the locally owned portion of the index set
  PetscInt step = 1;                                                                           // the change to the next index
  PetscCall(ISCreateStride(comm_jacobi_block, n, first, step, &is_rows_block_jacobi));

  for (PetscInt i = 0; i < njacobi_blocks; ++i)
  {
    // PetscCall(MatGetSubMatrix(A_block_jacobi, is_rows_block_jacobi, is_cols_block_jacobi[i], MAT_INITIAL_MATRIX, &A_block_jacobi_subMat[i]));
    PetscCall(MatCreateSubMatrix(A_block_jacobi, is_rows_block_jacobi, is_cols_block_jacobi[i], MAT_INITIAL_MATRIX, &A_block_jacobi_subMat[i]));
  }

  PetscCall(ISDestroy(&is_rows_block_jacobi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Initiliaze a KSP context
PetscErrorCode initializeKSP(MPI_Comm comm_jacobi_block, KSP *ksp, Mat operator_matrix, PetscScalar rank_jacobi_block, PetscBool zero_initial_guess, const char *ksp_prefix, const char *pc_prefix)
{
  PetscFunctionBeginUser;
  PC pc = NULL;

  PetscCall(KSPCreate(comm_jacobi_block, ksp));
  PetscCall(KSPSetOperators(*ksp, operator_matrix, operator_matrix));
  PetscCall(KSPSetOptionsPrefix(*ksp, ksp_prefix));

  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetOptionsPrefix(pc, pc_prefix));

  PetscCall(KSPSetInitialGuessNonzero(*ksp, !zero_initial_guess));// TODO: ici, revoir la conversion du boolean

  PetscCall(PCSetFromOptions(pc));
  PetscCall(KSPSetFromOptions(*ksp));

  PetscCall(PCSetUp(pc));
  PetscCall(KSPSetUp(*ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode updateKSPoperators(KSP *ksp, Mat operator_matrix)
{
  PetscFunctionBeginUser;

  PetscCall(KSPSetOperators(*ksp, operator_matrix, operator_matrix));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeFinalResidualNorm(Mat A_block_jacobi, Vec *x, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt proc_local_rank, PetscScalar *direct_residual_norm)
{
  PetscFunctionBegin;
  Vec direct_local_residual = NULL;
  PetscScalar direct_local_residual_norm2 = PETSC_MAX_REAL;
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &direct_local_residual));
  PetscCall(MatResidual(A_block_jacobi, b_block_jacobi[rank_jacobi_block], *x, direct_local_residual));
  PetscCall(VecNorm(direct_local_residual, NORM_2, &direct_local_residual_norm2));
  direct_local_residual_norm2 = direct_local_residual_norm2 * direct_local_residual_norm2;
  if (proc_local_rank != 0)
  {
    direct_local_residual_norm2 = 0.0;
  }

  *direct_residual_norm = PETSC_MAX_REAL;
  PetscCallMPI(MPI_Allreduce(&direct_local_residual_norm2, direct_residual_norm, 1, MPIU_SCALAR, MPI_SUM, MPI_COMM_WORLD));
  *direct_residual_norm = sqrt(*direct_residual_norm);

  PetscCall(VecDestroy(&direct_local_residual));

  // PetscCall(PetscPrintf(MPI_COMM_WORLD, " Total number of iterations: %d   ====  Direct norm 2 ====  %e \n", number_of_iterations, direct_global_residual_norm2));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode outer_solver(MPI_Comm comm_jacobi_block, KSP *outer_ksp, Vec x_minimized, Mat R, Mat S, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt s)
{

  PetscFunctionBegin;

  Mat R_transpose_R = NULL;
  Vec vec_R_transpose_b_block_jacobi = NULL;

  PetscCall(MatTransposeMatMult(R, R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &R_transpose_R));
  PetscCall(VecCreate(comm_jacobi_block, &vec_R_transpose_b_block_jacobi));
  PetscCall(VecSetType(vec_R_transpose_b_block_jacobi, VECMPI));
  PetscCall(VecSetSizes(vec_R_transpose_b_block_jacobi, PETSC_DECIDE, s));
  PetscCall(VecSetFromOptions(vec_R_transpose_b_block_jacobi));
  PetscCall(VecSetUp(vec_R_transpose_b_block_jacobi));
  PetscCall(MatMultTranspose(R, b_block_jacobi[rank_jacobi_block], vec_R_transpose_b_block_jacobi));

  PetscCall(updateKSPoperators(outer_ksp, R_transpose_R));

  PetscScalar ksp_rtol;
  PetscInt kps_max_iters;
  KSPType ksp_type;
  PCType pc_type;
  PC pc;

  PetscCall(KSPGetTolerances(*outer_ksp, &ksp_rtol, NULL, NULL, &kps_max_iters));
  PetscCall(KSPGetType(*outer_ksp, &ksp_type));
  PetscCall((KSPGetPC(*outer_ksp, &pc)));
  PetscCall(PCGetType(pc, &pc_type));
  PetscCall(KSPGetType(*outer_ksp, &ksp_type));

  Vec alpha = NULL;
  PetscCall(VecCreate(comm_jacobi_block, &alpha));
  PetscCall(VecSetType(alpha, VECMPI));
  PetscCall(VecSetSizes(alpha, PETSC_DECIDE, s));
  PetscCall(VecSetFromOptions(alpha));
  PetscCall(VecSetUp(alpha));
  PetscCall(KSPSolve(*outer_ksp, vec_R_transpose_b_block_jacobi, alpha));

  PetscCall(MatMult(S, alpha, x_minimized));

  PetscCall(MatDestroy(&R_transpose_R));
  PetscCall(VecDestroy(&vec_R_transpose_b_block_jacobi));
  PetscCall(VecDestroy(&alpha));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Compute the right hand side b in each block and then assemble the vector b
PetscErrorCode computeTheRightHandSideWithInitialGuess(MPI_Comm comm_jacobi_block, VecScatter *scatter_jacobi_vec_part_to_merged_vec, Mat A_block_jacobi, Vec *b, Vec *b_block_jacobi, Vec x_initial_guess, PetscInt rank_jacobi_block, PetscInt jacobi_block_size, PetscInt nprocs_per_jacobi_block, PetscInt proc_local_rank)
{
  PetscFunctionBegin;
  PetscCall(MatMult(A_block_jacobi, x_initial_guess, b_block_jacobi[rank_jacobi_block]));
  PetscInt idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;

  PetscScalar *send_buffer = NULL;
  PetscScalar *rcv_buffer = NULL;
  PetscInt vec_local_size = 0;
  PetscCall(VecGetLocalSize(b_block_jacobi[rank_jacobi_block], &vec_local_size));

  if (rank_jacobi_block == 0)
  {

    PetscCall(VecGetArray(b_block_jacobi[rank_jacobi_block], &send_buffer));
    PetscCallMPI(MPI_Send(send_buffer, vec_local_size, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 0, MPI_COMM_WORLD));
    PetscCall(VecRestoreArray(b_block_jacobi[rank_jacobi_block], &send_buffer));

    PetscCall(VecGetArray(b_block_jacobi[idx_non_current_block], &rcv_buffer));
    PetscCallMPI(MPI_Recv(rcv_buffer, vec_local_size, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    PetscCall(VecRestoreArray(b_block_jacobi[idx_non_current_block], &rcv_buffer));
  }
  else if (rank_jacobi_block == 1)
  {
    PetscCall(VecGetArray(b_block_jacobi[idx_non_current_block], &rcv_buffer));
    PetscCallMPI(MPI_Recv(rcv_buffer, vec_local_size, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    PetscCall(VecRestoreArray(b_block_jacobi[idx_non_current_block], &rcv_buffer));

    PetscCall(VecGetArray(b_block_jacobi[rank_jacobi_block], &send_buffer));
    PetscCallMPI(MPI_Send(send_buffer, vec_local_size, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 1, MPI_COMM_WORLD));
    PetscCall(VecRestoreArray(b_block_jacobi[rank_jacobi_block], &send_buffer));
  }

  PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], b_block_jacobi[rank_jacobi_block], *b, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], b_block_jacobi[rank_jacobi_block], *b, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], b_block_jacobi[idx_non_current_block], *b, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], b_block_jacobi[idx_non_current_block], *b, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(PetscFree(rcv_buffer));
  PetscCall(PetscFree(send_buffer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeDimensionRelatedVariables(PetscInt nprocs, PetscInt nprocs_per_jacobi_block, PetscInt proc_global_rank, PetscInt n_mesh_lines, PetscInt n_mesh_columns,

                                                PetscInt *njacobi_blocks, PetscInt *rank_jacobi_block, PetscInt *proc_local_rank, PetscInt *n_mesh_points, PetscInt *jacobi_block_size)
{
  PetscFunctionBeginUser;
  *njacobi_blocks = (PetscInt)(nprocs / nprocs_per_jacobi_block);
  *rank_jacobi_block = proc_global_rank / nprocs_per_jacobi_block;
  *proc_local_rank = (proc_global_rank % nprocs_per_jacobi_block);

  // Check if the number of lines (or columns) of the matrix resulting from discretization is divisible by the total number of processes
  *n_mesh_points = n_mesh_lines * n_mesh_columns;
  *jacobi_block_size = (*n_mesh_points) / (*njacobi_blocks);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode inner_solver(KSP ksp, Mat *A_block_jacobi_subMat, Vec *x_block_jacobi, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt *inner_solver_iterations)
{

  PetscFunctionBeginUser;
  Vec local_right_side_vector = NULL, mat_mult_vec_result = NULL;
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
  PetscCall(VecCopy(b_block_jacobi[rank_jacobi_block], local_right_side_vector));
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

  PetscInt idx = (rank_jacobi_block == ZERO ? ONE : ZERO);
  PetscCall(MatMult(A_block_jacobi_subMat[idx], x_block_jacobi[idx], mat_mult_vec_result));
  PetscCall(VecAXPY(local_right_side_vector, -1, mat_mult_vec_result));

  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSolve(ksp, local_right_side_vector, x_block_jacobi[rank_jacobi_block]));
  PetscInt n_iterations = 0;
  PetscCall(KSPGetIterationNumber(ksp, &n_iterations));
  if (rank_jacobi_block == 0)
  {
    MPI_Comm tmp;
    PetscCall(PetscObjectGetComm((PetscObject)local_right_side_vector, &tmp));
    PetscCall(PetscPrintf(tmp, "NUMBER OF ITERATIONS : %d  ====== ", n_iterations));
  }

  if (inner_solver_iterations != NULL)
  {
    *inner_solver_iterations = n_iterations;
  }

  PetscCall(VecDestroy(&local_right_side_vector));
  PetscCall(VecDestroy(&mat_mult_vec_result));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode poisson3DMatrix(Mat *A_block_jacobi, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block, PetscInt njacobi_blocks)
{
  PetscFunctionBeginUser;

  PetscInt Idx_start = 0, Idx_end = 0;
  PetscCall(MatGetOwnershipRange(*A_block_jacobi, &Idx_start, &Idx_end));

  PetscInt rowBlockSize = (n_grid_lines * n_grid_columns) / njacobi_blocks;
  // PetscInt columnBlockSize = n_grid_lines * n_grid_columns;

  PetscInt i, j, J;
  PetscScalar v;
  PetscInt Ii_new;

  for (PetscInt Ii = (rank_jacobi_block * rowBlockSize) + Idx_start; Ii < (rank_jacobi_block * rowBlockSize) + Idx_end; Ii++)
  {
    v = -1.0, i = Ii / n_grid_columns, j = Ii - i * n_grid_columns;
    Ii_new = Ii - (rank_jacobi_block * rowBlockSize);
    if (i > 0)
    {
      J = Ii - n_grid_columns;
      PetscCall(MatSetValue(*A_block_jacobi, Ii_new, J, v, INSERT_VALUES));
    }
    if (i < n_grid_lines - 1)
    {
      J = Ii + n_grid_columns;
      PetscCall(MatSetValue(*A_block_jacobi, Ii_new, J, v, INSERT_VALUES));
    }
    if (j > 0)
    {
      J = Ii - 1;
      PetscCall(MatSetValue(*A_block_jacobi, Ii_new, J, v, INSERT_VALUES));
    }
    if (j < n_grid_columns - 1)
    {
      J = Ii + 1;
      PetscCall(MatSetValue(*A_block_jacobi, Ii_new, J, v, INSERT_VALUES));
    }
    v = 4.0;
    PetscCall(MatSetValue(*A_block_jacobi, Ii_new, Ii, v, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(*A_block_jacobi, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A_block_jacobi, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printElapsedTime(double start_time, double end_time)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Elapsed time:   %f  seconds \n", end_time - start_time));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printResidualNorm(PetscScalar approximation_residual_infinity_norm)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Infinity norm of residual ==== %g \n", approximation_residual_infinity_norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}


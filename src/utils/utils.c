#include "constants.h"
#include "utils.h"
#include <petscts.h>
#include <petscdmda.h>

PetscErrorCode PetscArrayfill(PetscInt *x, PetscScalar val, PetscInt n)
{
  PetscFunctionBeginUser;
  PetscInt i;
  for (i = 0; i < n; i++)
  {
    x[i] = val;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode poisson3DMatrix(Mat *A_block_jacobi, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt n_grid_depth, PetscInt rank_jacobi_block, PetscInt njacobi_blocks)
{
  PetscFunctionBeginUser;

  PetscInt i, j, k;
  PetscInt row;
  PetscInt global_row;
  PetscInt z_start = 0, z_end = 0;

  PetscInt previous_lines = 0;
  PetscScalar v[7]; // Stencil values

  if (rank_jacobi_block == BLOCK_RANK_ZERO)
  {
    z_start = 0;
    z_end = n_grid_columns / 2;
    previous_lines = 0;
  }

  if (rank_jacobi_block == BLOCK_RANK_ONE)
  {
    z_start = n_grid_columns / 2;
    z_end = n_grid_columns;
    previous_lines = ((n_grid_lines * n_grid_columns * n_grid_depth) / 2);
  }

  // Fill the matrix
  for (k = z_start; k < z_end; k++)
  {
    for (j = 0; j < n_grid_columns; j++)
    {
      for (i = 0; i < n_grid_lines; i++)
      {
        row = i + (j * n_grid_lines) + (k * n_grid_lines * n_grid_columns);
        PetscInt ncols = 0;
        PetscInt cols[7];

        // Center point
        v[ncols] = 6.0;
        cols[ncols++] = row;

        // X-direction neighbors
        if (i > 0)
        {
          v[ncols] = -1.0;
          cols[ncols++] = row - 1;
        }

        if (i < n_grid_lines - 1)
        {
          v[ncols] = -1.0;
          cols[ncols++] = row + 1;
        }

        // Y-direction neighbors
        if (j > 0)
        {
          v[ncols] = -1.0;
          cols[ncols++] = row - n_grid_lines;
        }

        // Top neighbor
        if (j < n_grid_columns - 1)
        {
          v[ncols] = -1.0;
          cols[ncols++] = row + n_grid_lines;
        }

        // Z-direction neighbors
        if (k > 0)
        {
          v[ncols] = -1.0;
          cols[ncols++] = row - (n_grid_lines * n_grid_columns);
        }
        if (k < n_grid_depth - 1)
        {
          v[ncols] = -1.0;
          cols[ncols++] = row + (n_grid_lines * n_grid_columns);
        }

        global_row = i + (j * n_grid_lines) + (k * n_grid_lines * n_grid_columns) - previous_lines;
        PetscCall(MatSetValues(*A_block_jacobi, 1, &global_row, ncols, cols, v, INSERT_VALUES));
      }
    }
  }

  // Assemble the matrix
  PetscCall(MatAssemblyBegin(*A_block_jacobi, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A_block_jacobi, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode create_matrix_dense(MPI_Comm comm, Mat *mat, PetscInt n, PetscInt m, MatType mat_type)
{
  PetscFunctionBeginUser;

  PetscCall(MatCreate(comm, mat));
  PetscCall(MatSetType(*mat, mat_type));
  PetscCall(MatSetSizes(*mat, PETSC_DECIDE, PETSC_DECIDE, n, m));
  PetscCall(MatSetFromOptions(*mat));
  // PetscCall(MatSetOption(*mat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  // PetscCall(MatSetOption(*mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));

  //  PetscCall(MatSetUp(*mat));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode create_matrix_sparse(MPI_Comm comm, Mat *mat, PetscInt n, PetscInt m, MatType mat_type, PetscInt d_nz, PetscInt o_nz)
{
  PetscFunctionBeginUser;

  PetscCall(MatCreate(comm, mat));
  PetscCall(MatSetType(*mat, mat_type));
  PetscCall(MatSetSizes(*mat, PETSC_DECIDE, PETSC_DECIDE, n, m));
  PetscCall(MatSetFromOptions(*mat));
  PetscCall(MatSeqAIJSetPreallocation(*mat, d_nz, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*mat, d_nz, NULL, o_nz, NULL));
  // PetscCall(MatSetOption(*mat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  // PetscCall(MatSetOption(*mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));

  //  PetscCall(MatSetUp(*mat));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode create_vector(MPI_Comm comm, Vec *vec, PetscInt n, VecType vec_type)
{
  PetscFunctionBeginUser;

  PetscCall(VecCreate(comm, vec));
  PetscCall(VecSetSizes(*vec, PETSC_DECIDE, n));
  PetscCall(VecSetType(*vec, vec_type));
  PetscCall(VecSetFromOptions(*vec));
  // PetscCall(VecSetUp(*vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode poisson2DMatrix_old(Mat *A_block_jacobi, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block, PetscInt njacobi_blocks)
{

  PetscFunctionBeginUser;
  PetscInt i, j, row, global_row;
  PetscInt ystart = 0;
  PetscInt yend = 0;
  PetscScalar v[5]; // Stencil values
  PetscInt previous_lines = 0;

  if (rank_jacobi_block == BLOCK_RANK_ZERO)
  {
    ystart = 0;
    yend = n_grid_columns / 2;
    previous_lines = 0;
  }

  if (rank_jacobi_block == BLOCK_RANK_ONE)
  {
    ystart = n_grid_columns / 2;
    yend = n_grid_columns;
    previous_lines = ((n_grid_lines * n_grid_columns) / 2);
  }

  // Fill the matrix
  for (j = ystart; j < yend; j++)
  {
    for (i = 0; i < n_grid_lines; i++)
    {
      row = (j * n_grid_lines + i);
      PetscInt ncols = 0;
      PetscInt cols[5];

      // Center point
      v[ncols] = 4.0;
      cols[ncols++] = row;

      // Left neighbor
      if (i > 0)
      {
        v[ncols] = -1.0;
        cols[ncols++] = row - 1;
      }

      // Right neighbor
      if (i < n_grid_lines - 1)
      {
        v[ncols] = -1.0;
        cols[ncols++] = row + 1;
      }

      // Bottom neighbor
      if (j > 0)
      {
        v[ncols] = -1.0;
        cols[ncols++] = row - n_grid_lines;
      }

      // Top neighbor
      if (j < n_grid_columns - 1)
      {
        v[ncols] = -1.0;
        cols[ncols++] = row + n_grid_lines;
      }

      global_row = (j * n_grid_lines + i) - previous_lines;
      PetscCall(MatSetValues(*A_block_jacobi, 1, &global_row, ncols, cols, v, INSERT_VALUES));
    }
  }

  // Assemble the matrix
  PetscCall(MatAssemblyBegin(*A_block_jacobi, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A_block_jacobi, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}

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
// TODO: revoir ce code juste pour verifier la decoupe correspond a la logique mathematique

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

  // PetscCall(KSPSetNormType(*ksp,KSP_NORM_UNPRECONDITIONED));
  // PetscCall(KSPSetPCSide(*ksp, PC_RIGHT));

  PetscCall(KSPSetInitialGuessNonzero(*ksp, PetscNot(zero_initial_guess)));

  PetscCall(PCSetFromOptions(pc));
  PetscCall(KSPSetFromOptions(*ksp));

  // PetscCall(PCSetUp(pc));
  // PetscCall(KSPSetUp(*ksp));

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

PetscErrorCode computeFinalResidualNorm_new(Mat A_block_jacobi, Vec *x, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt proc_local_rank, PetscScalar *direct_residual_norm)
{
  PetscFunctionBegin;
  Vec direct_local_residual = NULL;
  PetscScalar direct_local_residual_norm2 = PETSC_MAX_REAL;
  PetscCall(VecDuplicate(*b_block_jacobi, &direct_local_residual));
  PetscCall(MatResidual(A_block_jacobi, *b_block_jacobi, *x, direct_local_residual));
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

PetscErrorCode printElapsedTime(double start_time, double end_time)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Elapsed time (iterations):   %f  seconds \n", end_time - start_time));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printResidualNorm(MPI_Comm comm_jacobi_block, PetscInt rank_jacobi_block, PetscScalar approximation_residual_infinity_norm, PetscInt outer_iteration_number)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(comm_jacobi_block, "[ Block rank %d ][ outer iter %d] --------------- Infinity norm of residual (Xk - Xk-1) = %e \n", rank_jacobi_block, outer_iteration_number, approximation_residual_infinity_norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printResidualNorm1(MPI_Comm comm_jacobi_block, PetscInt rank_jacobi_block, PetscScalar approximation_residual_infinity_norm, PetscInt outer_iteration_number)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(comm_jacobi_block, "[ Block rank %d ][ inner iter %d] --------------- Infinity norm of residual (Xk - Xk-1) = %e \n", rank_jacobi_block, outer_iteration_number, approximation_residual_infinity_norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printResidualNorm_no_data(PetscScalar approximation_residual_infinity_norm)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Infinity norm of residual (Xk - Xk-1) [no new data]= %e \n", approximation_residual_infinity_norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printFinalResidualNorm(PetscScalar global_residual_norm)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Final residual norm 2 = %e \n", global_residual_norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printTotalNumberOfIterations(MPI_Comm comm_jacobi_block, PetscInt rank_jacobi_block, PetscInt iterations)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(comm_jacobi_block, "[ Block rank %d ] Total number of iterations (outer_iterations) = %d \n", rank_jacobi_block, iterations));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printInnerSolverIterations(MPI_Comm comm_jacobi_block, PetscInt rank_jacobi_block, PetscInt iterations, PetscInt outer_iteration_number)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(comm_jacobi_block, "[ Block rank %d ][ outer iter %d] NUMBER OF INNER SOLVER ITERATIONS = %d  \n", rank_jacobi_block, outer_iteration_number, iterations));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printOuterSolverIterations(MPI_Comm comm_jacobi_block, PetscInt rank_jacobi_block, PetscInt iterations, PetscInt outer_iteration_number)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(comm_jacobi_block, "[ Block rank %d ][ outer iter %d] --------------- NUMBER OF OUTER SOLVER ITERATIONS = %d  \n", rank_jacobi_block, outer_iteration_number, iterations));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printTotalNumberOfIterations_2(MPI_Comm comm_jacobi_block, PetscInt rank_jacobi_block, PetscInt iterations, PetscInt s)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(comm_jacobi_block, "[ Block rank %d ] Total number of iterations (outer_iterations * s) = %d * %d = %d \n", rank_jacobi_block, iterations, s, s * iterations));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode exchange_R_block_jacobi_old(Mat R, Mat *R_block_jacobi_subMat, PetscInt s, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block, PetscInt njacobi_blocks, PetscInt proc_local_rank, PetscInt idx_non_current_block, PetscInt nprocs_per_jacobi_block)
{
  PetscFunctionBegin;

  PetscScalar *local_values = NULL;
  PetscScalar *remote_values = NULL;
  PetscScalar *local_values_buffer = NULL;
  PetscScalar *remote_values_buffer = NULL;
  PetscInt local_size;
  PetscCall(MatGetLocalSize(R_block_jacobi_subMat[rank_jacobi_block], &local_size, NULL));
  PetscInt nvalues = s * local_size;

  PetscCall(PetscMalloc1(nvalues, &local_values_buffer));
  PetscCall(PetscMalloc1(nvalues, &remote_values_buffer));

  // Get submat
  PetscCall(getHalfSubMatrixFromR(R, R_block_jacobi_subMat, n_grid_lines, n_grid_columns, rank_jacobi_block));
  PetscCall(MatDenseGetArrayWrite(R_block_jacobi_subMat[rank_jacobi_block], &local_values));
  PetscCall(PetscArraycpy(local_values_buffer, local_values, nvalues));
  PetscCall(MatDenseRestoreArrayWrite(R_block_jacobi_subMat[rank_jacobi_block], &local_values));
  PetscCall(restoreHalfSubMatrixToR(R, R_block_jacobi_subMat, rank_jacobi_block));
  // restore submat

  if (nvalues > ZERO)
  {
    if (rank_jacobi_block == BLOCK_RANK_ZERO)
    {
      PetscCallMPI(MPI_Send(local_values_buffer, nvalues, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 0, MPI_COMM_WORLD));
      PetscCallMPI(MPI_Recv(remote_values_buffer, nvalues, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    else if (rank_jacobi_block == BLOCK_RANK_ONE)
    {
      PetscCallMPI(MPI_Recv(remote_values_buffer, nvalues, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      PetscCallMPI(MPI_Send(local_values_buffer, nvalues, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 1, MPI_COMM_WORLD));
    }
  }

  // Get submat
  PetscCall(getHalfSubMatrixFromR(R, R_block_jacobi_subMat, n_grid_lines, n_grid_columns, idx_non_current_block));
  PetscCall(MatDenseGetArrayWrite(R_block_jacobi_subMat[idx_non_current_block], &remote_values));
  PetscCall(PetscArraycpy(remote_values, remote_values_buffer, nvalues));
  PetscCall(MatDenseRestoreArrayWrite(R_block_jacobi_subMat[idx_non_current_block], &remote_values));
  PetscCall(restoreHalfSubMatrixToR(R, R_block_jacobi_subMat, idx_non_current_block));

  // restore submat

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode exchange_R_block_jacobi(Mat R, Mat *R_block_jacobi_subMat, PetscInt s, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block, PetscInt njacobi_blocks, PetscInt proc_local_rank, PetscInt idx_non_current_block, PetscInt nprocs_per_jacobi_block)
{
  PetscFunctionBegin;

  PetscScalar *local_values = NULL;
  PetscScalar *remote_values = NULL;
  PetscScalar *local_values_buffer = NULL;
  PetscScalar *remote_values_buffer = NULL;
  PetscInt local_size;
  PetscCall(MatGetLocalSize(R, &local_size, NULL));
  PetscInt nvalues = s * local_size;

  PetscCall(PetscMalloc1(nvalues, &local_values_buffer));
  PetscCall(PetscMalloc1(nvalues, &remote_values_buffer));

  if (rank_jacobi_block == BLOCK_RANK_ZERO)
  {
    if (proc_local_rank < (nprocs_per_jacobi_block / 2))
    {
      PetscCall(MatDenseGetArrayWrite(R, &local_values));
      PetscCall(PetscArraycpy(local_values_buffer, local_values, nvalues));
      PetscCall(MatDenseRestoreArrayWrite(R, &local_values));
      PetscCallMPI(MPI_Send(local_values_buffer, nvalues, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 0, MPI_COMM_WORLD));
    }
    else
    {
      PetscCallMPI(MPI_Recv(remote_values_buffer, nvalues, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      PetscCall(MatDenseGetArrayWrite(R, &remote_values));
      PetscCall(PetscArraycpy(remote_values, remote_values_buffer, nvalues));
      PetscCall(MatDenseRestoreArrayWrite(R, &remote_values));
    }
  }
  else if (rank_jacobi_block == BLOCK_RANK_ONE)
  {
    if (proc_local_rank < (nprocs_per_jacobi_block / 2))
    {
      PetscCallMPI(MPI_Recv(remote_values_buffer, nvalues, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      PetscCall(MatDenseGetArrayWrite(R, &remote_values));
      PetscCall(PetscArraycpy(remote_values, remote_values_buffer, nvalues));
      PetscCall(MatDenseRestoreArrayWrite(R, &remote_values));
    }
    else
    {
      PetscCall(MatDenseGetArrayWrite(R, &local_values));
      PetscCall(PetscArraycpy(local_values_buffer, local_values, nvalues));
      PetscCall(MatDenseRestoreArrayWrite(R, &local_values));
      PetscCallMPI(MPI_Send(local_values_buffer, nvalues, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 1, MPI_COMM_WORLD));
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode fillArrayWithIncrement(PetscInt *array, PetscInt size, PetscInt start, PetscInt increment)
{
  PetscFunctionBegin;

  for (PetscInt i = 0; i < size; i++)
  {
    array[i] = start + (i * increment);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode divideRintoSubMatrices(MPI_Comm comm_jacobi_block, Mat R, Mat *R_block_jacobi, PetscInt rank_jacobi_block, PetscInt njacobi_blocks, PetscInt nprocs_per_jacobi_block, PetscInt proc_local_rank)
{

  PetscFunctionBegin;

  IS is_rows[njacobi_blocks];

  PetscInt nrows;
  PetscInt ncols;
  PetscCall(MatGetSize(R, &nrows, &ncols));

  PetscInt nrows_half = nrows / 2;

  PetscInt n = (nrows_half / nprocs_per_jacobi_block);
  PetscInt first = proc_local_rank * (nrows_half / nprocs_per_jacobi_block);
  PetscInt step = 1;

  for (PetscInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(ISCreateStride(comm_jacobi_block, n, (i * nrows_half) + first, step, &is_rows[i]));
    // PetscCall(ISCreateStride(comm_jacobi_block, nrows/nprocs_per_jacobi_block, proc_local_rank * (nrows / nprocs_per_jacobi_block), step, &is_rows[i]));
  }

  // Extract submatrices

  for (PetscInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(MatCreateSubMatrix(R, is_rows[i], NULL, MAT_INITIAL_MATRIX, &R_block_jacobi[i]));
    PetscCall(MatZeroEntries(R_block_jacobi[i]));
  }

  PetscCall(MatGetSize(R_block_jacobi[0], &nrows, &ncols));
  printf("rank jacobi block %d nrows %d ncols %d \n", rank_jacobi_block, nrows, ncols);

  // Clean up
  for (PetscInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(ISDestroy(&is_rows[i]));
  }

  PetscCall(MatDestroy(&R_block_jacobi[0]));
  PetscCall(MatDestroy(&R_block_jacobi[1]));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode create_redistributed_A_block_jacobi(MPI_Comm comm_jacobi_block, Mat A_block_jacobi, Mat *A_block_jacobi_redist, PetscInt nprocs_per_jacobi_block, PetscInt proc_local_rank, PetscInt proc_local_size, PetscInt first_row_owned)
{

  PetscFunctionBegin;

  IS isrows;
  PetscInt length_local_own_portion = ZERO;
  PetscInt first = ZERO;
  PetscInt step = ZERO;

  if (proc_local_size > ZERO)
  {
    length_local_own_portion = proc_local_size;
    // first = (proc_local_size * proc_local_rank);
    first = first_row_owned;
    step = ONE;
  }

  // printf("rank %d length_local_own_portion %d first %d step %d\n", proc_local_rank, length_local_own_portion, first, step);

  PetscCall(ISCreateStride(comm_jacobi_block, length_local_own_portion, first, step, &isrows));

  // ISView(isrows, PETSC_VIEWER_STDOUT_(comm_jacobi_block));

  PetscCall(MatCreateSubMatrix(A_block_jacobi, isrows, NULL, MAT_INITIAL_MATRIX, A_block_jacobi_redist));

  // MatView(*A_block_jacobi_redist, PETSC_VIEWER_STDOUT_(comm_jacobi_block));

  PetscCall(ISDestroy(&isrows));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode getHalfSubMatrixFromR(Mat R, Mat *R_block_jacobi_subMat, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block)
{
  PetscFunctionBegin;
  PetscInt idx_first_row = rank_jacobi_block * ((n_grid_lines * n_grid_columns) / 2);
  PetscInt idx_one_plus_last_row = (rank_jacobi_block + 1) * ((n_grid_lines * n_grid_columns) / 2);
  if (rank_jacobi_block == 0)
    PetscCall(MatDenseGetSubMatrix(R, idx_first_row, idx_one_plus_last_row, PETSC_DECIDE, PETSC_DECIDE, &R_block_jacobi_subMat[rank_jacobi_block]));
  if (rank_jacobi_block == 1)
    PetscCall(MatDenseGetSubMatrix(R, idx_first_row, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, &R_block_jacobi_subMat[rank_jacobi_block]));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode restoreHalfSubMatrixToR(Mat R, Mat *R_block_jacobi_subMat, PetscInt rank_jacobi_block)
{
  PetscFunctionBegin;
  PetscCall(MatDenseRestoreSubMatrix(R, &R_block_jacobi_subMat[rank_jacobi_block]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode updateLocalRHS(Vec local_right_side_vector, Mat *A_block_jacobi_subMat, Vec *x_block_jacobi, Vec *b_block_jacobi, Vec mat_mult_vec_result, PetscMPIInt rank_jacobi_block)
{
  PetscFunctionBegin;
  PetscInt idx_non_current_block = (rank_jacobi_block == ZERO ? ONE : ZERO);

  PetscCall(VecCopy(b_block_jacobi[rank_jacobi_block], local_right_side_vector));

  PetscCall(MatMult(A_block_jacobi_subMat[idx_non_current_block], x_block_jacobi[idx_non_current_block], mat_mult_vec_result));

  PetscCall(VecWAXPY(local_right_side_vector, -1.0, mat_mult_vec_result, b_block_jacobi[rank_jacobi_block]));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode inner_solver(MPI_Comm comm_jacobi_block, KSP ksp, Mat *A_block_jacobi_subMat, Vec *x_block_jacobi, Vec *b_block_jacobi, Vec local_right_side_vector, PetscInt rank_jacobi_block, PetscInt *inner_solver_iterations, PetscInt outer_iteration_number)
{

  PetscFunctionBeginUser;

  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSolve(ksp, local_right_side_vector, x_block_jacobi[rank_jacobi_block]));
  PetscInt n_iterations = 0;
  PetscCall(KSPGetIterationNumber(ksp, &n_iterations));

  PetscCall(printInnerSolverIterations(comm_jacobi_block, rank_jacobi_block, n_iterations, outer_iteration_number));

  if (inner_solver_iterations != NULL)
  {
    *inner_solver_iterations = n_iterations;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode outer_solver(MPI_Comm comm_jacobi_block, KSP outer_ksp, Vec x_minimized, Mat R, Mat S, Mat R_transpose_R, Vec vec_R_transpose_b_block_jacobi, Vec alpha, Vec b, PetscInt rank_jacobi_block, PetscInt s, PetscInt outer_iteration_number)
{

  PetscFunctionBegin;

  PetscCall(MatTransposeMatMult(R, R, MAT_REUSE_MATRIX, PETSC_DETERMINE, &R_transpose_R));

  PetscCall(MatMultTranspose(R, b, vec_R_transpose_b_block_jacobi));

  PetscCall(KSPSetOperators(outer_ksp, R_transpose_R, R_transpose_R));

  // PetscCall(KSPSetInitialGuessNonzero(*outer_ksp, PETSC_TRUE));

  PetscCall(KSPSolve(outer_ksp, vec_R_transpose_b_block_jacobi, alpha));

  PetscInt n_iterations = 0;
  PetscCall(KSPGetIterationNumber(outer_ksp, &n_iterations));

  PetscCall(printOuterSolverIterations(comm_jacobi_block, rank_jacobi_block, n_iterations, outer_iteration_number));

  PetscCall(MatMult(S, alpha, x_minimized));

  // if(rank_jacobi_block == 0){
  //   PetscCall(VecView(alpha, PETSC_VIEWER_STDOUT_(comm_jacobi_block)));
  // }

  PetscFunctionReturn(PETSC_SUCCESS);
}

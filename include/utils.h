
#ifndef SHARED_FUNCTIONS_H
#define SHARED_FUNCTIONS_H
#include <petscts.h>


PetscErrorCode create_matrix_dense(MPI_Comm comm, Mat *mat, PetscInt n, PetscInt m, MatType mat_type);
PetscErrorCode create_matrix_sparse(MPI_Comm comm, Mat *mat, PetscInt n, PetscInt m, MatType mat_type, PetscInt d_nz, PetscInt o_nz);

PetscErrorCode create_vector(MPI_Comm comm, Vec *vec, PetscInt n, VecType vec_type);

PetscErrorCode poisson2DMatrix(Mat *A_block_jacobi, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block, PetscInt njacobi_blocks);

PetscErrorCode poisson2DMatrix_old(Mat *A_block_jacobi, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block, PetscInt njacobi_blocks) __attribute__((deprecated("Use new_function instead")));

PetscErrorCode poisson3DMatrix(Mat *A_block_jacobi, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt n_grid_depth, PetscInt rank_jacobi_block, PetscInt njacobi_blocks);

PetscErrorCode divideSubDomainIntoBlockMatrices(MPI_Comm comm_jacobi_block, Mat A_block_jacobi, Mat *A_block_jacobi_subMat, IS *is_cols_block_jacobi, PetscInt rank_jacobi_block, PetscInt njacobi_blocks, PetscInt proc_local_rank, PetscInt nprocs_per_jacobi_block);

PetscErrorCode initializeKSP(MPI_Comm comm_jacobi_block, KSP *ksp, Mat operator_matrix, PetscScalar rank_jacobi_block, PetscBool zero_initial_guess, const char *ksp_prefix, const char *pc_prefix);



PetscErrorCode computeFinalResidualNorm(Mat A_block_jacobi, Vec *x, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt proc_local_rank, PetscScalar *direct_residual_norm);

PetscErrorCode computeFinalResidualNorm_new(Mat A_block_jacobi, Vec *x, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt proc_local_rank, PetscScalar *direct_residual_norm);

PetscErrorCode computeTheRightHandSideWithInitialGuess(MPI_Comm comm_jacobi_block, VecScatter *scatter_jacobi_vec_part_to_merged_vec, Mat A_block_jacobi, Vec *b, Vec *b_block_jacobi, Vec x_initial_guess, PetscInt rank_jacobi_block, PetscInt jacobi_block_size, PetscInt nprocs_per_jacobi_block, PetscInt proc_local_rank);

PetscErrorCode computeDimensionRelatedVariables(PetscInt nprocs, PetscInt nprocs_per_jacobi_block, PetscInt proc_global_rank, PetscInt n_mesh_lines, PetscInt n_mesh_columns,
                                                PetscInt *njacobi_blocks, PetscInt *rank_jacobi_block, PetscInt *proc_local_rank, PetscInt *n_mesh_points, PetscInt *jacobi_block_size);

PetscErrorCode printElapsedTime(double start_time, double end_time);


PetscErrorCode printResidualNorm(MPI_Comm comm_jacobi_block,PetscInt rank_jacobi_block, PetscScalar approximation_residual_infinity_norm,PetscInt outer_iteration_number);

PetscErrorCode printOuterSolverIterations(MPI_Comm comm_jacobi_block, PetscInt rank_jacobi_block, PetscInt iterations, PetscInt outer_iteration_number);


PetscErrorCode printResidualNorm_no_data(PetscScalar approximation_residual_infinity_norm);

PetscErrorCode printFinalResidualNorm(PetscScalar global_residual_norm);

PetscErrorCode printTotalNumberOfIterations(MPI_Comm comm_jacobi_block, PetscInt rank_jacobi_block, PetscInt iterations);

PetscErrorCode printTotalNumberOfIterations_2(MPI_Comm comm_jacobi_block, PetscInt rank_jacobi_block, PetscInt iterations, PetscInt s);

PetscErrorCode exchange_R_block_jacobi(Mat R, Mat *R_block_jacobi_subMat, PetscInt s, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block, PetscInt njacobi_blocks, PetscInt proc_local_rank,PetscInt idx_non_current_block ,PetscInt nprocs_per_jacobi_block);

PetscErrorCode divideRintoSubMatrices(MPI_Comm comm_jacobi_block, Mat R, Mat *R_block_jacobi, PetscInt rank_jacobi_block, PetscInt njacobi_blocks, PetscInt nprocs_per_jacobi_block, PetscInt proc_local_rank);

// PetscErrorCode fillArrayWithIncrement(int *array, int size, int start, int increment);

PetscErrorCode create_redistributed_A_block_jacobi(MPI_Comm comm_jacobi_block, Mat A_block_jacobi, Mat *A_block_jacobi_redist, PetscInt nprocs_per_jacobi_block, PetscInt proc_local_rank, PetscInt proc_local_size, PetscInt idx_first_row_owned);


PetscErrorCode getHalfSubMatrixFromR(Mat R, Mat *R_block_jacobi_subMat, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block);

PetscErrorCode restoreHalfSubMatrixToR(Mat R, Mat *R_block_jacobi_subMat, PetscInt rank_jacobi_block);

PetscErrorCode printInnerSolverIterations(MPI_Comm comm_jacobi_block, PetscInt rank_jacobi_block, PetscInt iterations,PetscInt outer_iteration_number);





PetscErrorCode updateLocalRHS(Vec local_right_side_vector, Mat *A_block_jacobi_subMat,Vec *x_block_jacobi, Vec *b_block_jacobi, Vec mat_mult_vec_result, PetscMPIInt rank_jacobi_block);

PetscErrorCode inner_solver(MPI_Comm comm_jacobi_block,KSP ksp, Mat *A_block_jacobi_subMat, Vec *x_block_jacobi, Vec *b_block_jacobi, Vec local_right_side_vector, PetscInt rank_jacobi_block, PetscInt *inner_solver_iterations, PetscInt outer_iteration_number);


PetscErrorCode outer_solver(MPI_Comm comm_jacobi_block, KSP outer_ksp, Vec x_minimized, Mat R, Mat S, Mat R_transpose_R, Vec vec_R_transpose_b_block_jacobi, Vec alpha, Vec b, PetscInt rank_jacobi_block, PetscInt s, PetscInt outer_iteration_number);

#endif // SHARED_FUNCTIONS_H

#ifndef SHARED_FUNCTIONS_H
#define SHARED_FUNCTIONS_H
#include <petscts.h>


PetscErrorCode create_matrix(MPI_Comm comm,Mat * mat, PetscInt n , PetscInt m, MatType mat_type,PetscInt d_nz, PetscInt o_nz);

PetscErrorCode create_vector(MPI_Comm comm,Vec *vec, PetscInt n, VecType vec_type);

PetscErrorCode poisson2DMatrix(Mat *A_block_jacobi, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block, PetscInt njacobi_blocks);


PetscErrorCode poisson3DMatrix(Mat *A_block_jacobi, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt n_grid_depth, PetscInt rank_jacobi_block, PetscInt njacobi_blocks);

PetscErrorCode poisson2DMatrix_old(Mat *A_block_jacobi, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block, PetscInt njacobi_blocks) __attribute__((deprecated("Use new_function instead")));


PetscErrorCode divideSubDomainIntoBlockMatrices(MPI_Comm comm_jacobi_block, Mat A_block_jacobi, Mat *A_block_jacobi_subMat, IS *is_cols_block_jacobi, PetscInt rank_jacobi_block, PetscInt njacobi_blocks, PetscInt proc_local_rank, PetscInt nprocs_per_jacobi_block);

PetscErrorCode initializeKSP(MPI_Comm comm_jacobi_block, KSP *ksp, Mat operator_matrix, PetscScalar rank_jacobi_block, PetscBool zero_initial_guess, const char *ksp_prefix, const char *pc_prefix);

PetscErrorCode updateKSPoperators(KSP *ksp, Mat operator_matrix);

PetscErrorCode inner_solver(KSP ksp, Mat *A_block_jacobi_subMat, Vec *x_block_jacobi, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt *inner_solver_iterations);

PetscErrorCode outer_solver(MPI_Comm comm_jacobi_block, KSP *outer_ksp, Vec x_minimized, Mat R, Mat S, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt s);

PetscErrorCode computeFinalResidualNorm(Mat A_block_jacobi, Vec *x, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt proc_local_rank, PetscScalar *direct_residual_norm);

PetscErrorCode computeFinalResidualNorm_new(Mat A_block_jacobi, Vec *x, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt proc_local_rank, PetscScalar *direct_residual_norm);


PetscErrorCode computeTheRightHandSideWithInitialGuess(MPI_Comm comm_jacobi_block, VecScatter *scatter_jacobi_vec_part_to_merged_vec, Mat A_block_jacobi, Vec *b, Vec *b_block_jacobi, Vec x_initial_guess, PetscInt rank_jacobi_block, PetscInt jacobi_block_size, PetscInt nprocs_per_jacobi_block, PetscInt proc_local_rank);

PetscErrorCode computeDimensionRelatedVariables(PetscInt nprocs, PetscInt nprocs_per_jacobi_block, PetscInt proc_global_rank, PetscInt n_mesh_lines, PetscInt n_mesh_columns,
                                                PetscInt *njacobi_blocks, PetscInt *rank_jacobi_block, PetscInt *proc_local_rank, PetscInt *n_mesh_points, PetscInt *jacobi_block_size);


PetscErrorCode printElapsedTime(double start_time, double end_time);

PetscErrorCode printResidualNorm(PetscScalar approximation_residual_infinity_norm);

PetscErrorCode printFinalResidualNorm(PetscScalar global_residual_norm);

PetscErrorCode printTotalNumberOfIterations(PetscInt iterations);


   

#endif // SHARED_FUNCTIONS_H
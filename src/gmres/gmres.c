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

    Mat A = NULL; // Operator matrix
    Vec x = NULL; // approximation solution at iteration (k)
    Vec b = NULL; // right hand side vector
    Vec x_initial_guess = NULL;

    PetscMPIInt nprocs;
    PetscMPIInt proc_global_rank;
    PetscInt n_mesh_lines = 4;
    PetscInt n_mesh_columns = 4;
    PetscInt njacobi_blocks;
    PetscMPIInt rank_jacobi_block;
    PetscMPIInt proc_local_rank;
    PetscInt n_mesh_points;

    KSP inner_ksp;
    PetscScalar relative_tolerance = 1e-5;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

    PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &n_mesh_lines, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n_mesh_columns, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &relative_tolerance, NULL));

    PetscAssert((n_mesh_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &inner_ksp));
    PetscCall(KSPSetOperators(inner_ksp, A, A));
    PetscCall(KSPSetInitialGuessNonzero(inner_ksp, PETSC_TRUE));
    PetscCall(KSPSetFromOptions(inner_ksp));

    PetscCall(create_vector(PETSC_COMM_WORLD, &x, n_mesh_points, VECMPI));
    PetscCall(VecCreate(PETSC_COMM_WORLD, x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
    PetscCall(VecSetType(x, VECMPI));
    PetscCall(VecSetFromOptions(x));



    PetscCall(VecDuplicate(x, &x_initial_guess));
    PetscCall(VecDuplicate(x, &b));

    
    // PetscCall(VecSet(x_initial_guess, initial_scalar_value));

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

    PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, &b, b_block_jacobi, x_initial_guess, rank_jacobi_block, jacobi_block_size, nprocs_per_jacobi_block, proc_local_rank));

    PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
    PetscCall(PetscMalloc1(vec_local_size, &send_multisplitting_data_buffer));
    PetscCall(PetscMalloc1(vec_local_size, &rcv_multisplitting_data_buffer));

    PetscCall(VecDuplicate(x, &approximation_residual));
    PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
    PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

    // PetscCall(PetscFinalize());
    // return 0;

    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    double start_time, end_time;
    start_time = MPI_Wtime();

    do
    {
        PetscCall(VecCopy(x, x_previous_iteration));

        PetscCall(updateLocalRHS(local_right_side_vector, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, mat_mult_vec_result, rank_jacobi_block));
        PetscCall(inner_solver(comm_jacobi_block, inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, local_right_side_vector, rank_jacobi_block, NULL, number_of_iterations));

        PetscCall(comm_sync_send_and_receive(x_block_jacobi, vec_local_size, message_dest, message_source, rank_jacobi_block, idx_non_current_block));

        PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

        PetscCall(VecWAXPY(approximation_residual, -1.0, x_previous_iteration, x));
        PetscCall(VecNorm(approximation_residual, NORM_INFINITY, &approximation_residual_infinity_norm));

        PetscCall(printResidualNorm(comm_jacobi_block, rank_jacobi_block, approximation_residual_infinity_norm, number_of_iterations));

        if (PetscApproximateLTE(approximation_residual_infinity_norm, relative_tolerance))
        {
            stop_condition = PETSC_TRUE;
        }
        number_of_iterations = number_of_iterations + 1;

    } while (stop_condition == PETSC_FALSE);

    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    end_time = MPI_Wtime();

    PetscCall(printElapsedTime(start_time, end_time));
    PetscCall(printTotalNumberOfIterations(comm_jacobi_block, rank_jacobi_block, number_of_iterations));

    PetscScalar direct_residual_norm;
    PetscCall(computeFinalResidualNorm(A_block_jacobi, &x, b_block_jacobi, rank_jacobi_block, proc_global_rank, &direct_residual_norm));

    PetscCall(printFinalResidualNorm(direct_residual_norm));

    Vec check_solution = NULL;
    Vec solution = NULL;
    PetscCall(VecDuplicate(x, &check_solution));
    PetscCall(VecDuplicate(x, &solution));
    PetscCall(VecZeroEntries(check_solution));
    PetscCall(VecSet(solution, 1.0));
    PetscScalar check_solution_norm = 0.0;
    PetscCall(VecWAXPY(check_solution, -1.0, solution, x));
    PetscCall(VecNorm(check_solution, NORM_2, &check_solution_norm));
    if (rank_jacobi_block == 0)
    {
        PetscCall(PetscPrintf(comm_jacobi_block, "Norm equal : %e \n", check_solution_norm));
    }
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

    PetscCall(VecDestroy(&approximation_residual));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x_initial_guess));
    PetscCall(VecDestroy(&local_right_side_vector));
    PetscCall(VecDestroy(&mat_mult_vec_result));
    PetscCall(VecDestroy(&x_previous_iteration));
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

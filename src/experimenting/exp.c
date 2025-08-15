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
    Vec u = NULL; // Preset-solution

    PetscMPIInt nprocs;
    PetscMPIInt proc_global_rank;
    PetscInt n_mesh_lines = 4;
    PetscInt n_mesh_columns = 4;
    // PetscMPIInt proc_local_rank;
    PetscInt n_mesh_points;
    PetscInt number_of_iterations = 0;
    PetscScalar b_norm;
    PetscScalar residual_norm = 0.0;

    KSP ksp_context = NULL;
    PC pc_context = NULL;
    PetscScalar relative_tolerance = 1e-5;
    PetscScalar absolute_tolerance = 1e-50;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

    PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &n_mesh_lines, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n_mesh_columns, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &relative_tolerance, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-atol", &absolute_tolerance, NULL));

    n_mesh_points = n_mesh_lines * n_mesh_columns;
    PetscAssert((n_mesh_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");

    PetscCall(create_matrix_sparse(PETSC_COMM_WORLD, &A, n_mesh_points, n_mesh_points, MATMPIAIJ, 10, 10));
    PetscCall(poisson2DMatrix_complete(A, n_mesh_lines, n_mesh_columns));
    PetscCall(MatCreateVecs(A, &x, &b));
    // PetscCall(create_vector(PETSC_COMM_WORLD, &x, n_mesh_points, VECMPI));
    // PetscCall(VecDuplicate(x, &b));
    PetscCall(VecDuplicate(x, &u));
    PetscCall(VecSet(u, ONE));
    PetscCall(VecSet(x, ZERO));
    PetscCall(MatMult(A, u, b));

    PetscCall(initializeKSP(PETSC_COMM_WORLD, &ksp_context, A, 0, PETSC_TRUE, NULL, NULL));
    PetscCall(KSPGetPC(ksp_context, &pc_context));
    PetscCall(PCSetType(pc_context, PCBJACOBI));
    PetscInt *blks;
    PetscInt i;
    PetscInt TOTAL_BLOCK = 2;
    PetscCall(PetscMalloc1(2, &blks));
    for (i = 0; i < TOTAL_BLOCK; i++)
        blks[i] = n_mesh_points / 2;

    PetscCall(PCBJacobiSetTotalBlocks(pc_context, TOTAL_BLOCK, blks));
    PetscCall(PetscFree(blks));
    PetscCall(KSPSetFromOptions(ksp_context));
    PetscCall(PCSetUp(pc_context));
    PetscCall(KSPSetUp(ksp_context));

    KSP *subksp;
    PetscInt n_local, first_local;
    PetscCall(PCBJacobiGetSubKSP(pc_context, &n_local, &first_local, &subksp));
    PC subpc = NULL;

    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Number of blocks on proc %d : %d - block %d \n", proc_global_rank, n_local, first_local));

    PetscCall(KSPGetPC(subksp[0], &subpc));
    PetscCall(PCSetType(subpc, PCNONE));
    PetscCall(KSPSetType(subksp[0], KSPGMRES));
    PetscCall(KSPSetTolerances(subksp[0], 1.e-2, PETSC_CURRENT, PETSC_CURRENT, 50));
    PetscCall(KSPConvergedDefaultSetUIRNorm(subksp[0]));
    PetscCall(PCSetUp(subpc));
    PetscCall(KSPSetUp(subksp[0]));

    PetscCall(VecNorm(b, NORM_2, &b_norm));

    Vec local_rhs;
    Vec temp;
    // Vec local_solution;
    // PetscScalar local_rhs_norm;
    PetscInt size;

    PetscCall(VecCreate(PETSC_COMM_SELF, &local_rhs));
    PetscCall(VecSetType(local_rhs, VECMPI));
    PetscCall(VecSetSizes(local_rhs, PETSC_DECIDE, n_mesh_points / 2));
    PetscCall(VecSet(local_rhs, 1.0));
    PetscCall(VecAssemblyBegin(local_rhs));
    PetscCall(VecAssemblyEnd(local_rhs));
    PetscCall(KSPGetRhs(subksp[0], &temp));
    PetscCall(VecCopy(temp, local_rhs));
    PetscCall(VecGetSize(local_rhs, &size));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "SIZE : %d \n", size));
    if (local_rhs == NULL)
        printf("null ici\n");

    if (proc_global_rank == 0)
        PetscCall(VecView(local_rhs, PETSC_VIEWER_STDOUT_SELF));

    // PetscCall(KSPBuildSolution( ,NULL,&local_solution))
    return 0;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Start solving...\n"));
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    double start_time, end_time;
    start_time = MPI_Wtime();

    /* ################################################## */

    PetscCall(KSPSolve(ksp_context, b, x));

    /* ################################################## */

    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    end_time = MPI_Wtime();
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "End solving...\n\n\n"));
    PetscCall(printElapsedTime(start_time, end_time));

    // PetscCall(KSPView(subksp[first_local], PETSC_VIEWER_STDOUT_SELF));

    PetscCall(KSPGetIterationNumber(subksp[0], &number_of_iterations));
    PetscCall(KSPGetResidualNorm(subksp[0], &residual_norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "======================== \n"));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Number of iterations : %d \n", number_of_iterations));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Residual norm : %e \n", residual_norm));

    // Mat local_mat;
    // PetscCall(KSPGetOperators(subksp[0], &local_mat, NULL));
    // PetscCall(MatCreateVecs(local_mat, NULL, &local_rhs));

    // PetscCall(VecNorm(local_rhs, NORM_2, &local_rhs_norm));
    // PetscCall(PetscPrintf(PETSC_COMM_SELF, "Right hand side norm : %e \n", local_rhs_norm));
    // PetscCall(PetscPrintf(PETSC_COMM_SELF, "||r(i)||/||b|| : %e \n", residual_norm / local_rhs_norm));

    PetscScalar error;
    PetscCall(computeError(x, u, &error));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Erreur : %e \n", error));
    PetscCall(PetscFinalize());
    return 0;
    PetscCall(PetscSleep(100000));

    // PetscCall(KSPGetIterationNumber(ksp_context, &number_of_iterations));
    // PetscCall(KSPGetResidualNorm(ksp_context, &residual_norm));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "======================== \n"));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of iterations : %d \n", number_of_iterations));

    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Right hand side norm : %e \n", b_norm));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Residual norm : %e \n", residual_norm));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "||r(i)||/||b|| : %e \n", residual_norm / b_norm));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "======================== \n"));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\n"));

    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&u));
    PetscCall(KSPDestroy(&ksp_context));

    PetscCall(PetscFinalize());
    return 0;
}

// #endif



   // PetscCall(VecView(x_minimized, PETSC_VIEWER_STDOUT_WORLD));
    // PetscSleep(3);
    // PetscCall(VecView(x_minimized_prev_iterate, PETSC_VIEWER_STDOUT_WORLD));


    // Vec tmp;
    // Vec tmp_diff;
    // PetscScalar tmp_diff_norm = 0.0;
    // PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &tmp));
    // PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &tmp_diff));
    // PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_minimized_prev_iterate, tmp, INSERT_VALUES, SCATTER_REVERSE));
    // PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_minimized_prev_iterate, tmp, INSERT_VALUES, SCATTER_REVERSE));
    // // PetscCall(PetscSleep(2));
    // // PetscCall(VecView(tmp, PETSC_VIEWER_STDOUT_WORLD));
    // PetscCall(VecWAXPY(tmp_diff, -1.0, x_block_jacobi[rank_jacobi_block], tmp));
    // PetscCall(VecNorm(tmp_diff, NORM_INFINITY, &tmp_diff_norm));
    // PetscCall(PetscPrintf(PETSC_COMM_SELF, "Inf norm local : %e \n", tmp_diff_norm));
    //






    //     PetscViewer viewer;
    // if (rank_jacobi_block == 0)
    // {

    //   PetscViewerCreate(comm_jacobi_block, &viewer);
    //   PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
    //   PetscViewerSetType(viewer, PETSCVIEWERASCII);
    //   PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
    //   PetscViewerFileSetName(viewer, "matrix_S_2_0.m");
    //   MatView(S, viewer);
    //   PetscViewerDestroy(&viewer);
    // }

    // if (rank_jacobi_block == 1)
    // {

    //   PetscViewerCreate(comm_jacobi_block, &viewer);
    //   PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
    //   PetscViewerSetType(viewer, PETSCVIEWERASCII);
    //   PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
    //   PetscViewerFileSetName(viewer, "matrix_S_2_1.m");
    //   MatView(S, viewer);
    //   PetscViewerDestroy(&viewer);
    // }

    // PetscCall(PetscFinalize());
    // return 0;
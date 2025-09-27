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
    Vec u = NULL;

    PetscMPIInt nprocs;
    PetscMPIInt proc_global_rank;
    PetscInt n_mesh_lines = 4;
    PetscInt n_mesh_columns = 4;
    // PetscMPIInt proc_local_rank;
    PetscInt n_mesh_points;
    PetscInt number_of_iterations;
    PetscScalar b_norm;
    PetscScalar gmres_residual_norm;

    KSP ksp_context = NULL;
    PetscScalar relative_tolerance = 1e-5;
    PetscScalar absolute_tolerance = 1e-100;

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

    PetscCall(create_matrix_sparse(PETSC_COMM_WORLD, &A, n_mesh_points, n_mesh_points, MATMPIAIJ, 5, 5));
    PetscCall(poisson2DMatrix_complete(A, n_mesh_lines, n_mesh_columns));
    PetscCall(MatCreateVecs(A, &x, &b));
    PetscCall(VecDuplicate(x, &u));
    PetscCall(VecSet(u, ONE));
    PetscCall(VecSet(x, ZERO));
    PetscCall(MatMult(A, u, b));

    PetscCall(initializeKSP(PETSC_COMM_WORLD, &ksp_context, A, 0, PETSC_TRUE, NULL, NULL));
    PetscCall(VecNorm(b, NORM_2, &b_norm));

    ////
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Start solving...\n"));
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    double start_time, end_time;
    start_time = MPI_Wtime();

    PetscLogStage solving_stage;
    PetscCall(PetscLogStageRegister("Solving stage", &solving_stage));
    PetscCall(PetscLogStagePush(solving_stage));
    PetscCall(KSPSolve(ksp_context, b, x));
    PetscCall(PetscLogStagePop());

    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    end_time = MPI_Wtime();
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "End solving...\n\n\n"));
    PetscCall(printElapsedTime(start_time, end_time));

    PetscCall(KSPGetIterationNumber(ksp_context, &number_of_iterations));
    PetscCall(KSPGetResidualNorm(ksp_context, &gmres_residual_norm));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "======================== \n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of iterations of GMRES : %d \n", number_of_iterations));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Right hand side norm : %e \n", b_norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "GMRES residual norm : %e \n", gmres_residual_norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "||r(i)||/||b|| : %e \n", gmres_residual_norm / b_norm));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "======================== \n"));

    PetscScalar error;
    PetscCall(computeError(x, u, &error));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Erreur : %e \n", error));

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

///////

// PetscViewer viewer;
// Mat A_dense;
// MatConvert(A, MATMPIDENSE, MAT_INITIAL_MATRIX, &A_dense);
// PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
// PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
// PetscViewerSetType(viewer, PETSCVIEWERASCII);
// PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
// PetscViewerFileSetName(viewer, "matrix_A_512.m");
// MatView(A_dense, viewer);
// PetscViewerDestroy(&viewer);
// PetscCall(PetscFinalize());
// return 0;

//////

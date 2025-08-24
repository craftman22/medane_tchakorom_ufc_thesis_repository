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

    PetscMPIInt proc_local_rank __attribute__((unused));
    PetscMPIInt nprocs __attribute__((unused));
    PetscMPIInt proc_global_rank __attribute__((unused));
    PetscInt n_mesh_lines __attribute__((unused)) = 4;
    PetscInt n_mesh_columns __attribute__((unused)) = 4;
    PetscInt n_mesh_points __attribute__((unused));
    PetscInt number_of_iterations __attribute__((unused));
    PetscScalar b_norm __attribute__((unused));
    PetscScalar gmres_residual_norm __attribute__((unused));

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

    DM dm;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n_mesh_lines, n_mesh_columns, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &dm));
    DMSetFromOptions(dm);
    DMSetUp(dm);

    PetscCall(DMCreateMatrix(dm, &A));
    PetscCall(poisson2DMatrix_complete_usingDMDA(dm, A));
    PetscCall(DMCreateGlobalVector(dm, &x));
    PetscCall(VecSet(x, ZERO));
    PetscCall(DMCreateGlobalVector(dm, &u));
    PetscCall(VecSet(u, ONE));
    PetscCall(DMCreateGlobalVector(dm, &b));
    PetscCall(MatMult(A, u, b));
    PetscCall(VecNorm(b, NORM_2, &b_norm));

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_context));
    PetscCall(KSPSetInitialGuessNonzero(ksp_context, PETSC_FALSE));
    PetscCall(KSPSetDM(ksp_context, dm));
    PetscCall(KSPSetDMActive(ksp_context, PETSC_FALSE));
    PetscCall(KSPSetOperators(ksp_context, A, A));
    PetscCall(KSPSetFromOptions(ksp_context));
    PetscCall(KSPSetUp(ksp_context));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Start solving...\n"));
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    double start_time, end_time;
    start_time = MPI_Wtime();

    PetscLogStage solving_stage;
    PetscLogStageRegister("Solving stage", &solving_stage);
    PetscLogStagePush(solving_stage);
    PetscCall(KSPSolve(ksp_context, b, x));
    PetscLogStagePop();

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

    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(KSPDestroy(&ksp_context));
    PetscCall(DMDestroy(&dm));

    PetscCall(PetscFinalize());
    return 0;
}

// #endif

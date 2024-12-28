#include "unity.h"
#include "utils.h"
#include <petsc.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include "constants.h"

#define MESH_SIZE 2

PetscInt n_mesh_lines = MESH_SIZE;
PetscInt n_mesh_columns = MESH_SIZE;
PetscInt n_grid_depth = MESH_SIZE;
PetscInt s = 4;
PetscInt nprocs_per_jacobi_block; // Set in main
PetscScalar relative_tolerance = 1e-5;
MPI_Comm comm_jacobi_block; // Set in main

Mat A_block_jacobi;
PetscInt proc_global_rank;
PetscInt nprocs;
PetscInt njacobi_blocks;
PetscInt rank_jacobi_block;
PetscInt proc_local_rank;
PetscInt n_mesh_points;
PetscInt jacobi_block_size;
PetscInt rank_jacobi_block;
PetscScalar direct_residual_norm;

void setUp(void)
{
}

void tearDown(void)
{
}

void test_computeDimensionRelatedVariables()
{
    // &njacobi_blocks, &rank_jacobi_block, &proc_local_rank, &n_mesh_points, &jacobi_block_size
    TEST_ASSERT_EQUAL(2, njacobi_blocks);
    TEST_ASSERT_EQUAL(4, n_mesh_points);
    TEST_ASSERT_EQUAL(2, jacobi_block_size);
    if (proc_global_rank == 0)
    {
        TEST_ASSERT_EQUAL(0, proc_local_rank);
        TEST_ASSERT_EQUAL(0, rank_jacobi_block);
    }
    if (proc_global_rank == 1)
    {
        TEST_ASSERT_EQUAL(1, proc_local_rank);
        TEST_ASSERT_EQUAL(0, rank_jacobi_block);
    }
    if (proc_global_rank == 2)
    {
        TEST_ASSERT_EQUAL(0, proc_local_rank);
        TEST_ASSERT_EQUAL(1, rank_jacobi_block);
    }
    if (proc_global_rank == 3)
    {
        TEST_ASSERT_EQUAL(1, proc_local_rank);
        TEST_ASSERT_EQUAL(1, rank_jacobi_block);
    }
}

void test_poisson3DMatrix(void)
{

    PetscScalar values[16];
    PetscInt cols[8];
    for (PetscInt i = 0; i < 8; i++)
    {
        cols[i] = i;
    }

    if (proc_global_rank == 0)
    {
        PetscInt rows[2] = {0, 1};
        MatGetValues(A_block_jacobi, 2, rows, 8, cols, values);

        TEST_ASSERT_EQUAL(6.0, values[0]);
        TEST_ASSERT_EQUAL(-1.0, values[1]);
        TEST_ASSERT_EQUAL(-1.0, values[2]);
        TEST_ASSERT_EQUAL(0.0, values[3]);
        TEST_ASSERT_EQUAL(-1.0, values[4]);
        TEST_ASSERT_EQUAL(0.0, values[5]);
        TEST_ASSERT_EQUAL(0.0, values[6]);
        TEST_ASSERT_EQUAL(0.0, values[7]);

        TEST_ASSERT_EQUAL(-1.0, values[8]);
        TEST_ASSERT_EQUAL(6.0, values[9]);
        TEST_ASSERT_EQUAL(0.0, values[10]);
        TEST_ASSERT_EQUAL(-1.0, values[11]);
        TEST_ASSERT_EQUAL(0.0, values[12]);
        TEST_ASSERT_EQUAL(-1.0, values[13]);
        TEST_ASSERT_EQUAL(0.0, values[14]);
        TEST_ASSERT_EQUAL(0.0, values[15]);
    }
    if (proc_global_rank == 1)
    {
        PetscInt rows[2] = {2, 3};
        MatGetValues(A_block_jacobi, 2, rows, 8, cols, values);

        TEST_ASSERT_EQUAL(-1.0, values[0]);
        TEST_ASSERT_EQUAL(0.0, values[1]);
        TEST_ASSERT_EQUAL(6.0, values[2]);
        TEST_ASSERT_EQUAL(-1.0, values[3]);
        TEST_ASSERT_EQUAL(0.0, values[4]);
        TEST_ASSERT_EQUAL(0.0, values[5]);
        TEST_ASSERT_EQUAL(-1.0, values[6]);
        TEST_ASSERT_EQUAL(0.0, values[7]);

        TEST_ASSERT_EQUAL(0.0, values[8]);
        TEST_ASSERT_EQUAL(-1.0, values[9]);
        TEST_ASSERT_EQUAL(-1.0, values[10]);
        TEST_ASSERT_EQUAL(6.0, values[11]);
        TEST_ASSERT_EQUAL(0.0, values[12]);
        TEST_ASSERT_EQUAL(0.0, values[13]);
        TEST_ASSERT_EQUAL(0.0, values[14]);
        TEST_ASSERT_EQUAL(-1.0, values[15]);
    }
    if (proc_global_rank == 2)
    {

        PetscInt rows[2] = {0, 1};

        MatGetValues(A_block_jacobi, 2, rows, 8, cols, values);

        TEST_ASSERT_EQUAL(-1.0, values[0]);
        TEST_ASSERT_EQUAL(0.0, values[1]);
        TEST_ASSERT_EQUAL(0.0, values[2]);
        TEST_ASSERT_EQUAL(0.0, values[3]);
        TEST_ASSERT_EQUAL(6.0, values[4]);
        TEST_ASSERT_EQUAL(-1.0, values[5]);
        TEST_ASSERT_EQUAL(-1.0, values[6]);
        TEST_ASSERT_EQUAL(0.0, values[7]);

        TEST_ASSERT_EQUAL(0.0, values[8]);
        TEST_ASSERT_EQUAL(-1.0, values[9]);
        TEST_ASSERT_EQUAL(0.0, values[10]);
        TEST_ASSERT_EQUAL(0.0, values[11]);
        TEST_ASSERT_EQUAL(-1.0, values[12]);
        TEST_ASSERT_EQUAL(6.0, values[13]);
        TEST_ASSERT_EQUAL(0.0, values[14]);
        TEST_ASSERT_EQUAL(-1.0, values[15]);
    }
    if (proc_global_rank == 3)
    {
        PetscInt rows[2] = {2, 3};
        MatGetValues(A_block_jacobi, 2, rows, 8, cols, values);

        TEST_ASSERT_EQUAL(0.0, values[0]);
        TEST_ASSERT_EQUAL(0.0, values[1]);
        TEST_ASSERT_EQUAL(-1.0, values[2]);
        TEST_ASSERT_EQUAL(0.0, values[3]);
        TEST_ASSERT_EQUAL(-1.0, values[4]);
        TEST_ASSERT_EQUAL(0.0, values[5]);
        TEST_ASSERT_EQUAL(6.0, values[6]);
        TEST_ASSERT_EQUAL(-1.0, values[7]);

        TEST_ASSERT_EQUAL(0.0, values[8]);
        TEST_ASSERT_EQUAL(0.0, values[9]);
        TEST_ASSERT_EQUAL(0.0, values[10]);
        TEST_ASSERT_EQUAL(-1.0, values[11]);
        TEST_ASSERT_EQUAL(0.0, values[12]);
        TEST_ASSERT_EQUAL(-1.0, values[13]);
        TEST_ASSERT_EQUAL(-1.0, values[14]);
        TEST_ASSERT_EQUAL(6.0, values[15]);
    }
}

void test_poisson2DMatrix(void)
{
    // PetscInt rstart, rend;

    PetscScalar values[4];
    PetscInt cols[4];
    for (PetscInt i = 0; i < 4; i++)
    {
        cols[i] = i;
    }

    if (proc_global_rank == 0)
    {
        PetscInt rows[1] = {0};
        MatGetValues(A_block_jacobi, 1, rows, 4, cols, values);

        TEST_ASSERT_EQUAL(4.0, values[0]);
        TEST_ASSERT_EQUAL(-1.0, values[1]);
        TEST_ASSERT_EQUAL(-1.0, values[2]);
        TEST_ASSERT_EQUAL(0.0, values[3]);
    }
    if (proc_global_rank == 1)
    {
        PetscInt rows[1] = {1};
        MatGetValues(A_block_jacobi, 1, rows, 4, cols, values);
        TEST_ASSERT_EQUAL(-1.0, values[0]);
        TEST_ASSERT_EQUAL(4.0, values[1]);
        TEST_ASSERT_EQUAL(0.0, values[2]);
        TEST_ASSERT_EQUAL(-1.0, values[3]);
    }
    if (proc_global_rank == 2)
    {
        PetscInt rows[1] = {0};
        MatGetValues(A_block_jacobi, 1, rows, 4, cols, values);

        TEST_ASSERT_EQUAL(-1.0, values[0]);
        TEST_ASSERT_EQUAL(0.0, values[1]);
        TEST_ASSERT_EQUAL(4.0, values[2]);
        TEST_ASSERT_EQUAL(-1.0, values[3]);
    }
    if (proc_global_rank == 3)
    {
        PetscInt rows[1] = {1};
        MatGetValues(A_block_jacobi, 1, rows, 4, cols, values);
        TEST_ASSERT_EQUAL(0.0, values[0]);
        TEST_ASSERT_EQUAL(-1.0, values[1]);
        TEST_ASSERT_EQUAL(-1.0, values[2]);
        TEST_ASSERT_EQUAL(4.0, values[3]);
    }
}

// PetscErrorCode computeFinalResidualNorm(Mat A_block_jacobi, Vec *x, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt proc_local_rank, PetscScalar *direct_residual_norm);

void test_computeFinalResidualNorm()
{
    TEST_ASSERT_EQUAL_FLOAT(2.54567588, direct_residual_norm);
}

// not needed when using generate_test_runner.rb
int main(int argc, char **argv)
{
    UNITY_BEGIN();

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));
    nprocs_per_jacobi_block = nprocs / 2;

    // this code is to redirect the standard output to /dev/null
    int stdout_fd = dup(STDOUT_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDOUT_FILENO);
    close(devnull);

    // #########
    computeDimensionRelatedVariables(nprocs, nprocs_per_jacobi_block, proc_global_rank, n_mesh_lines, n_mesh_columns, &njacobi_blocks, &rank_jacobi_block, &proc_local_rank, &n_mesh_points, &jacobi_block_size);
    RUN_TEST(test_computeDimensionRelatedVariables);

    PetscSubcomm sub_comm_context = NULL;
    MPI_Comm dcomm;
    PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD, &dcomm, NULL));
    PetscCall(PetscSubcommCreate(dcomm, &sub_comm_context));
    PetscCall(PetscSubcommSetNumber(sub_comm_context, njacobi_blocks));
    PetscCall(PetscSubcommSetType(sub_comm_context, PETSC_SUBCOMM_CONTIGUOUS));
    PetscCall(PetscSubcommSetFromOptions(sub_comm_context));
    comm_jacobi_block = PetscSubcommChild(sub_comm_context);

    // Initialize

    MatCreate(comm_jacobi_block, &A_block_jacobi);
    MatSetType(A_block_jacobi, MATMPIAIJ);
    MatSetSizes(A_block_jacobi, PETSC_DECIDE, PETSC_DECIDE, (n_mesh_lines * n_mesh_columns) / njacobi_blocks, (n_mesh_lines * n_mesh_columns));
    MatSetFromOptions(A_block_jacobi);
    MatSetUp(A_block_jacobi);

    poisson2DMatrix(&A_block_jacobi, n_mesh_lines, n_mesh_columns, rank_jacobi_block, njacobi_blocks);

    RUN_TEST(test_poisson2DMatrix);

    Vec x;
    Vec b_block_jacobi;

    VecCreate(comm_jacobi_block, &x);
    VecSetType(x, VECMPI);
    VecSetSizes(x, PETSC_DECIDE, n_mesh_lines * n_mesh_columns);
    VecSetFromOptions(x);

    VecCreate(comm_jacobi_block, &b_block_jacobi);
    VecSetType(b_block_jacobi, VECMPI);
    VecSetSizes(b_block_jacobi, PETSC_DECIDE, (n_mesh_lines * n_mesh_columns) / njacobi_blocks);
    VecSetFromOptions(b_block_jacobi);

    if (rank_jacobi_block == BLOCK_RANK_ZERO)
    {
        PetscScalar x_values[4] = {0.1234, 0.5678, 0.9101, 0.1121};
        PetscScalar b_values[2] = {0.3141, 0.5926};
        // Set hardcoded values for x
        for (PetscInt i = 0; i < 4; i++)
        {
            VecSetValue(x, i, x_values[i], INSERT_VALUES);
        }

        // Set hardcoded values for b
        for (PetscInt i = 0; i < 2; i++)
        {
            VecSetValue(b_block_jacobi, i, b_values[i], INSERT_VALUES);
        }
    }

    if (rank_jacobi_block == BLOCK_RANK_ONE)
    {
        PetscScalar x_values[4] = {0.8765, 0.4321, 0.5432, 0.6789};
        PetscScalar b_values[2] = {0.2468, 0.1357};
        // Set hardcoded values for x
        for (PetscInt i = 0; i < 4; i++)
        {
            VecSetValue(x, i, x_values[i], INSERT_VALUES);
        }

        // Set hardcoded values for b
        for (PetscInt i = 0; i < 2; i++)
        {
            VecSetValue(b_block_jacobi, i, b_values[i], INSERT_VALUES);
        }
    }

    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    VecAssemblyBegin(b_block_jacobi);
    VecAssemblyEnd(b_block_jacobi);

    computeFinalResidualNorm_new(A_block_jacobi, &x, &b_block_jacobi, rank_jacobi_block, proc_local_rank, &direct_residual_norm);
    RUN_TEST(test_computeFinalResidualNorm);
    MatDestroy(&A_block_jacobi);

    MatCreate(comm_jacobi_block, &A_block_jacobi);
    MatSetType(A_block_jacobi, MATMPIAIJ);
    MatSetSizes(A_block_jacobi, PETSC_DECIDE, PETSC_DECIDE, (n_mesh_columns * n_mesh_lines * n_grid_depth) / njacobi_blocks, (n_mesh_columns * n_mesh_lines * n_grid_depth));
    MatSetFromOptions(A_block_jacobi);
    MatSetUp(A_block_jacobi);

    poisson3DMatrix(&A_block_jacobi, n_mesh_lines, n_mesh_columns, n_grid_depth, rank_jacobi_block, njacobi_blocks);

    RUN_TEST(test_poisson3DMatrix);

    // sum up the number of test (done, failed, ignored)
    PetscInt n_test_done = 0;
    PetscInt n_test_failed = 0;
    PetscInt n_test_ignored = 0;
    PetscCall(MPI_Allreduce(&(Unity.NumberOfTests), &n_test_done, 1, MPIU_INT, MPI_SUM, MPI_COMM_WORLD));
    PetscCall(MPI_Allreduce(&(Unity.TestFailures), &n_test_failed, 1, MPIU_INT, MPI_SUM, MPI_COMM_WORLD));
    PetscCall(MPI_Allreduce(&(Unity.TestIgnores), &n_test_ignored, 1, MPIU_INT, MPI_SUM, MPI_COMM_WORLD));

    PetscCall(PetscFinalize());

    // ici pour le code qui permet de ne rien afficher
    dup2(stdout_fd, STDOUT_FILENO);
    close(stdout_fd);

    if (proc_global_rank == 0) //
    {

        printf("Number of tests %d\n", n_test_done);
        printf("Number of tests failure %d\n", n_test_failed);
        printf("Number of test ignore %d\n", n_test_ignored);
    }

    return 0;
}
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

PetscScalar values[4]; // Values buffer
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

void test_poisson2DMatrix(void)
{
    // PetscInt rstart, rend;

    PetscInt rstart, rend;
    MatGetOwnershipRange(A_block_jacobi, &rstart, &rend);
    PetscInt row = rstart, col;

    TEST_ASSERT_EQUAL(rstart + 1, rend);

    for (int j = 0; j < 4; j++) // Loop over all columns
    {
        col = j; // Column index
        MatGetValues(A_block_jacobi, 1, &row, 1, &col, &values[j]);
    }


    if (proc_global_rank == 0)
    {
        TEST_ASSERT_EQUAL(4.0,values[0]);
        TEST_ASSERT_EQUAL( -1.0 , values[1]);
        TEST_ASSERT_EQUAL( -1.0 , values[2]);
        TEST_ASSERT_EQUAL( 0.0 , values[3]);
    }
    if (proc_global_rank == 1)
    {
        TEST_ASSERT_EQUAL( -1.0 , values[0]);
        TEST_ASSERT_EQUAL( 4.0 , values[1]);
        TEST_ASSERT_EQUAL( 0.0 , values[2]);
        TEST_ASSERT_EQUAL( -1.0 , values[3]);
    }
    if (proc_global_rank == 2 )
    {
        TEST_ASSERT_EQUAL( -1.0 , values[0]);
        TEST_ASSERT_EQUAL( 0.0 , values[1]);
        TEST_ASSERT_EQUAL( 4.0 , values[2]);
        TEST_ASSERT_EQUAL( -1.0 , values[3]);
    }
    if (proc_global_rank == 3 )
    {
        TEST_ASSERT_EQUAL( 0.0 , values[0]);
        TEST_ASSERT_EQUAL( -1.0 , values[1]);
        TEST_ASSERT_EQUAL( -1.0 , values[2]);
        TEST_ASSERT_EQUAL( 4.0 , values[3]);
    }
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
    MatSetSizes(A_block_jacobi, PETSC_DECIDE, PETSC_DECIDE, n_mesh_points / njacobi_blocks, n_mesh_points);
    MatSetFromOptions(A_block_jacobi);
    MatSetUp(A_block_jacobi);

    dup2(stdout_fd, STDOUT_FILENO);
    close(stdout_fd);

    poisson2DMatrix(&A_block_jacobi, n_mesh_lines, n_mesh_columns, rank_jacobi_block, njacobi_blocks);

    RUN_TEST(test_poisson2DMatrix);



    PetscCall(PetscFinalize());

    // ici pour le code qui permet de ne rien afficher

    if (proc_global_rank == 1) // TODO: faire un print qui prend en compte tous les processeurs
    {

        printf("Number of tests %ld\n", (UNITY_INT)(Unity.NumberOfTests));
        printf("Number of tests failure %ld\n", (UNITY_INT)(Unity.TestFailures));
        printf("Number of test ignore %ld\n", (UNITY_INT)(Unity.TestIgnores));
    }

    return 0;
}
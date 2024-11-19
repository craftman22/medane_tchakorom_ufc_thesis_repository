#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"

#define ZERO 0
#define ONE 1
// Generate one block of jacobi blocks

#define INNER_KSP_PREFIX "inner_"
#define INNER_PC_PREFIX "inner_"

#define OUTER_KSP_PREFIX "outer_"
#define OUTER_PC_PREFIX "outer_"

#define TAG_INIT 0      // Initialization phase
#define TAG_DATA 1      // Standard data transmission
#define TAG_CONTROL 2   // Control or command messages
#define TAG_TERMINATE 3 // Termination signal
#define TAG_STATUS 4    // Status or heartbeat messages

#define BLOCK_RANK_ZERO 0
#define BLOCK_RANK_ONE 1

PetscErrorCode loadMatrix(Mat *A_block_jacobi, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block, PetscInt njacobi_blocks)
{
  PetscFunctionBeginUser;

  PetscInt Idx_start = 0, Idx_end = 0;
  MatGetOwnershipRange(*A_block_jacobi, &Idx_start, &Idx_end);

  PetscInt rowBlockSize = (n_grid_lines * n_grid_columns) / njacobi_blocks;

  PetscInt i, j, J;
  PetscScalar v;
  PetscInt Ii_new;

  for (int Ii = (rank_jacobi_block * rowBlockSize) + Idx_start; Ii < (rank_jacobi_block * rowBlockSize) + Idx_end; Ii++)
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

  MatAssemblyBegin(*A_block_jacobi, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*A_block_jacobi, MAT_FINAL_ASSEMBLY);

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
  // if (rank_jacobi_block == 0)
  // PetscCall(VecView(b_block_jacobi[rank_jacobi_block], PETSC_VIEWER_STDOUT_(comm_jacobi_block)));
  // PetscCall(VecView(*b, PETSC_VIEWER_STDOUT_(comm_jacobi_block)));

  PetscCall(PetscFree(send_buffer));
  PetscCall(PetscFree(rcv_buffer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Divide the A_block_jacobi matrix into number_of_blocks matrices in the y direction. Resulting matrix has the possesses the same distribution
// on the processor on the x axis, but different distribution on y-axis
PetscErrorCode divideSubDomainIntoBlockMatrices(MPI_Comm comm_jacobi_block, Mat A_block_jacobi, Mat *A_block_jacobi_subMat, IS *is_cols_block_jacobi, PetscInt rank_jacobi_block, PetscInt njacobi_blocks, PetscInt proc_local_rank, PetscInt nprocs_per_jacobi_block)
{
  PetscFunctionBeginUser;
  PetscInt n_rows;
  PetscCall(MatGetSize(A_block_jacobi, &n_rows, NULL)); // return the number of rows and columns of the matrix

  for (int i = 0; i < njacobi_blocks; ++i)
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

  for (int i = 0; i < njacobi_blocks; ++i)
  {
    // PetscCall(MatGetSubMatrix(A_block_jacobi, is_rows_block_jacobi, is_cols_block_jacobi[i], MAT_INITIAL_MATRIX, &A_block_jacobi_subMat[i]));
    PetscCall(MatCreateSubMatrix(A_block_jacobi, is_rows_block_jacobi, is_cols_block_jacobi[i], MAT_INITIAL_MATRIX, &A_block_jacobi_subMat[i]));
  }

  PetscCall(ISDestroy(&is_rows_block_jacobi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Initiliaze a KSP context
PetscErrorCode initialiazeKSP(MPI_Comm comm_jacobi_block, KSP *ksp, Mat A_block_jacobi_subMat)
{
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(comm_jacobi_block, ksp));
  PetscCall(KSPSetOperators(*ksp, A_block_jacobi_subMat, A_block_jacobi_subMat));
  // PetscCall(KSPSetTolerances(*ksp, 0.0000000001, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE));
  //  PetscCall(KSPSetType(*ksp, KSPGMRES));
  PetscCall(KSPSetOptionsPrefix(*ksp, INNER_KSP_PREFIX));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetUp(*ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode initialiazeKSPMinimizer(MPI_Comm comm_jacobi_block, KSP *ksp, Mat R)
{
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(comm_jacobi_block, ksp));
  PetscCall(KSPSetOperators(*ksp, R, R));
  // PetscCall(KSPSetType(*ksp, KSPGMRES));
  // PetscCall(KSPSetTolerances(*ksp, 0.0000000001, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(KSPSetOptionsPrefix(*ksp, OUTER_KSP_PREFIX));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetUp(*ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode subDomainsSolver(KSP ksp, Mat *A_block_jacobi_subMat, Vec *x_block_jacobi, Vec *b_block_jacobi, PetscInt rank_jacobi_block)
{

  PetscFunctionBeginUser;

  Vec local_right_side_vector = NULL, mat_mult_vec_result = NULL;
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
  PetscCall(VecCopy(b_block_jacobi[rank_jacobi_block], local_right_side_vector));
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

  PetscInt idx = (rank_jacobi_block == ZERO ? ONE : ZERO);
  PetscCall(MatMult(A_block_jacobi_subMat[idx], x_block_jacobi[idx], mat_mult_vec_result));
  PetscCall(VecAXPY(local_right_side_vector, -1, mat_mult_vec_result));

  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));
  PetscCall(KSPSolve(ksp, local_right_side_vector, x_block_jacobi[rank_jacobi_block]));
  PetscInt n_iterations = 0;
  PetscCall(KSPGetIterationNumber(ksp, &n_iterations));

  if (rank_jacobi_block == BLOCK_RANK_ZERO)
  {
    MPI_Comm tmp;
    PetscCall(PetscObjectGetComm((PetscObject)local_right_side_vector, &tmp));
    PetscCall(PetscPrintf(tmp, " [Block %d] NUMBER OF INNER SOLVER ITERATIONS : %d ...   \n", rank_jacobi_block, n_iterations));
  }
  PetscCall(VecDestroy(&local_right_side_vector));
  PetscCall(VecDestroy(&mat_mult_vec_result));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeResidualNorm2(Mat A_block_jacobi, Vec x, Vec *b_block_jacobi, PetscReal *global_residual_norm2, PetscInt rank_jacobi_block, PetscInt proc_local_rank)
{

  PetscFunctionBegin;

  PetscReal local_residual_norm2 = 0;
  Vec local_residual;
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_residual));
  PetscCall(MatResidual(A_block_jacobi, b_block_jacobi[rank_jacobi_block], x, local_residual));
  PetscCall(VecNorm(local_residual, NORM_2, &local_residual_norm2));
  local_residual_norm2 = local_residual_norm2 * local_residual_norm2;
  if (proc_local_rank != 0)
    local_residual_norm2 = 0.0;
  PetscCallMPI(MPI_Allreduce(&local_residual_norm2, global_residual_norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

  PetscCall(VecDestroy(&local_residual));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode minimizerSolver(MPI_Comm comm_jacobi_block, Vec x_minimized, Mat R, Mat S, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt s)
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

  KSP ksp_minimizer = NULL;
  PetscCall(initialiazeKSPMinimizer(comm_jacobi_block, &ksp_minimizer, R_transpose_R));

  PC ksp_minimizer_preconditionnner = NULL;
  PetscCall(PCCreate(comm_jacobi_block, &ksp_minimizer_preconditionnner));
  // PetscCall(PCSetType(ksp_minimizer_preconditionnner, PCNONE));
  PetscCall(PCSetOperators(ksp_minimizer_preconditionnner, R_transpose_R, R_transpose_R));
  PetscCall(PCSetOptionsPrefix(ksp_minimizer_preconditionnner, OUTER_PC_PREFIX));
  PetscCall(PCSetFromOptions(ksp_minimizer_preconditionnner));
  PetscCall(PCSetUp(ksp_minimizer_preconditionnner));
  PetscCall(KSPSetPC(ksp_minimizer, ksp_minimizer_preconditionnner));

  PetscReal ksp_minimizer_relative_tolerance;
  PetscInt ksp_minimizer_max_iterations;
  KSPType ksp_minimizer_type;
  PCType ksp_minimizer_pc_type;

  PetscCall(KSPGetTolerances(ksp_minimizer, &ksp_minimizer_relative_tolerance, NULL, NULL, &ksp_minimizer_max_iterations));
  PetscCall(KSPGetType(ksp_minimizer, &ksp_minimizer_type));
  PetscCall((KSPGetPC(ksp_minimizer, &ksp_minimizer_preconditionnner)));
  PetscCall(PCGetType(ksp_minimizer_preconditionnner, &ksp_minimizer_pc_type));
  PetscCall(KSPGetType(ksp_minimizer, &ksp_minimizer_type));

  Vec alpha = NULL;
  PetscCall(VecCreate(comm_jacobi_block, &alpha));
  PetscCall(VecSetType(alpha, VECMPI));
  PetscCall(VecSetSizes(alpha, PETSC_DECIDE, s));
  PetscCall(VecSetFromOptions(alpha));
  PetscCall(VecSetUp(alpha));
  PetscCall(KSPSolve(ksp_minimizer, vec_R_transpose_b_block_jacobi, alpha));

  PetscCall(MatMult(S, alpha, x_minimized));

  PetscCall(MatDestroy(&R_transpose_R));
  PetscCall(VecDestroy(&vec_R_transpose_b_block_jacobi));
  PetscCall(VecDestroy(&alpha));
  PetscCall(KSPDestroy(&ksp_minimizer));
  PetscCall(PCDestroy(&ksp_minimizer_preconditionnner));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ((n*n) * s) * ( s * 1 ) = ( n * 1)

int main(int argc, char **argv)
{

  // DM da_matrix = NULL;
  // DM da_vectors = NULL;
  Mat A_block_jacobi = NULL;
  Vec x = NULL; // vector of unknows
  Vec b = NULL; // right hand side vector
                // Vec x_prev_iteration = NULL;
  Vec x_initial_guess = NULL;

  PetscMPIInt nprocs;
  PetscInt proc_global_rank;
  PetscInt n_grid_lines = 4;
  PetscInt n_grid_columns = 4;
  PetscInt s;
  PetscInt nprocs_per_jacobi_block = 1;
  PetscReal relative_tolerance = 1e-5;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

  // Getting applications arguments
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &n_grid_lines, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n_grid_columns, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-s", &s, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npb", &nprocs_per_jacobi_block, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &relative_tolerance, NULL));

  // PetscPrintf(PETSC_COMM_WORLD, " =====> Total number of processes: %d \n =====>s : %d\n =====>nprocessor_per_jacobi_block : %d \n ====> Grid lines: %d \n ====> Grid columns : %d \n ====> Relative tolerance : %f\n", nprocs, s, nprocs_per_jacobi_block, n_grid_lines, n_grid_columns,relative_tolerance);

  PetscInt njacobi_blocks = (PetscInt)(nprocs / nprocs_per_jacobi_block);
  PetscInt rank_jacobi_block = proc_global_rank / nprocs_per_jacobi_block;
  PetscInt proc_local_rank = (proc_global_rank % nprocs_per_jacobi_block);

  // Check if the number of lines (or columns) of the matrix resulting from discretization is divisible by the total number of processes
  PetscInt n_grid_points = n_grid_columns * n_grid_lines;
  PetscInt jacobi_block_size = n_grid_points / njacobi_blocks;
  PetscAssert((n_grid_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");

  // Creating the sub communicator for each jacobi block
  PetscSubcomm sub_comm_context = NULL;
  MPI_Comm dcomm;
  PetscCommDuplicate(PETSC_COMM_WORLD, &dcomm, NULL);
  PetscCall(PetscSubcommCreate(dcomm, &sub_comm_context));
  PetscCall(PetscSubcommSetNumber(sub_comm_context, njacobi_blocks));
  PetscCall(PetscSubcommSetType(sub_comm_context, PETSC_SUBCOMM_CONTIGUOUS));
  PetscCall(PetscSubcommSetFromOptions(sub_comm_context));
  MPI_Comm comm_jacobi_block = PetscSubcommChild(sub_comm_context);

  PetscCall(VecCreate(comm_jacobi_block, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n_grid_points));
  PetscCall(VecSetType(x, VECMPI));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetUp(x));

  PetscCall(VecDuplicate(x, &b));

  PetscCall(VecDuplicate(x, &x_initial_guess));
  PetscScalar initial_scalar_value = 1.0;
  PetscCall(VecSet(x_initial_guess, initial_scalar_value));

  PetscCall(MatCreate(comm_jacobi_block, &A_block_jacobi));
  PetscCall(MatSetType(A_block_jacobi, MATMPIAIJ));
  PetscCall(MatSetSizes(A_block_jacobi, PETSC_DECIDE, PETSC_DECIDE, n_grid_points / njacobi_blocks, n_grid_points));
  PetscCall(MatSetFromOptions(A_block_jacobi));
  PetscCall(MatSetUp(A_block_jacobi));

  // Insert non-zeros entries into the operator matrix
  PetscCall(loadMatrix(&A_block_jacobi, n_grid_lines, n_grid_columns, rank_jacobi_block, njacobi_blocks));

  IS is_cols_block_jacobi[njacobi_blocks];
  Mat A_block_jacobi_subMat[njacobi_blocks];
  Vec b_block_jacobi[njacobi_blocks];
  Vec x_block_jacobi[njacobi_blocks];

  PetscCall(divideSubDomainIntoBlockMatrices(comm_jacobi_block, A_block_jacobi, A_block_jacobi_subMat, is_cols_block_jacobi, rank_jacobi_block, njacobi_blocks, proc_local_rank, nprocs_per_jacobi_block));

  for (PetscInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(VecCreate(comm_jacobi_block, &b_block_jacobi[i]));
    PetscCall(VecSetType(b_block_jacobi[i], VECMPI));
    PetscCall(VecSetSizes(b_block_jacobi[i], PETSC_DECIDE, jacobi_block_size));
    PetscCall(VecSetFromOptions(b_block_jacobi[i]));
    PetscCall(VecSetUp(b_block_jacobi[i]));
  }

  for (PetscInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(VecCreate(comm_jacobi_block, &x_block_jacobi[i]));
    PetscCall(VecSetType(x_block_jacobi[i], VECMPI));
    PetscCall(VecSetSizes(x_block_jacobi[i], PETSC_DECIDE, jacobi_block_size));
    PetscCall(VecSetType(x_block_jacobi[i], VECMPI));
    PetscCall(VecSetFromOptions(x_block_jacobi[i]));
    PetscCall(VecSetUp(x_block_jacobi[i]));
  }

  // creation of a scatter context to manage data transfert between complete b or x , and their part x_block_jacobi[..] and b_block_jacobi[...]
  VecScatter scatter_jacobi_vec_part_to_merged_vec[njacobi_blocks];
  IS is_jacobi_vec_parts;
  IS is_merged_vec[njacobi_blocks];

  PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, ZERO, ONE, &is_jacobi_vec_parts));

  for (int i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, (i * (jacobi_block_size)), ONE, &is_merged_vec[i]));
    PetscCall(VecScatterCreate(b_block_jacobi[i], is_jacobi_vec_parts, b, is_merged_vec[i], &scatter_jacobi_vec_part_to_merged_vec[i]));
  }

  PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, &b, b_block_jacobi, x_initial_guess, rank_jacobi_block, jacobi_block_size, nprocs_per_jacobi_block, proc_local_rank));

  PetscBool stop_condition = PETSC_FALSE;
  PetscInt number_of_iterations = 0;

  PetscInt idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;

  PetscReal global_residual_norm2 = PETSC_MAX_REAL;
  KSP ksp = NULL;
  PetscCall(initialiazeKSP(comm_jacobi_block, &ksp, A_block_jacobi_subMat[rank_jacobi_block]));

  PC ksp_preconditionnner = NULL;
  PetscCall(PCCreate(comm_jacobi_block, &ksp_preconditionnner));
  PetscCall(PCSetOperators(ksp_preconditionnner, A_block_jacobi_subMat[rank_jacobi_block], A_block_jacobi_subMat[rank_jacobi_block]));
  // PetscCall(PCSetType(ksp_preconditionnner, PCHYPRE));
  PetscCall(PCSetOptionsPrefix(ksp_preconditionnner, INNER_PC_PREFIX));
  PetscCall(PCSetFromOptions(ksp_preconditionnner));
  PetscCall(PCSetUp(ksp_preconditionnner));
  PetscCall(KSPSetPC(ksp, ksp_preconditionnner));

  // PetscReal ksp_relative_tolerance;
  // PetscInt ksp_max_iterations;
  // KSPType ksp_type;
  // PCType ksp_pc_type;
  // PetscCall(KSPGetTolerances(ksp, &ksp_relative_tolerance, NULL, NULL, &ksp_max_iterations));
  // PetscCall(KSPGetType(ksp, &ksp_type));
  // PetscCall((KSPGetPC(ksp, &ksp_preconditionnner)));
  // PetscCall(PCGetType(ksp_preconditionnner, &ksp_pc_type));
  // PetscCall(KSPGetType(ksp, &ksp_type));

  // PetscCall(PetscPrintf(MPI_COMM_WORLD, "KSP type: %s\n", ksp_type));
  // PetscCall(PetscPrintf(MPI_COMM_WORLD, "KSP preconditionner: %s \n", ksp_pc_type));
  // PetscCall(PetscPrintf(MPI_COMM_WORLD, "KSP relative tolerance: %g\n", ksp_relative_tolerance));
  // PetscCall(PetscPrintf(MPI_COMM_WORLD, "KSP max iterations: %d\n", ksp_max_iterations));
  // PetscCall(PetscPrintf(MPI_COMM_WORLD, "Initial value of norm 2  =  %g\n", global_residual_norm2));
  // PetscCall(PetscPrintf(MPI_COMM_WORLD, "*******************************************\n"));
  // PetscCall(PetscPrintf(MPI_COMM_WORLD, "*******************************************\n\n"));

  PetscScalar *send_buffer = NULL;
  PetscScalar *rcv_buffer = NULL;
  PetscScalar *temp_buffer = NULL;
  PetscInt vec_local_size = 0;
  PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
  PetscMalloc1((size_t)vec_local_size, &send_buffer);
  PetscMalloc1((size_t)vec_local_size, &rcv_buffer);

  // Minimization variables
  Mat R = NULL;
  Mat S = NULL;
  int n_vectors_inserted;
  Vec x_minimized = NULL;

  PetscCall(VecCreate(comm_jacobi_block, &x_minimized));
  PetscCall(VecSetType(x_minimized, VECMPI));
  PetscCall(VecSetSizes(x_minimized, PETSC_DECIDE, n_grid_points));
  PetscCall(VecSetFromOptions(x_minimized));
  PetscCall(VecSetUp(x_minimized));

  PetscCall(MatCreate(comm_jacobi_block, &R));
  PetscCall(MatSetType(R, MATMPIDENSE));
  PetscCall(MatSetSizes(R, PETSC_DECIDE, PETSC_DECIDE, jacobi_block_size, s));
  PetscCall(MatSetFromOptions(R));
  PetscCall(MatSetUp(R));

  PetscCall(MatCreate(comm_jacobi_block, &S));
  PetscCall(MatSetType(S, MATMPIDENSE));
  PetscCall(MatSetFromOptions(S));
  PetscCall(MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, n_grid_points, s));
  PetscCall(MatSetUp(S));

  PetscInt *vec_local_idx = NULL;
  PetscInt x_local_size;
  PetscCall(VecGetLocalSize(x, &x_local_size));
  vec_local_idx = (PetscInt *)malloc(x_local_size * sizeof(PetscInt));
  for (int i = 0; i < (x_local_size); i++)
  {
    vec_local_idx[i] = (proc_local_rank * x_local_size) + i;
  }
  PetscScalar *vector_to_insert_into_S = (PetscScalar *)malloc(x_local_size * sizeof(PetscScalar));
  Vec local_residual = NULL;
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_residual));

  MPI_Request rcv_request;
  MPI_Request send_request;
  PetscInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
  PetscInt message_dest = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
  PetscCallMPI(MPI_Recv_init(rcv_buffer, vec_local_size, MPIU_SCALAR, message_source, TAG_DATA, MPI_COMM_WORLD, &rcv_request));
  PetscCallMPI(MPI_Send_init(send_buffer, vec_local_size, MPIU_SCALAR, message_dest, TAG_DATA, MPI_COMM_WORLD, &send_request));

  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  do
  {

    n_vectors_inserted = 0;
    while (n_vectors_inserted < s)
    {
      PetscCall(subDomainsSolver(ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, rank_jacobi_block));

      if (rank_jacobi_block == BLOCK_RANK_ZERO)
      {

        PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
        memcpy(send_buffer, temp_buffer, vec_local_size * sizeof(PetscReal));
        PetscCallMPI(MPI_Start(&send_request));
        PetscCallMPI(MPI_Wait(&send_request, MPI_STATUS_IGNORE));
        PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));

        PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
        PetscCallMPI(MPI_Start(&rcv_request));
        PetscCallMPI(MPI_Wait(&rcv_request, MPI_STATUS_IGNORE));
        memcpy(temp_buffer, rcv_buffer, vec_local_size * sizeof(PetscReal));
        PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
      }
      else if (rank_jacobi_block == BLOCK_RANK_ONE)
      {
        PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
        PetscCallMPI(MPI_Start(&rcv_request));
        PetscCallMPI(MPI_Wait(&rcv_request, MPI_STATUS_IGNORE));
        memcpy(temp_buffer, rcv_buffer, vec_local_size * sizeof(PetscReal));
        PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_buffer));

        PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
        memcpy(send_buffer, temp_buffer, vec_local_size * sizeof(PetscReal));
        PetscCallMPI(MPI_Start(&send_request));
        PetscCallMPI(MPI_Wait(&send_request, MPI_STATUS_IGNORE));
        PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
      }

      PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));

      PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

      PetscCall(VecGetValues(x, x_local_size, vec_local_idx, vector_to_insert_into_S));
      PetscCall(MatSetValuesLocal(S, x_local_size, vec_local_idx, ONE, &n_vectors_inserted, vector_to_insert_into_S, INSERT_VALUES));

      n_vectors_inserted++;
    }

    PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));

    PetscCall(MatMatMult(A_block_jacobi, S, MAT_REUSE_MATRIX, PETSC_DETERMINE, &R););
    PetscCall(minimizerSolver(comm_jacobi_block, x_minimized, R, S, b_block_jacobi, rank_jacobi_block, s));

    // PetscCall(computeResidualNorm2(A_block_jacobi, x_minimized, b_block_jacobi, &global_residual_norm2, rank_jacobi_block, proc_local_rank));
    PetscReal local_residual_norm2 = 0;
    PetscCall(MatResidual(A_block_jacobi, b_block_jacobi[rank_jacobi_block], x_minimized, local_residual));
    PetscCall(VecNorm(local_residual, NORM_2, &local_residual_norm2));
    local_residual_norm2 = local_residual_norm2 * local_residual_norm2;
    if (proc_local_rank != 0)
      local_residual_norm2 = 0.0;
    PetscCallMPI(MPI_Allreduce(&local_residual_norm2, &global_residual_norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "After minimization :  Norm 2 ==== %g \n", global_residual_norm2));

    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));

    if (PetscApproximateLTE(global_residual_norm2, relative_tolerance))
    {
      stop_condition = PETSC_TRUE;
    }
    number_of_iterations = number_of_iterations + 1;

  } while (stop_condition == PETSC_FALSE);

  PetscCallMPI(MPI_Request_free(&send_request));
  PetscCallMPI(MPI_Request_free(&rcv_request));

  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(VecCopy(x_minimized, x));

  // VecView(x_minimized, PETSC_VIEWER_STDOUT_(MPI_COMM_WORLD));

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "\n Total number of iterations (s * outer iterations): %d   ====  Final Norm 2 :%g \n", s * number_of_iterations, (double)global_residual_norm2));

  PetscCall(ISDestroy(&is_jacobi_vec_parts));
  for (int i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(ISDestroy(&is_merged_vec[i]));
    PetscCall(ISDestroy(&is_cols_block_jacobi[i]));
    PetscCall(VecDestroy(&x_block_jacobi[i]));
    PetscCall(VecDestroy(&b_block_jacobi[i]));
    PetscCall(MatDestroy(&A_block_jacobi_subMat[i]));
    PetscCall(VecScatterDestroy(&scatter_jacobi_vec_part_to_merged_vec[i]));
  }

  PetscCall(VecDestroy(&local_residual));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x_minimized));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x_initial_guess));
  PetscCall(MatDestroy(&A_block_jacobi));
  PetscCall(MatDestroy(&R));
  PetscCall(MatDestroy(&S));
  PetscCall(PetscFree(send_buffer));
  PetscCall(PetscFree(rcv_buffer));
  PetscCall(PCDestroy(&ksp_preconditionnner));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
}

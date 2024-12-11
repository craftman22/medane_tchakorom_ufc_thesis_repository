/*

  1. inclure la detection de convergence utilisant un noeud racine
  2. tester cette nouvelle detection de convergence
  3. changer le code en un code asynchrone

  il y a des mpi barrier dans la partie concernant les slave processes
  MPI_Comm_attach_buffer()


  how important are tags in mpi communication
  Why use MPI_IPROB instead of directly use mpi_ircv

*/

#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"

#define ZERO 0
#define ONE 1
#define MINUS_ONE -1

#define INNER_KSP_PREFIX "inner_"
#define INNER_PC_PREFIX "inner_"
// Generate one block of jacobi blocks

#define NO_MESSAGE -14
#define NO_SIGNAL -15
#define CONVERGENCE_SIGNAL 418
#define DIVERGENCE_SIGNAL 421
#define TERMINATE_SIGNAL 884

#define TAG_INIT 0      // Initialization phase
#define TAG_DATA 1      // Standard data transmission
#define TAG_CONTROL 2   // Control or command messages
#define TAG_TERMINATE 3 // Termination signal
#define TAG_STATUS 4    // Status or heartbeat messages

#define BLOCK_RANK_ZERO 0
#define BLOCK_RANK_ONE 1
// #define BLOCK_RANK_TWO 2

PetscErrorCode loadMatrix(Mat *A_block_jacobi, PetscInt n_grid_lines, PetscInt n_grid_columns, PetscInt rank_jacobi_block, PetscInt njacobi_blocks)
{
  PetscFunctionBeginUser;

  PetscInt Idx_start = 0, Idx_end = 0;
  MatGetOwnershipRange(*A_block_jacobi, &Idx_start, &Idx_end);

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

  PetscCall(PetscFree(rcv_buffer));
  PetscCall(PetscFree(send_buffer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Divide the A_block_jacobi matrix into number_of_blocks matrices in the y direction. Resulting matrix has the possesses the same distribution
// on the processor on the x axis, but different distribution on y-axis
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
PetscErrorCode initialiazeKSP(MPI_Comm comm_jacobi_block, KSP *ksp, Mat A_block_jacobi_subMat)
{
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(comm_jacobi_block, ksp));
  PetscCall(KSPSetOperators(*ksp, A_block_jacobi_subMat, A_block_jacobi_subMat));
  // PetscCall(KSPSetType(*ksp, KSPCG));
  // PetscCall(KSPSetTolerances(*ksp, 0.0000000001, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(KSPSetOptionsPrefix(*ksp, INNER_KSP_PREFIX));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetUp(*ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode subDomainsSolver(KSP ksp, Mat *A_block_jacobi_subMat, Vec *x_block_jacobi, Vec *b_block_jacobi, PetscInt rank_jacobi_block, PetscInt *inner_solver_iterations)
{

  PetscFunctionBeginUser;
  Vec local_right_side_vector = NULL, mat_mult_vec_result = NULL;
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_right_side_vector));
  PetscCall(VecCopy(b_block_jacobi[rank_jacobi_block], local_right_side_vector));
  PetscCall(VecDuplicate(b_block_jacobi[rank_jacobi_block], &mat_mult_vec_result));

  PetscInt idx = (rank_jacobi_block == ZERO ? ONE : ZERO);
  PetscCall(MatMult(A_block_jacobi_subMat[idx], x_block_jacobi[idx], mat_mult_vec_result));
  PetscCall(VecAXPY(local_right_side_vector, -1, mat_mult_vec_result));

  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSolve(ksp, local_right_side_vector, x_block_jacobi[rank_jacobi_block]));
  PetscInt n_iterations = 0;
  PetscCall(KSPGetIterationNumber(ksp, &n_iterations));
  if (rank_jacobi_block == 0)
  {
    MPI_Comm tmp;
    PetscCall(PetscObjectGetComm((PetscObject)local_right_side_vector, &tmp));
    PetscCall(PetscPrintf(tmp, "NUMBER OF ITERATIONS : %d  ====== ", n_iterations));
  }

  *inner_solver_iterations = n_iterations;
  PetscCall(VecDestroy(&local_right_side_vector));
  PetscCall(VecDestroy(&mat_mult_vec_result));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeResidualNorm2(Mat A_block_jacobi, Vec b_block_jacobi, Vec x, PetscScalar *global_residual_norm2, PetscInt proc_local_rank)
{

  PetscFunctionBegin;
  Vec local_residual = NULL;
  VecDuplicate(b_block_jacobi, &local_residual);
  PetscInt grows, gcols;
  MatGetSize(A_block_jacobi, &grows, &gcols);
  PetscPrintf(PETSC_COMM_WORLD, "Mat  rows: %d   cols: %d\n", grows, gcols);
  PetscInt bsize;
  VecGetSize(b_block_jacobi, &bsize);
  PetscPrintf(PETSC_COMM_WORLD, "Vec b block jacobi  rows: %d  cols : 1\n", bsize);
  PetscInt xsize;
  VecGetSize(x, &xsize);
  PetscPrintf(PETSC_COMM_WORLD, "Vec x    rows: %d  cols : 1\n", xsize);
  PetscCall(MatResidual(A_block_jacobi, b_block_jacobi, x, local_residual));
  return 0;
  PetscScalar local_residual_norm2 = PETSC_MAX_REAL;
  PetscCall(VecNorm(local_residual, NORM_2, &local_residual_norm2));
  local_residual_norm2 = local_residual_norm2 * local_residual_norm2;
  if (proc_local_rank == 0)
    local_residual_norm2 = 0.0;
  PetscCallMPI(MPI_Allreduce(&local_residual_norm2, &global_residual_norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{

  Mat A_block_jacobi = NULL; // Operator matrix
  Vec x = NULL;              // approximation solution at iteration (k)
  Vec b = NULL;              // right hand side vector
  Vec x_initial_guess = NULL;

  PetscMPIInt nprocs;
  PetscInt proc_global_rank;
  PetscInt n_grid_lines = 4;
  PetscInt n_grid_columns = 4;
  PetscInt s;
  PetscScalar relative_tolerance = 1e-5;
  PetscInt nprocs_per_jacobi_block = 1;
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

  PetscPrintf(PETSC_COMM_WORLD, " =====> Total number of processes: %d \n =====>s : %d\n =====>nprocessor_per_jacobi_block : %d \n ====> Grid lines: %d \n ====> Grid columns : %d ====> Relative tolerance : %f\n", nprocs, s, nprocs_per_jacobi_block, n_grid_lines, n_grid_columns, relative_tolerance);

  PetscInt njacobi_blocks = (PetscInt)(nprocs / nprocs_per_jacobi_block);
  PetscInt rank_jacobi_block = (PetscInt)(proc_global_rank / nprocs_per_jacobi_block);
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
  // PetscCall(PetscSubcommSetTypeGeneral(sub_comm_context, rank_jacobi_block, proc_local_rank));
  // PetscCall(PetscSubcommSetFromOptions(sub_comm_context));
  MPI_Comm comm_jacobi_block = PetscSubcommChild(sub_comm_context);


  PetscInt send_signal = NO_SIGNAL;
  PetscInt rcv_signal = NO_SIGNAL;

  // Vector of unknowns
  PetscCall(VecCreate(comm_jacobi_block, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n_grid_points));
  PetscCall(VecSetType(x, VECMPI));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetUp(x));

  // Right hand side
  PetscCall(VecDuplicate(x, &b));

  // Initial guess solution
  PetscCall(VecDuplicate(x, &x_initial_guess));
  PetscScalar initial_scalar_value = 1.0;
  PetscCall(VecSet(x_initial_guess, initial_scalar_value));

  // Operator matrix
  PetscCall(MatCreate(comm_jacobi_block, &A_block_jacobi));
  PetscCall(MatSetType(A_block_jacobi, MATMPIAIJ));
  PetscCall(MatSetSizes(A_block_jacobi, PETSC_DECIDE, PETSC_DECIDE, n_grid_points / njacobi_blocks, n_grid_points));
  PetscCall(MatSetFromOptions(A_block_jacobi));
  PetscCall(MatSetUp(A_block_jacobi));

  // Insert non-zeros values into the sparse operator matrix
  PetscCall(loadMatrix(&A_block_jacobi, n_grid_lines, n_grid_columns, rank_jacobi_block, njacobi_blocks));

  Mat A_block_jacobi_subMat[njacobi_blocks];
  IS is_cols_block_jacobi[njacobi_blocks];
  Vec b_block_jacobi[njacobi_blocks];
  Vec x_block_jacobi[njacobi_blocks];

  // domain decomposition of matrix and vectors
  PetscCall(divideSubDomainIntoBlockMatrices(comm_jacobi_block, A_block_jacobi, A_block_jacobi_subMat, is_cols_block_jacobi, rank_jacobi_block, njacobi_blocks, proc_local_rank, nprocs_per_jacobi_block));

  for (PetscInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(VecCreate(comm_jacobi_block, &x_block_jacobi[i]));
    PetscCall(VecSetSizes(x_block_jacobi[i], PETSC_DECIDE, jacobi_block_size));
    PetscCall(VecSetType(x_block_jacobi[i], VECMPI));
    PetscCall(VecSetFromOptions(x_block_jacobi[i]));
    PetscCall(VecSetUp(x_block_jacobi[i]));
  }

  for (PetscInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(VecCreate(comm_jacobi_block, &b_block_jacobi[i]));
    PetscCall(VecSetSizes(b_block_jacobi[i], PETSC_DECIDE, jacobi_block_size));
    PetscCall(VecSetType(b_block_jacobi[i], VECMPI));
    PetscCall(VecSetFromOptions(b_block_jacobi[i]));
    PetscCall(VecSetUp(b_block_jacobi[i]));
  }

  // creation of a scatter context to manage data transfert between complete b or x , and their part x_block_jacobi[..] and b_block_jacobi[...]
  VecScatter scatter_jacobi_vec_part_to_merged_vec[njacobi_blocks];
  IS is_jacobi_vec_parts;
  IS is_merged_vec[njacobi_blocks];

  PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, ZERO, ONE, &is_jacobi_vec_parts));
  for (PetscInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, (i * (jacobi_block_size)), ONE, &is_merged_vec[i]));
    PetscCall(VecScatterCreate(b_block_jacobi[i], is_jacobi_vec_parts, b, is_merged_vec[i], &scatter_jacobi_vec_part_to_merged_vec[i]));
  }

  // compute right hand side vector based on the initial guess
  PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, &b, b_block_jacobi, x_initial_guess, rank_jacobi_block, jacobi_block_size, nprocs_per_jacobi_block, proc_local_rank));

  PetscInt number_of_iterations = 0;
  PetscInt idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;
  PetscScalar approximation_residual_infinity_norm = PETSC_MAX_REAL;

  KSP ksp = NULL;
  PetscCall(initialiazeKSP(comm_jacobi_block, &ksp, A_block_jacobi_subMat[rank_jacobi_block]));

  PC ksp_preconditionnner = NULL;
  PetscCall(PCCreate(comm_jacobi_block, &ksp_preconditionnner));
  PetscCall(PCSetOperators(ksp_preconditionnner, A_block_jacobi_subMat[rank_jacobi_block], A_block_jacobi_subMat[rank_jacobi_block]));
  // PetscCall(PCSetType(ksp_preconditionnner, PCNONE));
  PetscCall(PCSetOptionsPrefix(ksp_preconditionnner, INNER_PC_PREFIX));
  PetscCall(PCSetFromOptions(ksp_preconditionnner));
  PetscCall(PCSetUp(ksp_preconditionnner));
  PetscCall(KSPSetPC(ksp, ksp_preconditionnner));

  PetscInt vec_local_size = 0;
  PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
  PetscScalar *send_buffer = NULL;
  PetscScalar *rcv_buffer = NULL;
  PetscScalar *temp_buffer = NULL;
  PetscMalloc1((size_t)vec_local_size, &send_buffer);
  PetscMalloc1((size_t)vec_local_size, &rcv_buffer);

  Vec approximation_residual;
  PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &approximation_residual));

  PetscInt reduced_message = NO_MESSAGE;
  PetscInt message = NO_MESSAGE;
  PetscInt root_message = NO_MESSAGE;
  MPI_Status status;
  PetscInt send_flag = 0;
  PetscInt rcv_flag = 0;
  PetscInt send_signal_flag = 0;
  PetscInt rcv_signal_flag = 0;

  MPI_Request rcv_request;
  MPI_Request send_request;
  // MPI_Status rcv_status;
  // MPI_Request request;
  MPI_Request send_signal_request;
  MPI_Request rcv_signal_request;

  PetscInt inner_solver_iterations = 0;

  Vec x_block_jacobi_previous_iteration = NULL;
  PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &x_block_jacobi_previous_iteration));

  PetscInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
  PetscInt message_dest = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
  PetscCallMPI(MPI_Recv_init(rcv_buffer, vec_local_size, MPIU_SCALAR, message_source, TAG_DATA, MPI_COMM_WORLD, &rcv_request));
  PetscCallMPI(MPI_Send_init(send_buffer, vec_local_size, MPIU_SCALAR, message_dest, TAG_DATA, MPI_COMM_WORLD, &send_request));

  message_dest = ZERO;
  message_source = MPI_ANY_SOURCE;
  PetscCallMPI(MPI_Send_init(&send_signal, ONE, MPIU_INT, message_dest, TAG_STATUS, MPI_COMM_WORLD, &send_signal_request));
  PetscCallMPI(MPI_Recv_init(&rcv_signal, ONE, MPIU_INT, message_source, TAG_STATUS, MPI_COMM_WORLD, &rcv_signal_request));

  MPI_Request *send_requests;
  PetscMalloc1((size_t)nprocs, &send_requests);
  for (PetscInt proc_rank = ZERO; proc_rank < nprocs; proc_rank++)
  {
    PetscCallMPI(MPI_Send_init(&root_message, ONE, MPIU_INT, proc_rank, TAG_TERMINATE, MPI_COMM_WORLD, &send_requests[proc_rank]));
  }

  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  double start_time, end_time;
  start_time = MPI_Wtime();

  do
  {

    PetscCall(subDomainsSolver(ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, rank_jacobi_block, &inner_solver_iterations));

    if (rank_jacobi_block == BLOCK_RANK_ZERO)
    {

      MPI_Test(&send_request, &send_flag, MPI_STATUS_IGNORE);
      if (send_flag && (inner_solver_iterations > 0))
      {
        PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
        memcpy(send_buffer, temp_buffer, vec_local_size * sizeof(PetscScalar));
        PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
        MPI_Start(&send_request);
      }

      // MPI_Test(&rcv_request, &rcv_flag, MPI_STATUS_IGNORE);
      MPI_Iprobe(message_source, TAG_DATA, MPI_COMM_WORLD, &rcv_flag, MPI_STATUS_IGNORE);
      if (rcv_flag)
      {
        MPI_Start(&rcv_request);
        PetscCallMPI(MPI_Wait(&rcv_request, MPI_STATUS_IGNORE));
        PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
        memcpy(temp_buffer, rcv_buffer, vec_local_size * sizeof(PetscScalar));
        PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
      }
    }
    else if (rank_jacobi_block == BLOCK_RANK_ONE)
    {

      MPI_Test(&send_request, &send_flag, MPI_STATUS_IGNORE);
      if (send_flag && (inner_solver_iterations > 0))
      {
        PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
        memcpy(send_buffer, temp_buffer, vec_local_size * sizeof(PetscScalar));
        PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
        MPI_Start(&send_request);
      }

      // MPI_Test(&rcv_request, &rcv_flag, MPI_STATUS_IGNORE);
      MPI_Iprobe(message_source, TAG_DATA, MPI_COMM_WORLD, &rcv_flag, MPI_STATUS_IGNORE);
      if (rcv_flag)
      {
        MPI_Start(&rcv_request);
        PetscCallMPI(MPI_Wait(&rcv_request, MPI_STATUS_IGNORE));
        PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
        memcpy(temp_buffer, rcv_buffer, vec_local_size * sizeof(PetscScalar));
        PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
      }
    }

    if (inner_solver_iterations > 0)
    {
      PetscCall(VecWAXPY(approximation_residual, -1, x_block_jacobi_previous_iteration, x_block_jacobi[rank_jacobi_block]));
      PetscCall(VecNorm(approximation_residual, NORM_INFINITY, &approximation_residual_infinity_norm));
      PetscCall(VecCopy(x_block_jacobi[rank_jacobi_block], x_block_jacobi_previous_iteration));
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Infinity norm of residual  ==== %e \n", approximation_residual_infinity_norm));
    }
    else
    {
      PetscCall(PetscPrintf(MPI_COMM_WORLD, "Infinity norm of residual  (no new data) ==== %e \n", approximation_residual_infinity_norm));
    }

    // Setting convergence criteria to true, and then sending convergence signal to process rank 0 on mpi_comm_world
    if (PetscApproximateLTE(approximation_residual_infinity_norm, relative_tolerance))
    {
      // TODO: faudrait il que ce soit uniquement le process 0 de chaque bloque qui signal la convergence ?
      send_signal = CONVERGENCE_SIGNAL;
      if (rank_jacobi_block != BLOCK_RANK_ZERO && proc_local_rank == ZERO)
      {
        PetscCallMPI(MPI_Test(&send_signal_request, &send_signal_flag, MPI_STATUS_IGNORE));
        if (send_signal_flag)
        {
          MPI_Start(&send_signal_request);
        }
      }
    }

    // Checking on process rank 0 if any message of convergence arrived from other block
    if (rank_jacobi_block == BLOCK_RANK_ZERO && proc_local_rank == ZERO)
    {
      // message = ZERO;
      PetscCallMPI(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &rcv_signal_flag, &status));
      if (rcv_signal_flag && status.MPI_TAG == TAG_STATUS)
      {
        PetscCallMPI(MPI_Start(&rcv_signal_request));
        PetscCallMPI(MPI_Wait(&rcv_signal_request, &status));

        // if criterias are met, then send terminate message
        if (send_signal == CONVERGENCE_SIGNAL && rcv_signal == CONVERGENCE_SIGNAL)
        {
          PetscInt tmp_flag;
          PetscCallMPI(MPI_Testall(nprocs, send_requests, &tmp_flag, MPI_STATUSES_IGNORE));
          if (tmp_flag)
          {
            root_message = TERMINATE_SIGNAL;
            MPI_Startall(nprocs, send_requests);
          }
        }
      }
    }

    if (send_signal == CONVERGENCE_SIGNAL)
    {
      PetscCallMPI(MPI_Iprobe(MPI_ANY_SOURCE, TAG_TERMINATE, MPI_COMM_WORLD, &rcv_signal_flag, &status));
      if (rcv_signal_flag)
      {
        PetscCallMPI(MPI_Recv(&rcv_signal, ONE, MPIU_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        if (status.MPI_TAG == TAG_TERMINATE)
        {
          message = rcv_signal;
        }
      }
    }

    // The maximum value of signal is choosen, NO_SIGNAL < TERMINATE SIGNAL
    reduced_message = NO_MESSAGE;
    PetscCallMPI(MPI_Allreduce(&message, &reduced_message, ONE, MPIU_INT, MPI_MAX, comm_jacobi_block));

    number_of_iterations = number_of_iterations + 1;

  } while (reduced_message != TERMINATE_SIGNAL);

  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  end_time = MPI_Wtime();
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Elapsed time:   %f  seconds \n", end_time - start_time));

  PetscCallMPI(MPI_Test(&send_signal_request, &send_signal_flag, MPI_STATUS_IGNORE));
  while (!send_signal_flag)
  {
    PetscCallMPI(MPI_Cancel(&send_signal_request));
    PetscCallMPI(MPI_Test(&send_signal_request, &send_signal_flag, MPI_STATUS_IGNORE));
  }

  PetscCallMPI(MPI_Test(&send_request, &send_flag, MPI_STATUS_IGNORE));
  while (!send_flag)
  {
    PetscCallMPI(MPI_Cancel(&send_request));
    PetscCallMPI(MPI_Test(&send_request, &send_flag, MPI_STATUS_IGNORE));
  }

  PetscCallMPI(MPI_Test(&rcv_signal_request, &rcv_signal_flag, MPI_STATUS_IGNORE));
  while (!rcv_signal_flag)
  {
    PetscCallMPI(MPI_Cancel(&rcv_signal_request));
    PetscCallMPI(MPI_Test(&rcv_signal_request, &rcv_signal_flag, MPI_STATUS_IGNORE));
  }

  PetscCallMPI(MPI_Test(&rcv_request, &rcv_flag, MPI_STATUS_IGNORE));
  while (!rcv_flag)
  {
    PetscCallMPI(MPI_Cancel(&rcv_request));
    PetscCallMPI(MPI_Test(&rcv_request, &rcv_flag, MPI_STATUS_IGNORE));
  }

  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  if (rank_jacobi_block == BLOCK_RANK_ZERO)
  {
    PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
    memcpy(send_buffer, temp_buffer, vec_local_size * sizeof(PetscScalar));
    PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
    PetscCallMPI(MPI_Start(&send_request));
    PetscCallMPI(MPI_Wait(&send_request, MPI_STATUS_IGNORE));

    PetscCallMPI(MPI_Start(&rcv_request));
    PetscCallMPI(MPI_Wait(&rcv_request, MPI_STATUS_IGNORE));
    PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
    memcpy(temp_buffer, rcv_buffer, vec_local_size * sizeof(PetscScalar));
    PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
  }
  else if (rank_jacobi_block == BLOCK_RANK_ONE)
  {

    PetscCallMPI(MPI_Start(&rcv_request));
    PetscCallMPI(MPI_Wait(&rcv_request, MPI_STATUS_IGNORE));
    PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
    memcpy(temp_buffer, rcv_buffer, vec_local_size * sizeof(PetscScalar));
    PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_buffer));

    PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
    memcpy(send_buffer, temp_buffer, vec_local_size * sizeof(PetscScalar));
    PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
    PetscCallMPI(MPI_Start(&send_request));
    PetscCallMPI(MPI_Wait(&send_request, MPI_STATUS_IGNORE));
  }

  PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

  Vec local_residual = NULL;
  VecDuplicate(b_block_jacobi[rank_jacobi_block], &local_residual);
  PetscScalar local_residual_norm2;
  PetscScalar global_residual_norm2;
  PetscCall(MatResidual(A_block_jacobi, b_block_jacobi[rank_jacobi_block], x, local_residual));
  PetscCall(VecNorm(local_residual, NORM_2, &local_residual_norm2));
  local_residual_norm2 = local_residual_norm2 * local_residual_norm2;
  if (proc_local_rank != 0)
    local_residual_norm2 = 0.0;

  PetscCallMPI(MPI_Allreduce(&local_residual_norm2, &global_residual_norm2, 1, MPIU_SCALAR, MPI_SUM, MPI_COMM_WORLD));

  //////////////

  PetscCall(PetscPrintf(MPI_COMM_WORLD, " Total number of iterations: %d   ====  Direct norm 2 %e \n", number_of_iterations, (double)global_residual_norm2));
  // PetscPrintf(comm_jacobi_block, " Sleeping state ... block number [%d]\n", rank_jacobi_block);
  // PetscSleep(60 * 10);

  PetscCallMPI(MPI_Request_free(&rcv_request));
  PetscCallMPI(MPI_Request_free(&send_request));
  PetscCallMPI(MPI_Request_free(&send_signal_request));
  PetscCallMPI(MPI_Request_free(&rcv_signal_request));
  for (PetscInt proc_rank = ZERO; proc_rank < nprocs; proc_rank++)
  {
    PetscCallMPI(MPI_Request_free(&send_requests[proc_rank]));
  }

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

  PetscCall(PetscFree(send_requests));
  PetscCall(VecDestroy(&local_residual));
  PetscCall(VecDestroy(&x_block_jacobi_previous_iteration));
  PetscCall(VecDestroy(&approximation_residual));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x_initial_guess));
  PetscCall(MatDestroy(&A_block_jacobi));
  PetscCall(PetscFree(send_buffer));
  PetscCall(PetscFree(rcv_buffer));
  PetscCall(PCDestroy(&ksp_preconditionnner));
  PetscCall(KSPDestroy(&ksp));

  // Discard any pending message
  PetscInt count;

  do
  {
    MPI_Datatype data_type = MPIU_INT;
    PetscCallMPI(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &message, &status));
    if (message)
    {

      // printf("MESSAGE HERE ON %d PROCESSS\n", proc_global_rank);
      if (status.MPI_TAG == TAG_DATA)
      {
        data_type = MPIU_SCALAR;
        PetscCall(MPI_Get_count(&status, data_type, &count));
        PetscScalar *buffer;
        PetscCall(PetscMalloc1(count, &buffer));
        PetscCallMPI(MPI_Recv(buffer, count, data_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        PetscCall(PetscFree(buffer));
      }
      // if (status.MPI_TAG == TAG_STATUS || status.MPI_TAG == TAG_TERMINATE)
      else
      {
        data_type = MPIU_INT;
        PetscCall(MPI_Get_count(&status, data_type, &count));
        PetscInt *buffer;
        PetscCall(PetscMalloc1(count, &buffer));
        PetscCallMPI(MPI_Recv(buffer, count, data_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        PetscCall(PetscFree(buffer));
      }
    }
  } while (message);

  PetscCall(PetscCommDestroy(&comm_jacobi_block));
  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  PetscCall(PetscFinalize());

  return 0;
}

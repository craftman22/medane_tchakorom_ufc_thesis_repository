#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"
#include "constants.h"
#include "utils.h"

int main(int argc, char **argv)
{

  Mat A_block_jacobi = NULL;       // Operator matrix
  Vec x = NULL;                    // approximation solution at iteration (k)
  Vec x_previous_iteration = NULL; // approximation solution at iteration (k-1)
  Vec b = NULL;                    // right hand side vector
  Vec x_initial_guess = NULL;

  PetscMPIInt nprocs;
  PetscInt proc_global_rank;
  PetscInt n_mesh_lines = 4;
  PetscInt n_mesh_columns = 4;
  PetscInt njacobi_blocks;
  PetscInt rank_jacobi_block;
  PetscInt proc_local_rank;
  PetscInt n_mesh_points;
  PetscInt jacobi_block_size;

  PetscInt s;
  PetscInt nprocs_per_jacobi_block = 1;
  PetscScalar relative_tolerance = 1e-5;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

  // Getting applications arguments
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &n_mesh_lines, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n_mesh_columns, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-s", &s, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npb", &nprocs_per_jacobi_block, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &relative_tolerance, NULL));

  // PetscPrintf(PETSC_COMM_WORLD, " =====> Total number of processes: %d \n =====>s : %d\n =====>nprocessor_per_jacobi_block : %d \n ====> Grid lines: %d \n ====> Grid columns : %d \n", nprocs, s, nprocs_per_jacobi_block, n_mesh_lines, n_mesh_columns);

  PetscCall(computeDimensionRelatedVariables(nprocs, nprocs_per_jacobi_block, proc_global_rank, n_mesh_lines, n_mesh_columns, &njacobi_blocks, &rank_jacobi_block, &proc_local_rank, &n_mesh_points, &jacobi_block_size));
  PetscAssert((n_mesh_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");

  // Creating the sub communicator for each jacobi block
  PetscSubcomm sub_comm_context = NULL;
  MPI_Comm dcomm;
  PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD, &dcomm, NULL));
  PetscCall(PetscSubcommCreate(dcomm, &sub_comm_context));
  PetscCall(PetscSubcommSetNumber(sub_comm_context, njacobi_blocks));
  PetscCall(PetscSubcommSetType(sub_comm_context, PETSC_SUBCOMM_CONTIGUOUS));
  PetscCall(PetscSubcommSetFromOptions(sub_comm_context));
  MPI_Comm comm_jacobi_block = PetscSubcommChild(sub_comm_context);

  // Vector of unknowns
  PetscCall(create_vector(comm_jacobi_block, &x, n_mesh_points, VECMPI));

  // approximation solution at iteration (k-1)
  PetscCall(VecDuplicate(x, &x_previous_iteration));

  // Right hand side
  PetscCall(VecDuplicate(x, &b));

  // Initial guess solution
  PetscCall(VecDuplicate(x, &x_initial_guess));
  PetscScalar initial_scalar_value = 1.0;
  PetscCall(VecSet(x_initial_guess, initial_scalar_value));

  // Operator matrix
  PetscCall(create_matrix_sparse(comm_jacobi_block, &A_block_jacobi, n_mesh_points / njacobi_blocks, n_mesh_points, MATMPIAIJ, 5, 5));

  // Insert non-zeros values into the sparse operator matrix
  PetscCall(poisson2DMatrix(&A_block_jacobi, n_mesh_lines, n_mesh_columns, rank_jacobi_block, njacobi_blocks));

  Mat A_block_jacobi_subMat[njacobi_blocks];
  IS is_cols_block_jacobi[njacobi_blocks];
  Vec b_block_jacobi[njacobi_blocks];
  Vec x_block_jacobi[njacobi_blocks];

  // domain decomposition of matrix and vectors
  PetscCall(divideSubDomainIntoBlockMatrices(comm_jacobi_block, A_block_jacobi, A_block_jacobi_subMat, is_cols_block_jacobi, rank_jacobi_block, njacobi_blocks, proc_local_rank, nprocs_per_jacobi_block));

  for (PetscInt i = 0; i < njacobi_blocks; i++)
  {
    PetscCall(create_vector(comm_jacobi_block, &x_block_jacobi[i], jacobi_block_size, VECMPI));
    PetscCall(create_vector(comm_jacobi_block, &b_block_jacobi[i], jacobi_block_size, VECMPI));
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

  PetscBool stop_condition = PETSC_FALSE;
  PetscInt number_of_iterations = 0;
  PetscInt idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;
  PetscScalar approximation_residual_infinity_norm = PETSC_MAX_REAL;

  KSP inner_ksp = NULL;
  PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, INNER_KSP_PREFIX, INNER_PC_PREFIX));

  PetscScalar *send_multisplitting_data_buffer = NULL;
  PetscScalar *rcv_multisplitting_data_buffer = NULL;
  PetscScalar *temp_multisplitting_data_buffer = NULL;
  // MPI_Request send_request = MPI_REQUEST_NULL;
  // MPI_Request rcv_request = MPI_REQUEST_NULL;
  PetscInt vec_local_size = 0;
  PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
  PetscMalloc1((size_t)vec_local_size, &send_multisplitting_data_buffer);
  PetscMalloc1((size_t)vec_local_size, &rcv_multisplitting_data_buffer);
  Vec approximation_residual;
  PetscCall(VecDuplicate(x, &approximation_residual));

  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  double start_time, end_time;
  start_time = MPI_Wtime();

  do
  {
    PetscCall(VecCopy(x, x_previous_iteration)); // copy approximation solution at iteration k into approximation solution at iteration (k-1)
    PetscCall(inner_solver(inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, rank_jacobi_block, NULL,number_of_iterations));

    if (rank_jacobi_block == BLOCK_RANK_ZERO)
    {

      PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_multisplitting_data_buffer));
      PetscCall(PetscArraycpy(send_multisplitting_data_buffer, temp_multisplitting_data_buffer, vec_local_size));
      PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_multisplitting_data_buffer));

      PetscCallMPI(MPI_Sendrecv(send_multisplitting_data_buffer, vec_local_size, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 0, rcv_multisplitting_data_buffer, vec_local_size, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

      PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_multisplitting_data_buffer));
      PetscCall(PetscArraycpy(temp_multisplitting_data_buffer, rcv_multisplitting_data_buffer, vec_local_size));
      PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_multisplitting_data_buffer));
    }
    else if (rank_jacobi_block == BLOCK_RANK_ONE)
    {
      PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_multisplitting_data_buffer));
      PetscCall(PetscArraycpy(send_multisplitting_data_buffer, temp_multisplitting_data_buffer, vec_local_size));
      PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_multisplitting_data_buffer));

      PetscCallMPI(MPI_Sendrecv(send_multisplitting_data_buffer, vec_local_size, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 1, rcv_multisplitting_data_buffer, vec_local_size, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

      PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_multisplitting_data_buffer));
      PetscCall(PetscArraycpy(temp_multisplitting_data_buffer, rcv_multisplitting_data_buffer, vec_local_size));
      PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_multisplitting_data_buffer));
    }

    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(VecWAXPY(approximation_residual, -1, x_previous_iteration, x));
    PetscCall(VecNorm(approximation_residual, NORM_INFINITY, &approximation_residual_infinity_norm));

    PetscCall(printResidualNorm(approximation_residual_infinity_norm));

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
  PetscCall(VecDestroy(&x_previous_iteration));
  PetscCall(MatDestroy(&A_block_jacobi));
  PetscCall(PetscFree(send_multisplitting_data_buffer));
  PetscCall(PetscFree(rcv_multisplitting_data_buffer));
  PetscCall(KSPDestroy(&inner_ksp));

  PetscCall(PetscFinalize());
  return 0;
}

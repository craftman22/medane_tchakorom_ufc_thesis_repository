#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"
#include "constants.h"
#include "utils.h"

int main(int argc, char **argv)
{

  Mat A_block_jacobi = NULL; // Operator matrix
  Vec x = NULL;              // approximation solution at iteration (k)
  Vec b = NULL;              // right hand side vector
  Vec x_initial_guess = NULL;

  PetscMPIInt nprocs;
  PetscMPIInt proc_global_rank;
  PetscInt n_mesh_lines = 4;
  PetscInt n_mesh_columns = 4;
  PetscInt njacobi_blocks;
  PetscMPIInt rank_jacobi_block;
  PetscInt proc_local_rank;
  PetscInt n_mesh_points;
  PetscInt jacobi_block_size;
  PetscInt s;
  PetscScalar relative_tolerance = 1e-5;
  PetscMPIInt nprocs_per_jacobi_block = 1;

  PetscInt MIN_CONVERGENCE_COUNT = 5;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

  // Getting applications arguments
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &n_mesh_lines, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n_mesh_columns, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-s", &s, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-min_convergence_count", &MIN_CONVERGENCE_COUNT, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npb", &nprocs_per_jacobi_block, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &relative_tolerance, NULL));

  // PetscPrintf(PETSC_COMM_WORLD, " =====> Total number of processes: %d \n =====>s : %d\n =====>nprocessor_per_jacobi_block : %d \n ====> Grid lines: %d \n ====> Grid columns : %d ====> Relative tolerance : %f\n", nprocs, s, nprocs_per_jacobi_block, n_mesh_lines, n_mesh_columns, relative_tolerance);

  PetscCall(computeDimensionRelatedVariables(nprocs, nprocs_per_jacobi_block, proc_global_rank, n_mesh_lines, n_mesh_columns, &njacobi_blocks, &rank_jacobi_block, &proc_local_rank, &n_mesh_points, &jacobi_block_size));

  PetscAssert((n_mesh_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");

  // Creating the sub communicator for each jacobi block
  PetscSubcomm sub_comm_context = NULL;
  MPI_Comm dcomm;
  PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD, &dcomm, NULL));

  PetscCall(PetscSubcommCreate(dcomm, &sub_comm_context));
  PetscCall(PetscSubcommSetNumber(sub_comm_context, njacobi_blocks));
  PetscCall(PetscSubcommSetType(sub_comm_context, PETSC_SUBCOMM_CONTIGUOUS));
  // PetscCall(PetscSubcommSetTypeGeneral(sub_comm_context, rank_jacobi_block, proc_local_rank));
  // PetscCall(PetscSubcommSetFromOptions(sub_comm_context));
  MPI_Comm comm_jacobi_block = PetscSubcommChild(sub_comm_context);

  PetscMPIInt send_signal = NO_SIGNAL;
  PetscMPIInt rcv_signal = NO_SIGNAL;

  // Vector of unknowns
  PetscCall(VecCreate(comm_jacobi_block, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n_mesh_points));
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
    PetscCall(create_vector(comm_jacobi_block, &b_block_jacobi[i], jacobi_block_size, VECMPI));
    PetscCall(create_vector(comm_jacobi_block, &x_block_jacobi[i], jacobi_block_size, VECMPI));
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
  PetscMPIInt idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;
  PetscScalar approximation_residual_infinity_norm = PETSC_MAX_REAL;

  KSP inner_ksp = NULL;
  PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, INNER_KSP_PREFIX, INNER_PC_PREFIX));

  PetscMPIInt vec_local_size = 0;
  PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
  PetscScalar *send_buffer = NULL;
  PetscScalar *rcv_buffer = NULL;
  PetscScalar *temp_buffer = NULL;
  PetscMalloc1(vec_local_size, &send_buffer);
  PetscMalloc1(vec_local_size, &rcv_buffer);

  Vec approximation_residual;
  PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &approximation_residual));

  MPI_Status status;
  PetscMPIInt broadcast_message = NO_MESSAGE;
  PetscMPIInt send_data_flag = 0;
  PetscMPIInt rcv_data_flag = 0;
  PetscMPIInt send_signal_flag = 0;
  PetscMPIInt rcv_signal_flag = 0;

  // MPI_Request rcv_data_request = MPI_REQUEST_NULL;
  MPI_Request send_data_request = MPI_REQUEST_NULL;
  // MPI_Status rcv_status;
  // MPI_Request request;
  MPI_Request send_signal_request = MPI_REQUEST_NULL;
  // MPI_Request rcv_signal_request = MPI_REQUEST_NULL;

  PetscInt inner_solver_iterations = 0;

  Vec x_block_jacobi_previous_iteration = NULL;
  PetscCall(VecDuplicate(x_block_jacobi[rank_jacobi_block], &x_block_jacobi_previous_iteration));

  PetscMPIInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
  PetscMPIInt message_dest = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;

  PetscInt convergence_count = ZERO;
  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  double start_time, end_time;
  start_time = MPI_Wtime();

  do
  {

    PetscCallMPI(MPI_Iprobe(message_source, (TAG_MULTISPLITTING_DATA + idx_non_current_block), MPI_COMM_WORLD, &rcv_data_flag, MPI_STATUS_IGNORE));
    if (rcv_data_flag)
    {
      do
      {
        PetscCallMPI(MPI_Recv(rcv_buffer,vec_local_size,MPIU_SCALAR, message_source, (TAG_MULTISPLITTING_DATA + idx_non_current_block), MPI_COMM_WORLD,MPI_STATUS_IGNORE));
        PetscCallMPI(MPI_Iprobe(message_source, (TAG_MULTISPLITTING_DATA + idx_non_current_block), MPI_COMM_WORLD, &rcv_data_flag, MPI_STATUS_IGNORE));
      } while (rcv_data_flag);
      PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
      PetscCall(PetscArraycpy(temp_buffer, rcv_buffer, vec_local_size));
      PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
    }

    PetscCall(inner_solver(inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, rank_jacobi_block, &inner_solver_iterations, number_of_iterations));

    MPI_Test(&send_data_request, &send_data_flag, MPI_STATUS_IGNORE);
    if (send_data_flag)
    {
      PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
      PetscCall(PetscArraycpy(send_buffer, temp_buffer, vec_local_size));
      PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
      PetscCallMPI(MPI_Isend(send_buffer, vec_local_size, MPIU_SCALAR, message_dest, TAG_MULTISPLITTING_DATA + rank_jacobi_block, MPI_COMM_WORLD, &send_data_request));
    }
    

    PetscCallMPI(MPI_Iprobe(message_source, (TAG_MULTISPLITTING_DATA + idx_non_current_block), MPI_COMM_WORLD, &rcv_data_flag, MPI_STATUS_IGNORE));
    if (rcv_data_flag)
    {
      do
      {
        PetscCallMPI(MPI_Recv(rcv_buffer,vec_local_size,MPIU_SCALAR, message_source, (TAG_MULTISPLITTING_DATA + idx_non_current_block), MPI_COMM_WORLD,MPI_STATUS_IGNORE));
        PetscCallMPI(MPI_Iprobe(message_source, (TAG_MULTISPLITTING_DATA + idx_non_current_block), MPI_COMM_WORLD, &rcv_data_flag, MPI_STATUS_IGNORE));
      } while (rcv_data_flag);
      PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
      PetscCall(PetscArraycpy(temp_buffer, rcv_buffer, vec_local_size));
      PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_buffer));
    }

    

    PetscCall(VecWAXPY(approximation_residual, -1.0, x_block_jacobi_previous_iteration, x_block_jacobi[rank_jacobi_block]));
    PetscCall(VecNorm(approximation_residual, NORM_INFINITY, &approximation_residual_infinity_norm));
    PetscCall(VecCopy(x_block_jacobi[rank_jacobi_block], x_block_jacobi_previous_iteration));
    PetscCall(printResidualNorm(comm_jacobi_block, rank_jacobi_block, approximation_residual_infinity_norm,number_of_iterations));

    if (PetscApproximateLTE(approximation_residual_infinity_norm, relative_tolerance))
      convergence_count++;
    else
      convergence_count = ZERO;

    if (proc_local_rank == ZERO)
    {
      if (convergence_count >= MIN_CONVERGENCE_COUNT)
        send_signal = CONVERGENCE_SIGNAL;
      else
        send_signal = NO_SIGNAL;

      PetscCallMPI(MPI_Test(&send_signal_request, &send_signal_flag, MPI_STATUS_IGNORE));
      if (send_signal_flag)
      {
        PetscCallMPI(MPI_Isend(&send_signal, ONE, MPIU_INT, message_dest, TAG_STATUS + rank_jacobi_block, MPI_COMM_WORLD, &send_signal_request));
      }
    }

    if (proc_local_rank == ZERO)
    {
      PetscCallMPI(MPI_Iprobe(message_source, TAG_STATUS + idx_non_current_block, MPI_COMM_WORLD, &rcv_signal_flag, &status));
      if (rcv_signal_flag)
      {
        PetscCallMPI(MPI_Recv(&rcv_signal, ONE, MPIU_INT, message_source, TAG_STATUS + idx_non_current_block, MPI_COMM_WORLD,MPI_STATUS_IGNORE));
      }
    }

    if (proc_local_rank == ZERO)
    {
      if (send_signal == CONVERGENCE_SIGNAL && rcv_signal == CONVERGENCE_SIGNAL)
      {
        broadcast_message = TERMINATE_SIGNAL;
      }
    }
    PetscCallMPI(MPI_Bcast(&broadcast_message, ONE, MPIU_INT, proc_local_rank, comm_jacobi_block));

    number_of_iterations = number_of_iterations + 1;

  } while (broadcast_message != TERMINATE_SIGNAL);

  // printf("Arrived at the end block %d rank %d\n", rank_jacobi_block, proc_local_rank);
  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  end_time = MPI_Wtime();
  PetscCall(printElapsedTime(start_time, end_time));
  PetscCall(printTotalNumberOfIterations(comm_jacobi_block, rank_jacobi_block, number_of_iterations));

  PetscScalar *send_multisplitting_data_bis = NULL;
  PetscScalar *rcv_multisplitting_data_bis = NULL;
  PetscScalar *temp_mutlisplitting_bis = NULL;
  PetscMalloc1(vec_local_size, &send_multisplitting_data_bis);
  PetscMalloc1(vec_local_size, &rcv_multisplitting_data_bis);

  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));

  PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_mutlisplitting_bis));
  PetscCall(PetscArraycpy(send_multisplitting_data_bis, temp_mutlisplitting_bis, vec_local_size));
  PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_mutlisplitting_bis));

  PetscCallMPI(MPI_Sendrecv(send_multisplitting_data_bis, vec_local_size, MPIU_SCALAR, message_dest, TAG_FINAL_DATA_EXCHANGE + rank_jacobi_block, rcv_multisplitting_data_bis, vec_local_size, MPIU_SCALAR, message_source, TAG_FINAL_DATA_EXCHANGE + idx_non_current_block, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

  PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_mutlisplitting_bis));
  PetscCall(PetscArraycpy(temp_mutlisplitting_bis, rcv_multisplitting_data_bis, vec_local_size));
  PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_mutlisplitting_bis));

  PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

  PetscScalar direct_residual_norm;
  PetscCall(computeFinalResidualNorm(A_block_jacobi, &x, b_block_jacobi, rank_jacobi_block, proc_global_rank, &direct_residual_norm));
  PetscCall(printFinalResidualNorm(direct_residual_norm));

  // END OF PROGRAM  - FREE MEMORY

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

  PetscCall(VecDestroy(&x_block_jacobi_previous_iteration));
  PetscCall(VecDestroy(&approximation_residual));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x_initial_guess));
  PetscCall(MatDestroy(&A_block_jacobi));
  PetscCall(KSPDestroy(&inner_ksp));

  PetscCall(PetscFree(send_multisplitting_data_bis));
  PetscCall(PetscFree(rcv_multisplitting_data_bis));

  // Discard any pending message
  PetscInt count;
  PetscInt message = NO_MESSAGE;

  do
  {
    MPI_Datatype data_type = MPIU_INT;
    PetscCallMPI(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &message, &status));
    if (message)
    {
      if (status.MPI_TAG == (TAG_MULTISPLITTING_DATA + idx_non_current_block))
      {
        data_type = MPIU_SCALAR;
        PetscCall(MPI_Get_count(&status, data_type, &count));
        PetscScalar *buffer;
        PetscCall(PetscMalloc1(count, &buffer));
        PetscCallMPI(MPI_Recv(buffer, count, data_type, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        PetscCall(PetscFree(buffer));
      }
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

  PetscCallMPI(MPI_Wait(&send_data_request, MPI_STATUS_IGNORE));
  PetscCallMPI(MPI_Wait(&send_signal_request, MPI_STATUS_IGNORE));
  PetscCall(PetscFree(send_buffer));
  PetscCall(PetscFree(rcv_buffer));

  PetscCall(PetscSubcommDestroy(&sub_comm_context));
  PetscCall(PetscCommDestroy(&dcomm));
  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
  PetscCall(PetscFinalize());

  return 0;
}

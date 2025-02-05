#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"
#include "constants.h"
#include "utils.h"
#include "petscdraw.h"
#include "petscviewer.h"

int main(int argc, char **argv)
{

    Mat A_block_jacobi = NULL;
    Mat A_block_jacobi_resdistributed = NULL;
    Vec x = NULL; // vector of unknows
    Vec b = NULL; // right hand side vector
    Vec x_initial_guess = NULL;
    PetscInt s;
    PetscInt nprocs;
    PetscInt proc_global_rank;
    PetscInt n_mesh_lines = 4;
    PetscInt n_mesh_columns = 4;
    PetscInt njacobi_blocks;
    PetscInt rank_jacobi_block;
    PetscInt proc_local_rank;
    PetscInt n_mesh_points;
    PetscInt jacobi_block_size;
    PetscInt nprocs_per_jacobi_block = 1;
    PetscScalar relative_tolerance = 1e-5;
    PetscSubcomm sub_comm_context;
    MPI_Comm dcomm;
    MPI_Comm comm_jacobi_block;
    PetscInt send_signal = NO_SIGNAL;
    PetscInt rcv_signal = NO_SIGNAL;
    PetscInt send_signal_flag = NO_SIGNAL;
    PetscInt rcv_signal_flag = NO_SIGNAL;
    // PetscInt rcv_signal = NO_SIGNAL;

    IS is_jacobi_vec_parts;
    PetscInt number_of_iterations;
    PetscInt idx_non_current_block;
    PetscScalar approximation_residual_infinity_norm;
    KSP inner_ksp = NULL;
    KSP outer_ksp = NULL;
    PetscInt vec_local_size = 0;
    PetscScalar *send_multisplitting_data_buffer = NULL;
    PetscScalar *rcv_multisplitting_data_buffer = NULL;
    PetscScalar *temp_multisplitting_data_buffer = NULL;
    PetscScalar *send_minimization_data_buffer = NULL;
    PetscScalar *rcv_minimization_data_buffer = NULL;
    PetscScalar *temp_minimization_data_buffer = NULL;

    MPI_Request send_multisplitting_data_request = MPI_REQUEST_NULL;
    MPI_Request rcv_multisplitting_data_request = MPI_REQUEST_NULL;
    MPI_Request send_minimization_data_request = MPI_REQUEST_NULL;
    MPI_Request rcv_minimization_data_request = MPI_REQUEST_NULL;

    PetscMPIInt send_multisplitting_data_flag = 0;
    PetscMPIInt rcv_multisplitting_data_flag = 0;
    PetscMPIInt send_minimization_data_flag = 0;
    PetscMPIInt rcv_minimization_data_flag = 0;

    PetscInt *vec_local_idx = NULL;
    PetscInt x_local_size;
    PetscScalar *vector_to_insert_into_S;

    MPI_Request send_signal_request = MPI_REQUEST_NULL;
    MPI_Request rcv_signal_request = MPI_REQUEST_NULL;

    MPI_Status status;

    // Minimization variables

    Mat R = NULL;
    Mat S = NULL;
    PetscInt n_vectors_inserted;
    Vec x_minimized = NULL;
    Vec x_minimized_prev_iteration = NULL;
    Vec approximate_residual = NULL;

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

    PetscCall(computeDimensionRelatedVariables(nprocs, nprocs_per_jacobi_block, proc_global_rank, n_mesh_lines, n_mesh_columns, &njacobi_blocks, &rank_jacobi_block, &proc_local_rank, &n_mesh_points, &jacobi_block_size));
    PetscAssert((n_mesh_points % nprocs == 0), PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of grid points should be divisible by the number of procs \n Programm exit ...\n");
    // Creating the sub communicator for each jacobi block
    sub_comm_context = NULL;
    PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD, &dcomm, NULL));
    PetscCall(PetscSubcommCreate(dcomm, &sub_comm_context));
    PetscCall(PetscSubcommSetNumber(sub_comm_context, njacobi_blocks));
    PetscCall(PetscSubcommSetType(sub_comm_context, PETSC_SUBCOMM_CONTIGUOUS));
    comm_jacobi_block = PetscSubcommChild(sub_comm_context);

    IS is_cols_block_jacobi[njacobi_blocks];
    Mat A_block_jacobi_subMat[njacobi_blocks];
    Mat R_block_jacobi_subMat[njacobi_blocks];
    Vec b_block_jacobi[njacobi_blocks];
    Vec x_block_jacobi[njacobi_blocks];
    VecScatter scatter_jacobi_vec_part_to_merged_vec[njacobi_blocks];
    IS is_merged_vec[njacobi_blocks];
    Mat R_transpose_R = NULL;
    Vec vec_R_transpose_b_block_jacobi = NULL;
    Vec alpha = NULL;

    idx_non_current_block = (rank_jacobi_block == ZERO) ? ONE : ZERO;

    for (PetscInt i = 0; i < njacobi_blocks; i++)
    {
        PetscCall(create_vector(comm_jacobi_block, &b_block_jacobi[i], jacobi_block_size, VECMPI));
        PetscCall(create_vector(comm_jacobi_block, &x_block_jacobi[i], jacobi_block_size, VECMPI));
    }

    PetscCall(create_matrix_sparse(comm_jacobi_block, &A_block_jacobi, n_mesh_points / njacobi_blocks, n_mesh_points, MATMPIAIJ, 5, 5));
    // Insert non-zeros entries into the operator matrix
    PetscCall(poisson2DMatrix(&A_block_jacobi, n_mesh_lines, n_mesh_columns, rank_jacobi_block, njacobi_blocks));

    PetscCall(create_matrix_dense(comm_jacobi_block, &R, n_mesh_points, s, MATMPIDENSE));
    MatZeroEntries(R);
    MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY);

    PetscCall(getHalfSubMatrixFromR(R, R_block_jacobi_subMat, n_mesh_lines, n_mesh_columns, rank_jacobi_block));

    PetscInt R_local_size;
    PetscCall(MatGetLocalSize(R, &R_local_size, NULL));
    PetscInt R_local_values_count = s * R_local_size;
    PetscMalloc1((size_t)R_local_values_count, &send_minimization_data_buffer);
    PetscMalloc1((size_t)R_local_values_count, &rcv_minimization_data_buffer);

    PetscInt redistributed_local_size;
    PetscInt first_row_owned;
    PetscCall(MatGetLocalSize(R_block_jacobi_subMat[rank_jacobi_block], &redistributed_local_size, NULL));
    PetscCall(MatGetOwnershipRange(R_block_jacobi_subMat[rank_jacobi_block], &first_row_owned, NULL));
    PetscCall(create_redistributed_A_block_jacobi(comm_jacobi_block, A_block_jacobi, &A_block_jacobi_resdistributed, nprocs_per_jacobi_block, proc_local_rank, redistributed_local_size, first_row_owned));

    PetscCall(restoreHalfSubMatrixToR(R, R_block_jacobi_subMat, rank_jacobi_block));

    PetscCall(create_matrix_dense(comm_jacobi_block, &S, n_mesh_points, s, MATMPIDENSE));

    PetscCall(create_matrix_dense(comm_jacobi_block, &R_transpose_R, s, s, MATMPIDENSE));

    PetscCall(create_vector(comm_jacobi_block, &vec_R_transpose_b_block_jacobi, s, VECMPI));
    PetscCall(create_vector(comm_jacobi_block, &alpha, s, VECMPI));
    PetscCall(create_vector(comm_jacobi_block, &x, n_mesh_points, VECMPI));

    PetscCall(VecDuplicate(x, &b));
    PetscCall(VecDuplicate(x, &x_initial_guess));
    PetscCall(VecSet(x_initial_guess, ONE));

    PetscCall(divideSubDomainIntoBlockMatrices(comm_jacobi_block, A_block_jacobi, A_block_jacobi_subMat, is_cols_block_jacobi, rank_jacobi_block, njacobi_blocks, proc_local_rank, nprocs_per_jacobi_block));

    // creation of a scatter context to manage data transfert between complete b or x , and their part x_block_jacobi[..] and b_block_jacobi[...]

    PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, ZERO, ONE, &is_jacobi_vec_parts));

    for (PetscInt i = 0; i < njacobi_blocks; i++)
    {
        PetscCall(ISCreateStride(comm_jacobi_block, jacobi_block_size, (i * (jacobi_block_size)), ONE, &is_merged_vec[i]));
        PetscCall(VecScatterCreate(b_block_jacobi[i], is_jacobi_vec_parts, b, is_merged_vec[i], &scatter_jacobi_vec_part_to_merged_vec[i]));
    }

    PetscCall(computeTheRightHandSideWithInitialGuess(comm_jacobi_block, scatter_jacobi_vec_part_to_merged_vec, A_block_jacobi, &b, b_block_jacobi, x_initial_guess, rank_jacobi_block, jacobi_block_size, nprocs_per_jacobi_block, proc_local_rank));

    number_of_iterations = 0;

    approximation_residual_infinity_norm = PETSC_MAX_REAL;
    PetscCall(initializeKSP(comm_jacobi_block, &inner_ksp, A_block_jacobi_subMat[rank_jacobi_block], rank_jacobi_block, PETSC_FALSE, INNER_KSP_PREFIX, INNER_PC_PREFIX));
    PetscCall(initializeKSP(comm_jacobi_block, &outer_ksp, NULL, rank_jacobi_block, PETSC_TRUE, OUTER_KSP_PREFIX, OUTER_PC_PREFIX));

    PetscCall(VecGetLocalSize(x_block_jacobi[rank_jacobi_block], &vec_local_size));
    PetscMalloc1((size_t)vec_local_size, &send_multisplitting_data_buffer);
    PetscMalloc1((size_t)vec_local_size, &rcv_multisplitting_data_buffer);

    PetscCall(create_vector(comm_jacobi_block, &x_minimized, n_mesh_points, VECMPI));
    PetscCall(VecSet(x_minimized, ZERO));

    // Initialize x_minimized_prev_iteration
    PetscCall(VecDuplicate(x_minimized, &x_minimized_prev_iteration));

    PetscCall(VecGetLocalSize(x, &x_local_size));
    // vec_local_idx = (PetscInt *)malloc(x_local_size * sizeof(PetscInt));
    PetscMalloc1(x_local_size,&vec_local_idx);
    for (PetscInt i = 0; i < (x_local_size); i++)
    {
        vec_local_idx[i] = (proc_local_rank * x_local_size) + i;
    }
    // vector_to_insert_into_S = (PetscScalar *)malloc(x_local_size * sizeof(PetscScalar));
    PetscMalloc1(x_local_size,&vector_to_insert_into_S);

    PetscCall(VecDuplicate(x_minimized, &approximate_residual));

    PetscMPIInt message_source = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
    PetscMPIInt message_destination = (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank;
    PetscInt convergence_count = ZERO;
    PetscInt broadcast_message = NO_MESSAGE;

    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    double start_time, end_time;
    start_time = MPI_Wtime();

    do
    {

        n_vectors_inserted = 0;
        PetscCall(VecCopy(x_minimized, x_minimized_prev_iteration));

        while (n_vectors_inserted < s)
        {
            PetscCall(inner_solver(inner_ksp, A_block_jacobi_subMat, x_block_jacobi, b_block_jacobi, rank_jacobi_block, NULL));

            if (rank_jacobi_block == BLOCK_RANK_ZERO)
            {
                PetscCallMPI(MPI_Test(&send_multisplitting_data_request, &send_multisplitting_data_flag, MPI_STATUS_IGNORE));
                if (send_multisplitting_data_flag)
                {
                    PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_multisplitting_data_buffer));
                    PetscCall(PetscArraycpy(send_multisplitting_data_buffer, temp_multisplitting_data_buffer, vec_local_size));
                    PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_multisplitting_data_buffer));
                    PetscCallMPI(MPI_Isend(send_multisplitting_data_buffer, vec_local_size, MPIU_SCALAR, message_destination, 0, MPI_COMM_WORLD, &send_multisplitting_data_request));
                    // PetscCallMPI(MPI_Wait(&send_multisplitting_data_request, MPI_STATUS_IGNORE));
                }

                PetscCallMPI(MPI_Iprobe(message_source, 1, MPI_COMM_WORLD, &rcv_multisplitting_data_flag, MPI_STATUS_IGNORE));
                if (rcv_multisplitting_data_flag)
                {
                    PetscCallMPI(MPI_Irecv(rcv_multisplitting_data_buffer, vec_local_size, MPIU_SCALAR, message_source, 1, MPI_COMM_WORLD, &rcv_multisplitting_data_request));
                    PetscCallMPI(MPI_Wait(&rcv_multisplitting_data_request, MPI_STATUS_IGNORE));
                    PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_multisplitting_data_buffer));
                    PetscCall(PetscArraycpy(temp_multisplitting_data_buffer, rcv_multisplitting_data_buffer, vec_local_size));
                    PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_multisplitting_data_buffer));
                }
            }
            else if (rank_jacobi_block == BLOCK_RANK_ONE)
            {

                PetscCallMPI(MPI_Iprobe(message_source, 0, MPI_COMM_WORLD, &rcv_multisplitting_data_flag, MPI_STATUS_IGNORE));
                if (rcv_multisplitting_data_flag)
                {
                    PetscCallMPI(MPI_Irecv(rcv_multisplitting_data_buffer, vec_local_size, MPIU_SCALAR, message_source, 0, MPI_COMM_WORLD, &rcv_multisplitting_data_request));
                    PetscCallMPI(MPI_Wait(&rcv_multisplitting_data_request, MPI_STATUS_IGNORE));
                    PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &temp_multisplitting_data_buffer));
                    PetscCall(PetscArraycpy(temp_multisplitting_data_buffer, rcv_multisplitting_data_buffer, vec_local_size));
                    PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &temp_multisplitting_data_buffer));
                }

                PetscCallMPI(MPI_Test(&send_multisplitting_data_request, &send_multisplitting_data_flag, MPI_STATUS_IGNORE));
                if (send_multisplitting_data_flag)
                {
                    PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_multisplitting_data_buffer));
                    PetscCall(PetscArraycpy(send_multisplitting_data_buffer, temp_multisplitting_data_buffer, vec_local_size));
                    PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_multisplitting_data_buffer));
                    PetscCallMPI(MPI_Isend(send_multisplitting_data_buffer, vec_local_size, MPIU_SCALAR, message_destination, 1, MPI_COMM_WORLD, &send_multisplitting_data_request));
                    // PetscCallMPI(MPI_Wait(&send_multisplitting_data_request, MPI_STATUS_IGNORE));
                }
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

        PetscCall(getHalfSubMatrixFromR(R, R_block_jacobi_subMat, n_mesh_lines, n_mesh_columns, rank_jacobi_block));
        PetscCall(MatMatMult(A_block_jacobi_resdistributed, S, MAT_REUSE_MATRIX, PETSC_DETERMINE, &R_block_jacobi_subMat[rank_jacobi_block]));
        PetscCall(restoreHalfSubMatrixToR(R, R_block_jacobi_subMat, rank_jacobi_block));

        /////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (rank_jacobi_block == BLOCK_RANK_ZERO)
        {
            if (proc_local_rank < (nprocs_per_jacobi_block / 2))
            {
                PetscCallMPI(MPI_Test(&send_minimization_data_request, &send_minimization_data_flag, MPI_STATUS_IGNORE));
                if (send_minimization_data_flag)
                {
                    PetscCall(MatDenseGetArrayWrite(R, &temp_minimization_data_buffer));
                    PetscCall(PetscArraycpy(send_minimization_data_buffer, temp_minimization_data_buffer, R_local_values_count));
                    PetscCall(MatDenseRestoreArrayWrite(R, &temp_minimization_data_buffer));
                    PetscCallMPI(MPI_Isend(send_minimization_data_buffer, R_local_values_count, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 2, MPI_COMM_WORLD, &send_minimization_data_request));
                    // PetscCallMPI(MPI_Wait(&send_minimization_data_request, MPI_STATUS_IGNORE));
                }
            }
            else
            {
                PetscCallMPI(MPI_Iprobe(message_source, 3, MPI_COMM_WORLD, &rcv_minimization_data_flag, MPI_STATUS_IGNORE));
                if (rcv_minimization_data_flag)
                {
                    PetscCallMPI(MPI_Irecv(rcv_minimization_data_buffer, R_local_values_count, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 3, MPI_COMM_WORLD, &rcv_minimization_data_request));
                    PetscCallMPI(MPI_Wait(&rcv_minimization_data_request, MPI_STATUS_IGNORE));
                    PetscCall(MatDenseGetArrayWrite(R, &temp_minimization_data_buffer));
                    PetscCall(PetscArraycpy(temp_minimization_data_buffer, rcv_minimization_data_buffer, R_local_values_count));
                    PetscCall(MatDenseRestoreArrayWrite(R, &temp_minimization_data_buffer));
                }
            }
        }
        else if (rank_jacobi_block == BLOCK_RANK_ONE)
        {
            if (proc_local_rank < (nprocs_per_jacobi_block / 2))
            {
                PetscCallMPI(MPI_Iprobe(message_source, 2, MPI_COMM_WORLD, &rcv_minimization_data_flag, MPI_STATUS_IGNORE));
                if (rcv_minimization_data_flag)
                {
                    PetscCallMPI(MPI_Irecv(rcv_minimization_data_buffer, R_local_values_count, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 2, MPI_COMM_WORLD, &rcv_minimization_data_request));
                    PetscCallMPI(MPI_Wait(&rcv_minimization_data_request, MPI_STATUS_IGNORE));
                    PetscCall(MatDenseGetArrayWrite(R, &temp_minimization_data_buffer));
                    PetscCall(PetscArraycpy(temp_minimization_data_buffer, rcv_minimization_data_buffer, R_local_values_count));
                    PetscCall(MatDenseRestoreArrayWrite(R, &temp_minimization_data_buffer));
                }
            }
            else
            {
                PetscCallMPI(MPI_Test(&send_minimization_data_request, &send_minimization_data_flag, MPI_STATUS_IGNORE));
                if (send_minimization_data_flag)
                {
                    PetscCall(MatDenseGetArrayWrite(R, &temp_minimization_data_buffer));
                    PetscCall(PetscArraycpy(send_minimization_data_buffer, temp_minimization_data_buffer, R_local_values_count));
                    PetscCall(MatDenseRestoreArrayWrite(R, &temp_minimization_data_buffer));
                    PetscCallMPI(MPI_Isend(send_minimization_data_buffer, R_local_values_count, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 3, MPI_COMM_WORLD, &send_minimization_data_request));
                    // PetscCallMPI(MPI_Wait(&send_minimization_data_request, MPI_STATUS_IGNORE));
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////

        PetscCall(outer_solver_global_R(comm_jacobi_block, &outer_ksp, x_minimized, R, S, R_transpose_R, vec_R_transpose_b_block_jacobi, alpha, b, rank_jacobi_block, s));

        PetscCall(VecWAXPY(approximate_residual, -1.0, x_minimized_prev_iteration, x_minimized));

        PetscCall(VecNormBegin(approximate_residual, NORM_INFINITY, &approximation_residual_infinity_norm));
        PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_minimized, x_block_jacobi[idx_non_current_block], INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecNormEnd(approximate_residual, NORM_INFINITY, &approximation_residual_infinity_norm));

        PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_minimized, x_block_jacobi[rank_jacobi_block], INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_minimized, x_block_jacobi[rank_jacobi_block], INSERT_VALUES, SCATTER_REVERSE));

        PetscCall(printResidualNorm(approximation_residual_infinity_norm));

        if (PetscApproximateLTE(approximation_residual_infinity_norm, relative_tolerance)) // TODO: increase convergence count only on receiving data
        {
            convergence_count++;
        }
        else
        {
            convergence_count = ZERO;
        }

        // if (convergence_count >= CONVERGENCE_COUNT_MIN)
        // {
        if (proc_local_rank == ZERO)
        {
            if (convergence_count >= CONVERGENCE_COUNT_MIN)
                send_signal = CONVERGENCE_SIGNAL;
            else
                send_signal = NO_SIGNAL;
            PetscCallMPI(MPI_Test(&send_signal_request, &send_signal_flag, MPI_STATUS_IGNORE));
            if (send_signal_flag)
            {
                PetscCallMPI(MPI_Isend(&send_signal, ONE, MPIU_INT, message_destination, TAG_STATUS, MPI_COMM_WORLD, &send_signal_request));
            }
        }
        // }

        if (proc_local_rank == ZERO)
        {

            PetscCallMPI(MPI_Iprobe(message_source, TAG_STATUS, MPI_COMM_WORLD, &rcv_signal_flag, &status));
            if (rcv_signal_flag)
            {
                // printf("receiving ... rank block %d\n", rank_jacobi_block);
                PetscCallMPI(MPI_Irecv(&rcv_signal, ONE, MPIU_INT, message_source, TAG_STATUS, MPI_COMM_WORLD, &rcv_signal_request));
                PetscCallMPI(MPI_Wait(&rcv_signal_request, MPI_STATUS_IGNORE));
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

    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    end_time = MPI_Wtime();
    PetscCall(printElapsedTime(start_time, end_time));
    PetscCall(printTotalNumberOfIterations_2(number_of_iterations, s));

    
    PetscScalar *send_multisplitting_data_buffer_bis = NULL;
    PetscScalar *rcv_multisplitting_data_buffer_bis = NULL;

    if (rank_jacobi_block == BLOCK_RANK_ZERO)
    {
        PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &send_multisplitting_data_buffer_bis));
        PetscCallMPI(MPI_Send(send_multisplitting_data_buffer_bis, vec_local_size, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 10, MPI_COMM_WORLD));
        PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &send_multisplitting_data_buffer_bis));
        // printf("hello world ... rank block %d proc %d \n", rank_jacobi_block, proc_local_rank);

        PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &rcv_multisplitting_data_buffer_bis));
        PetscCallMPI(MPI_Recv(rcv_multisplitting_data_buffer_bis, vec_local_size, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &rcv_multisplitting_data_buffer_bis));
    }
    else if (rank_jacobi_block == BLOCK_RANK_ONE)
    {
        PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &rcv_multisplitting_data_buffer_bis));
        PetscCallMPI(MPI_Recv(rcv_multisplitting_data_buffer_bis, vec_local_size, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &rcv_multisplitting_data_buffer_bis));
        // printf("============> hello world ... rank block %d proc %d\n", rank_jacobi_block, proc_local_rank);

        PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &send_multisplitting_data_buffer_bis));
        PetscCallMPI(MPI_Send(send_multisplitting_data_buffer_bis, vec_local_size, MPIU_SCALAR, (idx_non_current_block * nprocs_per_jacobi_block) + proc_local_rank, 11, MPI_COMM_WORLD));
        PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &send_multisplitting_data_buffer_bis));
    }

    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[rank_jacobi_block], x_block_jacobi[rank_jacobi_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterBegin(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter_jacobi_vec_part_to_merged_vec[idx_non_current_block], x_block_jacobi[idx_non_current_block], x, INSERT_VALUES, SCATTER_FORWARD));

    PetscScalar direct_residual_norm;
    PetscCall(computeFinalResidualNorm(A_block_jacobi, &x, b_block_jacobi, rank_jacobi_block, proc_global_rank, &direct_residual_norm));

    PetscCall(printFinalResidualNorm(direct_residual_norm));

    for (PetscInt i = 0; i < njacobi_blocks; i++)
    {
        PetscCall(ISDestroy(&is_cols_block_jacobi[i]));
        PetscCall(VecDestroy(&x_block_jacobi[i]));
        PetscCall(VecDestroy(&b_block_jacobi[i]));
        PetscCall(MatDestroy(&A_block_jacobi_subMat[i]));
        PetscCall(VecScatterDestroy(&scatter_jacobi_vec_part_to_merged_vec[i]));
        PetscCall(ISDestroy(&is_merged_vec[i]));
    }

    
    PetscFree(vec_local_idx);
    PetscFree(vector_to_insert_into_S);
    PetscCall(VecDestroy(&x_minimized_prev_iteration));
    PetscCall(VecDestroy(&approximate_residual));
    PetscCall(ISDestroy(&is_jacobi_vec_parts));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&x_minimized));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x_initial_guess));
    PetscCall(MatDestroy(&A_block_jacobi));
    PetscCall(MatDestroy(&A_block_jacobi_resdistributed));
    PetscCall(MatDestroy(&S));
    PetscCall(MatDestroy(&R));

    PetscCall(PetscFree(send_multisplitting_data_buffer));
    PetscCall(PetscFree(rcv_multisplitting_data_buffer));
    PetscCall(KSPDestroy(&inner_ksp));
    PetscCall(KSPDestroy(&outer_ksp));
    PetscCall(MatDestroy(&R_transpose_R));
    PetscCall(VecDestroy(&vec_R_transpose_b_block_jacobi));
    PetscCall(VecDestroy(&alpha));

    // Maybe delete the rest of this code, not necessary
    PetscInt message = ZERO;
    PetscInt count;

    do
    {
        MPI_Datatype data_type = MPIU_INT;
        PetscCallMPI(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &message, &status));
        if (message)
        {
            if (status.MPI_TAG == TAG_DATA)
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

    PetscCallMPI(MPI_Wait(&send_minimization_data_request, MPI_STATUS_IGNORE));

    PetscCallMPI(MPI_Wait(&send_multisplitting_data_request, MPI_STATUS_IGNORE));

    PetscCallMPI(MPI_Wait(&send_signal_request, MPI_STATUS_IGNORE));

    PetscCall(PetscCommDestroy(&comm_jacobi_block));
    PetscCall(PetscFinalize());
    return 0;
}

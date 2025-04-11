#include "constants.h"
#include "utils.h"
#include "comm.h"

PetscErrorCode comm_async_probe_and_receive(Vec *x_block_jacobi, PetscScalar *rcv_buffer, PetscMPIInt vec_local_size, PetscMPIInt rcv_data_flag, PetscMPIInt message_source, PetscMPIInt idx_non_current_block, PetscInt *message_received)
{
    PetscFunctionBeginUser;

    PetscMPIInt loop_counter = 0;
    PetscMPIInt rank_jacobi_block __attribute__((unused));
    if (idx_non_current_block == 1)
        rank_jacobi_block = 0;
    if (idx_non_current_block == 0)
        rank_jacobi_block = 1;

    PetscCallMPI(MPI_Iprobe(message_source, (TAG_MULTISPLITTING_DATA + idx_non_current_block), MPI_COMM_WORLD, &rcv_data_flag, MPI_STATUS_IGNORE));
    if (rcv_data_flag)
    {
        PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &rcv_buffer));
        do
        {
            printf("=============Block rank %d START multipsplitting RCV communication\n", rank_jacobi_block);
            PetscCallMPI(MPI_Recv(rcv_buffer, vec_local_size, MPIU_SCALAR, message_source, (TAG_MULTISPLITTING_DATA + idx_non_current_block), MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            printf("=============Block rank %d END multipsplitting RCV communication\n", rank_jacobi_block);
            loop_counter++;
            if (loop_counter >= 3) // TODO: remember this, possibly make it an argument of the program
            {
                break;
            }
            PetscCallMPI(MPI_Iprobe(message_source, (TAG_MULTISPLITTING_DATA + idx_non_current_block), MPI_COMM_WORLD, &rcv_data_flag, MPI_STATUS_IGNORE));
        } while (rcv_data_flag);
        PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &rcv_buffer));

        if (message_received != NULL)
            (*message_received) = 1;
    }
    printf("=============Block rank %d END RCV communication function\n", rank_jacobi_block);

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_async_test_and_send(Vec *x_block_jacobi, PetscScalar *send_buffer, PetscScalar *temp_buffer, MPI_Request *send_data_request, PetscMPIInt vec_local_size, PetscMPIInt send_data_flag, PetscMPIInt message_dest, PetscMPIInt rank_jacobi_block)
{
    PetscFunctionBeginUser;

    MPI_Test(send_data_request, &send_data_flag, MPI_STATUS_IGNORE);
    if (send_data_flag)
    {
        PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
        PetscCall(PetscArraycpy(send_buffer, temp_buffer, vec_local_size));
        PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &temp_buffer));
        PetscCallMPI(MPI_Isend(send_buffer, vec_local_size, MPIU_SCALAR, message_dest, TAG_MULTISPLITTING_DATA + rank_jacobi_block, MPI_COMM_WORLD, send_data_request));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_async_convergence_detection(PetscMPIInt *broadcast_message, PetscInt convergence_count, PetscInt MIN_CONVERGENCE_COUNT, PetscMPIInt *send_signal, MPI_Request *send_signal_request, PetscMPIInt *rcv_signal, PetscMPIInt message_dest, PetscMPIInt message_source, PetscMPIInt rank_jacobi_block, PetscMPIInt idx_non_current_block, PetscMPIInt proc_local_rank)
{
    PetscFunctionBeginUser;
    PetscMPIInt send_signal_flag;
    PetscMPIInt rcv_signal_flag;

    if (proc_local_rank == ZERO)
    {
        if (convergence_count >= MIN_CONVERGENCE_COUNT)
            (*send_signal) = CONVERGENCE_SIGNAL;
        else
            (*send_signal) = NO_SIGNAL;

        PetscCallMPI(MPI_Test(send_signal_request, &send_signal_flag, MPI_STATUS_IGNORE));
        if (send_signal_flag)
        {
            PetscCallMPI(MPI_Isend(send_signal, ONE, MPIU_INT, message_dest, TAG_STATUS + rank_jacobi_block, MPI_COMM_WORLD, send_signal_request));
        }
    }

    if (proc_local_rank == ZERO)
    {
        PetscCallMPI(MPI_Iprobe(message_source, TAG_STATUS + idx_non_current_block, MPI_COMM_WORLD, &rcv_signal_flag, MPI_STATUS_IGNORE));
        if (rcv_signal_flag)
        {
            do
            {

                // printf("=============Block rank %d START multipsplitting RCV communication - SIGNAL\n", idx_non_current_block);
                PetscCallMPI(MPI_Recv(rcv_signal, ONE, MPIU_INT, message_source, TAG_STATUS + idx_non_current_block, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                PetscCallMPI(MPI_Iprobe(message_source, TAG_STATUS + idx_non_current_block, MPI_COMM_WORLD, &rcv_signal_flag, MPI_STATUS_IGNORE));
                // printf("=============Block rank %d END multipsplitting RCV communication - SIGNAL\n", idx_non_current_block);
            } while (rcv_signal_flag);
        }
    }

    if (proc_local_rank == ZERO)
    {
        if ((*send_signal) == CONVERGENCE_SIGNAL && (*rcv_signal) == CONVERGENCE_SIGNAL)
        {
            (*broadcast_message) = TERMINATE_SIGNAL;
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_sync_send_and_receive(Vec *x_block_jacobi, PetscMPIInt vec_local_size, PetscMPIInt message_dest, PetscMPIInt message_source, PetscMPIInt rank_jacobi_block, PetscMPIInt idx_non_current_block)
{
    PetscFunctionBeginUser;
    PetscScalar *send_multisplitting_data_buffer = NULL;
    PetscScalar *rcv_multisplitting_data_buffer = NULL;

    PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &send_multisplitting_data_buffer));
    PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &rcv_multisplitting_data_buffer));

    PetscCallMPI(MPI_Sendrecv(send_multisplitting_data_buffer, vec_local_size, MPIU_SCALAR, message_dest, (TAG_MULTISPLITTING_DATA + rank_jacobi_block), rcv_multisplitting_data_buffer, vec_local_size, MPIU_SCALAR, message_source, (TAG_MULTISPLITTING_DATA + idx_non_current_block), MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &send_multisplitting_data_buffer));
    PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &rcv_multisplitting_data_buffer));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_sync_send_and_receive_final(Vec *x_block_jacobi, PetscMPIInt vec_local_size, PetscMPIInt message_dest, PetscMPIInt message_source, PetscMPIInt rank_jacobi_block, PetscMPIInt idx_non_current_block)
{
    PetscFunctionBeginUser;
    PetscScalar *send_multisplitting_data_buffer = NULL;
    PetscScalar *rcv_multisplitting_data_buffer = NULL;

    PetscCall(VecGetArray(x_block_jacobi[rank_jacobi_block], &send_multisplitting_data_buffer));
    PetscCall(VecGetArray(x_block_jacobi[idx_non_current_block], &rcv_multisplitting_data_buffer));

    PetscCallMPI(MPI_Sendrecv(send_multisplitting_data_buffer, vec_local_size, MPIU_SCALAR, message_dest, (TAG_FINAL_DATA_EXCHANGE + rank_jacobi_block), rcv_multisplitting_data_buffer, vec_local_size, MPIU_SCALAR, message_source, (TAG_FINAL_DATA_EXCHANGE + idx_non_current_block), MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    PetscCall(VecRestoreArray(x_block_jacobi[rank_jacobi_block], &send_multisplitting_data_buffer));
    PetscCall(VecRestoreArray(x_block_jacobi[idx_non_current_block], &rcv_multisplitting_data_buffer));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_sync_convergence_detection(PetscMPIInt *broadcast_message, PetscMPIInt send_signal, PetscMPIInt rcv_signal, PetscMPIInt message_dest, PetscMPIInt message_source, PetscMPIInt rank_jacobi_block, PetscMPIInt idx_non_current_block, PetscMPIInt proc_local_rank)
{
    PetscFunctionBeginUser;

    if (proc_local_rank == ZERO)
    {
        PetscCallMPI(MPI_Sendrecv(&send_signal, ONE, MPIU_INT, message_dest, (TAG_STATUS + rank_jacobi_block), &rcv_signal, ONE, MPIU_INT, message_source, (TAG_STATUS + idx_non_current_block), MPI_COMM_WORLD, MPI_STATUS_IGNORE));

        if (send_signal == CONVERGENCE_SIGNAL && rcv_signal == CONVERGENCE_SIGNAL)
        {
            (*broadcast_message) = TERMINATE_SIGNAL;
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_sync_send_and_receive_minimization(Mat R, PetscScalar *send_minimization_data_buffer, PetscScalar *rcv_minimization_data_buffer, PetscMPIInt R_local_values_count, PetscMPIInt message_dest, PetscMPIInt message_source, PetscMPIInt rank_jacobi_block, PetscMPIInt idx_non_current_block, PetscInt n_mesh_points, PetscInt rstart, PetscInt rend, PetscInt lda, PetscInt s)
{
    PetscFunctionBeginUser;
    PetscScalar *temp_minimization_data_buffer;

    PetscCall(MatDenseGetArray(R, &temp_minimization_data_buffer));
    PetscCall(PetscArraycpy(send_minimization_data_buffer, temp_minimization_data_buffer, R_local_values_count));
    PetscCall(MatDenseRestoreArray(R, &temp_minimization_data_buffer));

    PetscCallMPI(MPI_Sendrecv(send_minimization_data_buffer, R_local_values_count, MPIU_SCALAR, message_dest, (TAG_MINIMIZATION_DATA + rank_jacobi_block), rcv_minimization_data_buffer, R_local_values_count, MPIU_SCALAR, message_source, (TAG_MINIMIZATION_DATA + idx_non_current_block), MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    PetscCall(MatDenseGetArray(R, &temp_minimization_data_buffer));
    if (rstart < (n_mesh_points / 2) && (n_mesh_points / 2) < rend)
    {
        for (PetscInt j = 0; j < s; j++)
        {
            PetscInt idx = (idx_non_current_block * (lda / 2)) + (j * lda);
            PetscCall(PetscArraycpy(&temp_minimization_data_buffer[idx], &rcv_minimization_data_buffer[idx], lda / 2));
        }
    }

    if (rstart >= (n_mesh_points / 2) && rank_jacobi_block == BLOCK_RANK_ZERO)
    {
        PetscCall(PetscArraycpy(temp_minimization_data_buffer, rcv_minimization_data_buffer, R_local_values_count));
    }

    if (rend <= (n_mesh_points / 2) && rank_jacobi_block == BLOCK_RANK_ONE)
    {
        PetscCall(PetscArraycpy(temp_minimization_data_buffer, rcv_minimization_data_buffer, R_local_values_count));
    }

    PetscCall(MatDenseRestoreArray(R, &temp_minimization_data_buffer));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_async_probe_and_receive_min(Mat R, PetscScalar *rcv_minimization_data_buffer, PetscScalar *temp_minimization_data_buffer, PetscMPIInt R_local_values_count, PetscMPIInt rcv_minimization_data_flag, PetscMPIInt message_source, PetscMPIInt rank_jacobi_block, PetscMPIInt idx_non_current_block, PetscInt n_mesh_points, PetscInt rstart, PetscInt rend, PetscInt lda, PetscInt s)
{
    PetscFunctionBeginUser;
    PetscMPIInt loop_counter = 0;

    PetscCallMPI(MPI_Iprobe(message_source, (TAG_MINIMIZATION_DATA + idx_non_current_block), MPI_COMM_WORLD, &rcv_minimization_data_flag, MPI_STATUS_IGNORE));
    if (rcv_minimization_data_flag)
    {

        do
        {
            printf("=============Block rank %d START minimization RCV communication\n", rank_jacobi_block);
            PetscCallMPI(MPI_Recv(rcv_minimization_data_buffer, R_local_values_count, MPIU_SCALAR, message_source, (TAG_MINIMIZATION_DATA + idx_non_current_block), MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            printf("=============Block rank %d END minimization RCV communication\n", rank_jacobi_block);
            loop_counter++;
            if (loop_counter >= 3) // TODO: remember this, possibly make it an argument of the program
            {
                break;
            }

            PetscCallMPI(MPI_Iprobe(message_source, (TAG_MINIMIZATION_DATA + idx_non_current_block), MPI_COMM_WORLD, &rcv_minimization_data_flag, MPI_STATUS_IGNORE));
        } while (rcv_minimization_data_flag);

        PetscCall(MatDenseGetArray(R, &temp_minimization_data_buffer));
        if (rstart < (n_mesh_points / 2) && (n_mesh_points / 2) < rend)
        {
            for (PetscInt j = 0; j < s; j++)
            {
                PetscInt idx = (idx_non_current_block * (lda / 2)) + (j * lda);
                PetscCall(PetscArraycpy(&temp_minimization_data_buffer[idx], &rcv_minimization_data_buffer[idx], lda / 2));
            }
        }

        if (rstart >= (n_mesh_points / 2) && rank_jacobi_block == BLOCK_RANK_ZERO)
        {
            PetscCall(PetscArraycpy(temp_minimization_data_buffer, rcv_minimization_data_buffer, R_local_values_count));
        }

        if (rend <= (n_mesh_points / 2) && rank_jacobi_block == BLOCK_RANK_ONE)
        {
            PetscCall(PetscArraycpy(temp_minimization_data_buffer, rcv_minimization_data_buffer, R_local_values_count));
        }
        PetscCall(MatDenseRestoreArray(R, &temp_minimization_data_buffer));
    }

    printf("=============Block rank %d END RCV minimization communication function\n", rank_jacobi_block);

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode comm_async_test_and_send_min(Mat R, PetscScalar *send_minimization_data_buffer, PetscScalar *temp_minimization_data_buffer, MPI_Request send_minimization_data_request, PetscMPIInt R_local_values_count, PetscMPIInt message_dest, PetscMPIInt rank_jacobi_block)
{
    PetscFunctionBeginUser;
    PetscMPIInt send_minimization_data_flag;
    PetscCallMPI(MPI_Test(&send_minimization_data_request, &send_minimization_data_flag, MPI_STATUS_IGNORE));
    if (send_minimization_data_flag)
    {
        PetscCall(MatDenseGetArray(R, &temp_minimization_data_buffer));
        PetscCall(PetscArraycpy(send_minimization_data_buffer, temp_minimization_data_buffer, R_local_values_count));

        PetscCall(MatDenseRestoreArray(R, &temp_minimization_data_buffer));
        PetscCallMPI(MPI_Isend(send_minimization_data_buffer, R_local_values_count, MPIU_SCALAR, message_dest, (TAG_MINIMIZATION_DATA + rank_jacobi_block), MPI_COMM_WORLD, &send_minimization_data_request));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}
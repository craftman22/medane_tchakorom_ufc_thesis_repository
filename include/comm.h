
#ifndef SHARED_COMM_FUNCTIONS_H
#define SHARED_COMM_FUNCTIONS_H


//PetscErrorCode comm_async_probe_and_receive(Vec *x_block_jacobi, PetscScalar *rcv_buffer, PetscMPIInt vec_local_size, PetscMPIInt rcv_data_flag, PetscMPIInt message_source, PetscMPIInt idx_non_current_block);
// PetscErrorCode comm_async_probe_and_receive(Vec *x_block_jacobi, PetscScalar *rcv_buffer, PetscMPIInt vec_local_size, PetscMPIInt rcv_data_flag, PetscMPIInt message_source, PetscMPIInt idx_non_current_block, PetscInt *message_received);

// PetscErrorCode comm_async_test_and_send(Vec *x_block_jacobi, PetscScalar *send_buffer, PetscScalar *temp_buffer, MPI_Request *send_data_request, PetscMPIInt vec_local_size, PetscMPIInt send_data_flag, PetscMPIInt message_dest, PetscMPIInt rank_jacobi_block);

PetscErrorCode comm_async_probe_and_receive(Vec *x_block_jacobi, PetscScalar *rcv_buffer, PetscMPIInt vec_local_size, PetscMPIInt rcv_data_flag, PetscMPIInt message_source, PetscMPIInt idx_non_current_block, PetscInt *message_received, PetscMPIInt *other_block_current_iteration, char **pack_buffer);

PetscErrorCode comm_async_test_and_send(Vec *x_block_jacobi, PetscScalar *send_buffer, PetscScalar *temp_buffer, MPI_Request *send_data_request, PetscMPIInt vec_local_size, PetscMPIInt send_data_flag, PetscMPIInt message_dest, PetscMPIInt rank_jacobi_block, PetscMPIInt * current_number_of_iterations, char **pack_buffer);

PetscErrorCode comm_async_convergence_detection(PetscMPIInt *broadcast_message, PetscInt convergence_count, PetscInt MIN_CONVERGENCE_COUNT, PetscMPIInt *send_signal, MPI_Request *send_signal_request, PetscMPIInt *rcv_signal, PetscMPIInt message_dest, PetscMPIInt message_source, PetscMPIInt rank_jacobi_block, PetscMPIInt idx_non_current_block, PetscMPIInt proc_local_rank);

PetscErrorCode comm_async_probe_and_receive_min(Mat R, PetscScalar *rcv_minimization_data_buffer, PetscScalar *temp_minimization_data_buffer, PetscMPIInt R_local_values_count, PetscMPIInt rcv_minimization_data_flag, PetscMPIInt message_source, PetscMPIInt rank_jacobi_block, PetscMPIInt idx_non_current_block, PetscInt n_mesh_points, PetscInt rstart, PetscInt rend,PetscInt lda, PetscInt s);

PetscErrorCode comm_async_test_and_send_min(Mat R, PetscScalar *send_minimization_data_buffer, PetscScalar *temp_minimization_data_buffer, MPI_Request send_minimization_data_request, PetscMPIInt R_local_values_count, PetscMPIInt message_dest, PetscMPIInt rank_jacobi_block);

PetscErrorCode comm_sync_send_and_receive(Vec *x_block_jacobi, PetscMPIInt vec_local_size, PetscMPIInt message_dest, PetscMPIInt message_source, PetscMPIInt rank_jacobi_block ,PetscMPIInt idx_non_current_block);

PetscErrorCode comm_sync_convergence_detection(PetscMPIInt *broadcast_message, PetscMPIInt send_signal, PetscMPIInt rcv_signal, PetscMPIInt message_dest, PetscMPIInt message_source, PetscMPIInt rank_jacobi_block, PetscMPIInt idx_non_current_block, PetscMPIInt proc_local_rank);

PetscErrorCode comm_sync_send_and_receive_minimization(Mat R,PetscScalar *send_minimization_data_buffer, PetscScalar *rcv_minimization_data_buffer, PetscMPIInt R_local_values_count,PetscMPIInt message_dest ,PetscMPIInt message_source, PetscMPIInt rank_jacobi_block, PetscMPIInt idx_non_current_block, PetscInt n_mesh_points, PetscInt rstart, PetscInt rend, PetscInt lda, PetscInt s);

PetscErrorCode comm_sync_send_and_receive_final(Vec *x_block_jacobi, PetscMPIInt vec_local_size, PetscMPIInt message_dest, PetscMPIInt message_source, PetscMPIInt rank_jacobi_block, PetscMPIInt idx_non_current_block);

PetscErrorCode comm_sync_measure_latency_between_two_nodes(PetscMPIInt proc_rank_node_1, PetscMPIInt proc_rank_node_2, PetscMPIInt actual_rank);    

PetscErrorCode mpi_pack_multisplitting_data(PetscScalar *send_buffer, PetscMPIInt data_size, PetscInt *version, char **pack_buffer, PetscMPIInt *position);

PetscErrorCode mpi_unpack_multisplitting_data(PetscScalar *rcv_buffer, PetscMPIInt data_size, PetscInt *version, char **pack_buffer, PetscMPIInt pack_size);

#endif // SHARED_COMM_FUNCTIONS_H
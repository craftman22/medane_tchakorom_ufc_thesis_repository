#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"
#include "utils.h"
#include "conv_detection_prime.h"

PetscErrorCode _pack_computed_data_dependency(PetscInt *PhaseTag_param, PetscInt *NumIteration_param, const PetscInt data_count, PetscScalar *data, char **pack_buffer, PetscMPIInt *position)
{
    PetscFunctionBeginUser;
    PetscMPIInt size_1 = 0;
    PetscMPIInt size_2 = 0;
    PetscMPIInt size_3 = 0;
    (*position) = 0;

    PetscCallMPI(MPI_Pack_size(1, MPIU_INT, MPI_COMM_WORLD, &size_1));
    PetscCallMPI(MPI_Pack_size(1, MPIU_INT, MPI_COMM_WORLD, &size_2));
    PetscCallMPI(MPI_Pack_size(data_count, MPIU_SCALAR, MPI_COMM_WORLD, &size_3));

    PetscMPIInt total_pack_size;
    total_pack_size = size_1 + size_2 + size_3;
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "size: %d\n", total_pack_size));

    if ((*pack_buffer) == NULL)
    {
        PetscCall(PetscPrintf(MPI_COMM_SELF, " malloc done! \n"));
        PetscCall(PetscMalloc1(total_pack_size, &(*pack_buffer)));
    }

    // PetscCallMPI(MPI_Pack(PhaseTag_param, 1, MPIU_INT, (pack_buffer), total_pack_size, position, MPI_COMM_WORLD));
    // PetscCallMPI(MPI_Pack(NumIteration_param, 1, MPIU_INT, (pack_buffer), total_pack_size, position, MPI_COMM_WORLD));
    // PetscCallMPI(MPI_Pack(data, data_count, MPIU_SCALAR, (pack_buffer), total_pack_size, position, MPI_COMM_WORLD));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode _unpack_computed_data_dependency(PetscInt *PhaseTag_param, PetscInt *NumIteration_param, const PetscInt data_count, PetscScalar *data, char **pack_buffer, PetscMPIInt pack_size)
{

    PetscFunctionBeginUser;

    PetscMPIInt position = 0;
    PetscCallMPI(MPI_Unpack((*pack_buffer), pack_size, &position, PhaseTag_param, 1, MPIU_INT, MPI_COMM_WORLD));
    PetscCallMPI(MPI_Unpack((*pack_buffer), pack_size, &position, NumIteration_param, 1, MPIU_INT, MPI_COMM_WORLD));
    PetscCallMPI(MPI_Unpack((*pack_buffer), pack_size, &position, data, data_count, MPIU_SCALAR, MPI_COMM_WORLD));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// #ifdef VERSION_1_0

PetscErrorCode foo(PetscInt *PhaseTag_param, PetscInt *NumIteration_param, const PetscInt data_count, PetscScalar *data, char **pack_buffer, PetscMPIInt *position)
{
    PetscFunctionBeginUser;
    PetscCall(pack_computed_data_dependency(PhaseTag_param, NumIteration_param, data_count, data, pack_buffer, position));
    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{

    PetscMPIInt nprocs;
    PetscMPIInt proc_global_rank;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

    Vec vector;
    PetscCall(create_vector(MPI_COMM_WORLD, &vector, 5, VECMPI));
    PetscRandom random;
    PetscCall(PetscRandomCreate(MPI_COMM_WORLD, &random));
    PetscCall(PetscRandomSetType(random, PETSCRAND48));

    PetscCall(VecSetRandom(vector, random));
    PetscCall(VecAssemblyBegin(vector));
    PetscCall(VecAssemblyEnd(vector));

    PetscCall(VecView(vector, PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(PetscRandomDestroy(&random));
    PetscCall(VecDestroy(&vector));

    // PetscMPIInt first_place_size;
    // PetscCallMPI(MPI_Pack_size(1, MPIU_INT, MPI_COMM_WORLD, &first_place_size));
    // PetscCall(PetscPrintf(PETSC_COMM_SELF, "size: %d\n", first_place_size));

    // const PetscInt arr_size = 6;
    // MPI_Request request = MPI_REQUEST_NULL;
    // if (proc_global_rank == 0)
    // {
    //     PetscInt a = 666;
    //     PetscInt b = 111;
    //     PetscScalar *arr = NULL;
    //     PetscCall(PetscMalloc1(arr_size, &arr));

    //     arr[0] = 23;
    //     arr[1] = -4;
    //     arr[2] = 1.99;
    //     arr[3] = 344;
    //     arr[4] = 99;
    //     arr[5] = 110;

    //     char *pack_buffer = NULL;
    //     PetscMPIInt position = 0;

    //     PetscCall(foo(&a, &b, arr_size, arr, &pack_buffer, &position));
    //     if (pack_buffer == NULL)
    //     {
    //         PetscCall(PetscPrintf(MPI_COMM_SELF, "still NULL \n"));
    //     }
    //     else
    //     {
    //         PetscCall(PetscPrintf(MPI_COMM_SELF, "not NULL \n"));
    //     }

    //     PetscCallMPI(MPI_Isend(pack_buffer, position, MPI_PACKED, 1, 0, MPI_COMM_WORLD, &request));
    //     PetscCallMPI(MPI_Wait(&request, MPI_STATUS_IGNORE));
    // }

    // if (proc_global_rank == 1)
    // {
    //     PetscInt a = -1;
    //     PetscInt b = -1;
    //     PetscScalar *arr = NULL;
    //     PetscCall(PetscMalloc1(arr_size, &arr));

    //     char *pack_buffer = NULL;
    //     PetscCall(PetscMalloc1(56, &pack_buffer));

    //     PetscCall(PetscPrintf(PETSC_COMM_SELF, "AVANT: a = %d && b = %d\n", a, b));
    //     PetscCallMPI(MPI_Recv(pack_buffer, 56, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    //     PetscCall(unpack_computed_data_dependency(&a, &b, arr_size, arr, &pack_buffer, 56));
    //     PetscCall(PetscPrintf(PETSC_COMM_SELF, "APRES: a = %d && b = %d\n", a, b));
    //     for (PetscInt idx = 0; idx < arr_size; idx++)
    //     {
    //         PetscCall(PetscPrintf(PETSC_COMM_SELF, "Valeur %d = %e \n", idx, arr[idx]));
    //     }
    // }

    PetscCall(PetscFinalize());
    return 0;
}

//     char hostname[PETSC_MAX_PATH_LEN];
//     size_t len;
//  PetscCall(PetscGetHostName(hostname, sizeof(hostname)));
//     PetscCall(PetscStrlen(hostname, &len));

//     if (proc_global_rank <= 1)
//     {
//         if (proc_global_rank == 1)
//             PetscCall(PetscSleep(2));

//         extern char **environ; // POSIX global for env vars
//         char **env = environ;
//         PetscPrintf(PETSC_COMM_SELF, "=== Environment variables visible to rank %d %s ===\n", proc_global_rank,hostname);
//         while (*env)
//         {
//             PetscPrintf(PETSC_COMM_SELF, "%s\n", *env);
//             env++;
//         }
//     }
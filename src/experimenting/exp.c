#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"
#include "constants.h"
#include "utils.h"
#include "comm.h"
#include "conv_detection_prime.h"

// #ifdef VERSION_1_0

int main(int argc, char **argv)
{

    PetscMPIInt nprocs;
    PetscMPIInt proc_global_rank;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

    // PetscMPIInt first_place_size;
    // PetscCallMPI(MPI_Pack_size(1, MPIU_INT, MPI_COMM_WORLD, &first_place_size));
    // PetscCall(PetscPrintf(PETSC_COMM_SELF, "size: %d\n", first_place_size));

    MPI_Request request = MPI_REQUEST_NULL;
    if (proc_global_rank == 0)
    {
        PetscInt a = 666;
        PetscInt b = 111;
        char *pack_buffer = NULL;
        PetscMPIInt position = 0;
        PetscCall(pack_convergence_data(&a, &b, &pack_buffer, &position));
        PetscCallMPI(MPI_Isend(pack_buffer, position, MPI_PACKED, 1, 0, MPI_COMM_WORLD, &request));
        PetscCallMPI(MPI_Wait(&request, MPI_STATUS_IGNORE));
    }

    if (proc_global_rank == 1)
    {
        PetscInt a = -1;
        PetscInt b = -1;

        char *pack_buffer = NULL;
        PetscCall(PetscMalloc1(8, &pack_buffer));

        PetscCall(PetscPrintf(PETSC_COMM_SELF, "AVANT: a = %d && b = %d\n", a, b));
        PetscCallMPI(MPI_Recv(pack_buffer, 8, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        PetscCall(unpack_convergence_data(&a, &b, &pack_buffer, 8));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "APRES: a = %d && b = %d\n", a, b));
    }

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
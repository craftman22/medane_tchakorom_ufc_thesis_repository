#include <petscts.h>
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include "petscdmda.h"
#include "constants.h"
#include "utils.h"
#include "comm.h"

// #ifdef VERSION_1_0

int main(int argc, char **argv)
{

    PetscMPIInt nprocs;
    PetscMPIInt proc_global_rank;

    char hostname[PETSC_MAX_PATH_LEN];
    size_t len;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

    PetscCall(PetscGetHostName(hostname, sizeof(hostname)));
    PetscCall(PetscStrlen(hostname, &len));

    if (proc_global_rank <= 1)
    {
        if (proc_global_rank == 1)
            PetscCall(PetscSleep(2));

        extern char **environ; // POSIX global for env vars
        char **env = environ;
        PetscPrintf(PETSC_COMM_SELF, "=== Environment variables visible to rank %d %s ===\n", proc_global_rank,hostname);
        while (*env)
        {
            PetscPrintf(PETSC_COMM_SELF, "%s\n", *env);
            env++;
        }
    }
    PetscFinalize();
    return 0;
}
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

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &proc_global_rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nprocs));

    if (proc_global_rank)
    {
        extern char **environ; // POSIX global for env vars
        char **env = environ;
        PetscPrintf(PETSC_COMM_SELF, "=== Environment variables visible to rank %d ===\n", proc_global_rank);
        while (*env)
        {
            PetscPrintf(PETSC_COMM_SELF, "%s\n", *env);
            env++;
        }
    }
    PetscFinalize();
    return 0;
}
#ifndef CONSTANTS_H
#define CONSTANTS_H


#define ZERO 0
#define ONE 1
#define MINUS_ONE -1

#define INNER_KSP_PREFIX "inner_"
#define INNER_PC_PREFIX "inner_"

#define OUTER_KSP_PREFIX "outer_"
#define OUTER_PC_PREFIX "outer_"

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
#define TAG_DATA_MINIMIZATION 5
#define TAG_DATA_OUT_LOOP 6

#define BLOCK_RANK_ZERO 0
#define BLOCK_RANK_ONE 1

// #define CONVERGENCE_COUNT_MIN 4


#endif // CONSTANTS_H
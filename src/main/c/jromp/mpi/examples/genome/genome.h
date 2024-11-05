#ifndef GENOME_H
#define GENOME_H

#include <dirent.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "cvector.h"

#define MAX_PATH_SIZE 1024
#define string char *

#define DEBUG_LOGGING 1

#if DEBUG_LOGGING == 1
    /**
     * Prints the given message if the current rank is 0 (Master process).
     */
    #define LOG_MASTER(...)                                                                                            \
        if (rank == 0) {                                                                                               \
            printf("       Master: " __VA_ARGS__);                                                                     \
        }
    /**
     * Prints the given message if the current rank is not 0 (Worker process).
     * In addition, if the context is parallel, the thread number is also printed synchronously.
     */
    #define LOG_WORKER(...)                                                                                            \
        if (rank != 0) {                                                                                               \
            if (omp_get_num_threads() == 1) {                                                                          \
                printf("   Worker %03d: ", rank);                                                                      \
                printf(__VA_ARGS__);                                                                                   \
            } else {                                                                                                   \
                omp_set_lock(&print_lock);                                                                             \
                printf("Worker %03d-%02d: ", rank, omp_get_thread_num());                                              \
                printf(__VA_ARGS__);                                                                                   \
                omp_unset_lock(&print_lock);                                                                           \
            }                                                                                                          \
        }
#else
    #define LOG_MASTER(...)
    #define LOG_WORKER(...)
#endif

#endif // GENOME_H

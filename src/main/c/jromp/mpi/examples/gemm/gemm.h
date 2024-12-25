#ifndef GEMM_H
#define GEMM_H

#include <assert.h>
#include <mpi.h>
#include <omp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define assert_non_null(ptr) assert(ptr != NULL)
#define UNUSED __attribute__((unused))

#define START_BLOCK {
#define END_BLOCK }

#define START_MPI_TIMER(name) \
    double name##_mpi_start = MPI_Wtime(); \
    START_BLOCK

#define STOP_MPI_TIMER(name) \
    END_BLOCK \
    double name##_mpi_end = MPI_Wtime(); \
    double name##_mpi_elapsed = name##_mpi_end - name##_mpi_start;

#define GET_MPI_TIMER(name) name##_mpi_elapsed

#define START_OMP_TIMER(name) \
    double name##_omp_start = omp_get_wtime(); \
    START_BLOCK

#define STOP_OMP_TIMER(name) \
    END_BLOCK \
    double name##_omp_end = omp_get_wtime(); \
    double name##_omp_elapsed = name##_omp_end - name##_omp_start;

#define GET_OMP_TIMER(name) name##_omp_elapsed

#define MASTER
#define WORKER
#define MASTER_RANK 0
#define DATA_TAG 0
#define FINISH_TAG 1

/**
 * Prints the given message if the current rank is 0 (Master process).
 */
#define LOG_MASTER(...)                        \
    if (rank == 0) {                           \
        printf("      Master: " __VA_ARGS__);  \
    }

/**
 * Prints the given message if the current rank is not 0 (Worker process).
 * In addition, if the context is parallel, the thread number is also printed synchronously.
 */
#define LOG_WORKER(...)                                               \
    if (rank != 0) {                                                  \
        if (omp_get_num_threads() == 1) {                             \
            printf("   Worker %02d: ", rank);                         \
            printf(__VA_ARGS__);                                      \
        } else {                                                      \
            omp_set_lock(&print_lock);                                \
            printf("Worker %02d-%02d: ", rank, omp_get_thread_num()); \
            printf(__VA_ARGS__);                                      \
            omp_unset_lock(&print_lock);                              \
        }                                                             \
    }

/**
 * Performs the matrix multiplication C = A * B (GEMM).
 *
 * @param a Matrix A.
 * @param b Matrix B.
 * @param c Matrix C.
 * @param rows_per_worker The number of rows to be processed by each worker.
 */
void gemm(const double *a, const double *b, double *c, int rows_per_worker);

/**
 * Initializes the given matrix with random values.
 *
 * @param a The first matrix.
 * @param b The second matrix.
 * @param n The size of the matrices.
 */
void matrix_initialization(double *a, double *b, int n);

/**
 * Generates a random integer in the range [min, max].
 *
 * @param min The minimum value.
 * @param max The maximum value.
 *
 * @return The random integer.
 */
static int random_in_range(const int min, const int max) {
    return min + random() % (max - min + 1);
}

/**
 * Writes the execution configuration to the file.
 *
 * @param n The size of the matrices.
 * @param workers The number of workers.
 * @param threads The number of threads.
 * @param opt_level The optimization level.
 * @param time The total execution time.
 */
static void write_execution_configuration_to_file(const int n, const int workers,
                                                  const int threads, const int opt_level,
                                                  const double time) {
    FILE *file = fopen("execution_configs_c.csv", "a");

    if (file == NULL) {
        perror("Error: fopen failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Check if the header exists. If exists, do not write it again, otherwise write it first.
    if (ftell(file) == 0) {
        fprintf(file, "n,workers,threads,total_cpus,opt_level,time\n");
    }

    fprintf(file, "%d,%d,%d,%d,%d,%f\n", n, workers, threads, workers * threads, opt_level, time);
    fclose(file);
}

/**
 * Sets a secure random seed based on the current time.
 *
 * @see https://wiki.sei.cmu.edu/confluence/display/c/MSC32-C.+Properly+seed+pseudorandom+number+generators
 */
static void set_random_seed_secure(const int rank) {
    struct timespec ts;

    if (timespec_get(&ts, TIME_UTC) == 0) {
        perror("Error: timespec_get failed\n");
        exit(EXIT_FAILURE);
    }

    srandom(ts.tv_nsec ^ ts.tv_sec ^ rank);
}

#endif // GEMM_H

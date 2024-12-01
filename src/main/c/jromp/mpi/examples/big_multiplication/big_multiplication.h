#ifndef BIG_MULTIPLICATION_H
#define BIG_MULTIPLICATION_H

#include <assert.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define assert_non_null(ptr) assert(ptr != NULL)
#define UNUSED __attribute__((unused))

#define START_MPI_TIMER(name) \
    double name##_mpi_start = MPI_Wtime();

#define STOP_MPI_TIMER(name) \
    double name##_mpi_end = MPI_Wtime(); \
    double name##_mpi_elapsed = name##_mpi_end - name##_mpi_start;

#define GET_MPI_TIMER(name) name##_mpi_elapsed

#define START_OMP_TIMER(name) \
    double name##_omp_start = omp_get_wtime();

#define STOP_OMP_TIMER(name) \
    double name##_omp_end = omp_get_wtime(); \
    double name##_omp_elapsed = name##_omp_end - name##_omp_start;

#define GET_OMP_TIMER(name) name##_omp_elapsed

#ifdef __CUDA_RUNTIME_API_H__
#define START_CUDA_TIMER(name) \
    cudaEvent_t name##_start, name##_end; \
    cudaEventCreate(&name##_start); \
    cudaEventCreate(&name##_end); \
    cudaEventRecord(name##_start);

#define STOP_CUDA_TIMER(name) \
    cudaEventRecord(name##_end); \
    cudaEventSynchronize(name##_end); \
    float name##_elapsed; \
    cudaEventElapsedTime(&name##_elapsed, name##_start, name##_end);

#define GET_CUDA_TIMER(name) name##_elapsed

#define CUDA_CALL(x)                                                                                                   \
    {                                                                                                                  \
        const cudaError_t a = (x);                                                                                     \
        if (a != cudaSuccess) {                                                                                        \
            printf("\nError: %s (error code: %d) at %s:%d\n", cudaGetErrorString(a), a, __FILE__, __LINE__);           \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }

#define CUBLAS_CALL(x)                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t stat = (x);                                                                                     \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                                                           \
            printf("\nError: %d at %s:%d\n", stat, __FILE__, __LINE__);                                                \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }
#endif

#define MASTER
#define WORKER
#define MASTER_RANK 0
#define DATA_TAG 0
#define PROGRESS_TAG 1
#define FINISH_TAG 2

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

/**
 * Prints the given message if the current rank is 0 (Master process).
 */
#define LOG_MASTER(...)                        \
    if (rank == 0) {                           \
        printf("      Master: " __VA_ARGS__); \
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

typedef struct _progress {
    int rank;
    int rows_processed;
    double progress;
} progress;

void matrix_multiplication(const double *a, const double *b, double *c, int n, progress *progress);

void matrix_initialization(double *a, double *b, int n);

static int random_in_range(const int min, const int max) {
    return min + random() % (max - min + 1);
}

static void create_progress_type(MPI_Datatype *progress_type) {
    const MPI_Datatype types[3] = { MPI_INT, MPI_INT, MPI_DOUBLE };
    const int block_lengths[3] = { 1, 1, 1 };
    MPI_Aint offsets[3];

    offsets[0] = offsetof(progress, rank);
    offsets[1] = offsetof(progress, rows_processed);
    offsets[2] = offsetof(progress, progress);

    MPI_Type_create_struct(3, block_lengths, offsets, types, progress_type);
}

static void write_execution_configuration_to_file(const int n, const int workers,
                                                  const int threads, const int opt_level,
                                                  const double time) {
    FILE *file = fopen("execution_configs.csv", "a");

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
 * Estimated time to finish (etf) the calculations based on the progress of the process.
 *
 * @param start_time Initial time of the calculations.
 * @param progress Progress of the process.
 *
 * @return Estimated time to finish the calculations.
 */
static double etf(const double start_time, const double progress) {
    return (100.0 - progress) * (MPI_Wtime() - start_time) / progress;
}

/**
 * Estimated duration to finish the calculations based on the progress of the process.
 *
 * @param start_time Initial time of the calculations.
 * @param rows_processed Number of rows processed.
 * @param total_rows Total number of rows to be processed.
 *
 * @return Estimated duration to finish the calculations.
 */
UNUSED static double ed(const double start_time, const int rows_processed, const int total_rows) {
    return total_rows * (MPI_Wtime() - start_time) / rows_processed;
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

#endif // BIG_MULTIPLICATION_H

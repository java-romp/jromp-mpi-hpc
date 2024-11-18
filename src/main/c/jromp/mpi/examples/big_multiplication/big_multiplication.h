#ifndef BIG_MULTIPLICATION_H
#define BIG_MULTIPLICATION_H

#include <assert.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define assert_non_null(ptr) assert(ptr != NULL)

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

#define MASTER
#define WORKER
#define DATA_TAG 0
#define PROGRESS_TAG 1
#define FINISH_TAG 2

typedef struct _progress {
    int rank;
    int progress;
} progress;

void matrix_multiplication(const double *a, const double *b, double *c, int n, progress *progress);

void matrix_initialization(double *a, double *b, double *c, int n);

static int random_in_range(const int min, const int max) {
    return min + random() % (max - min + 1);
}

static void create_progress_type(MPI_Datatype *progress_type) {
    const MPI_Datatype types[2] = { MPI_INT, MPI_INT };
    const int block_lengths[2] = { 1, 1 };
    MPI_Aint offsets[2];

    offsets[0] = offsetof(progress, rank);
    offsets[1] = offsetof(progress, progress);

    MPI_Type_create_struct(2, block_lengths, offsets, types, progress_type);
}

#endif // BIG_MULTIPLICATION_H

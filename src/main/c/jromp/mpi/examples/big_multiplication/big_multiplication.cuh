#ifndef BIG_MULTIPLICATION_CUH
#define BIG_MULTIPLICATION_CUH

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

#define assert_non_null(ptr) assert(ptr != NULL)
#define UNUSED __attribute__((unused))

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
            std::cout << "\nCUDA Error: " << a << " at " << __FILE__ << ":" << __LINE__ << std::endl;                  \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }

#define CUBLAS_CALL(x)                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t stat = (x);                                                                                     \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                                                           \
            std::cout << "\nCUBLAS Error: " << stat << " at " << __FILE__ << ":" << __LINE__ << std::endl;              \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }

void matrixInitialization(double *a, double *b, size_t n);

static int randomInRange(const int min, const int max) {
    return min + random() % (max - min + 1);
}

/**
 * Sets a secure random seed based on the current time.
 *
 * @see https://wiki.sei.cmu.edu/confluence/display/c/MSC32-C.+Properly+seed+pseudorandom+number+generators
 */
static void setRandomSeedSecure(const int rank) {
    timespec ts;

    if (timespec_get(&ts, TIME_UTC) == 0) {
        perror("Error: timespec_get failed\n");
        exit(EXIT_FAILURE);
    }

    srandom(ts.tv_nsec ^ ts.tv_sec ^ rank);
}

#endif // BIG_MULTIPLICATION_CUH

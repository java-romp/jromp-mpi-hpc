#ifndef GEMM_CUH
#define GEMM_CUH

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include <vector>

#define assert_non_null(ptr) assert(ptr != NULL)
#define UNUSED __attribute__((unused))

#define START_BLOCK {
#define END_BLOCK }

#define GET_CUDA_ELAPSED(name) name##_elapsed

#define START_CUDA_TIMER(name) \
    cudaEvent_t name##_start, name##_end; \
    cudaEventCreate(&name##_start); \
    cudaEventCreate(&name##_end); \
    cudaEventRecord(name##_start); \
    START_BLOCK

#define STOP_CUDA_TIMER(name) \
    END_BLOCK \
    cudaEventRecord(name##_end); \
    cudaEventSynchronize(name##_end); \
    float name##_elapsed; \
    cudaEventElapsedTime(&name##_elapsed, name##_start, name##_end);

#define STOP_CUDA_TIMER_PRINT_ELAPSED(name) \
    STOP_CUDA_TIMER(name) \
    std::cout << "Elapsed time (" << #name << "): " << GET_CUDA_ELAPSED(name) << " ms" << std::endl;

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

using data_type = double;

/**
 * Generates a random number in the range [min, max].
 *
 * @param min minimum value.
 * @param max maximum value.
 *
 * @return a random number in the range [min, max].
 */
inline int randomInRange(const int min, const int max) {
    return min + random() % (max - min + 1);
}

/**
 * Initializes the matrices with random values.
 *
 * @param a first matrix.
 * @param b second matrix.
 * @param n matrix size.
 */
inline void matrixInitialization(double *a, double *b, const size_t n) {
    assert_non_null(a);
    assert_non_null(b);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = randomInRange(1, 1000);
            b[i * n + j] = randomInRange(1, 1000);
        }
    }
}

/**
 * Sets a secure random seed based on the current time.
 *
 * @param rank the MPI rank.
 *
 * @see https://wiki.sei.cmu.edu/confluence/display/c/MSC32-C.+Properly+seed+pseudorandom+number+generators
 */
inline void setRandomSeedSecure(const int rank) {
    timespec ts;

    if (timespec_get(&ts, TIME_UTC) == 0) {
        perror("Error: timespec_get failed\n");
        exit(EXIT_FAILURE);
    }

    srandom(ts.tv_nsec ^ ts.tv_sec ^ rank);
}

/**
 * Writes the execution configuration to a CSV file.
 *
 * @param n the matrix size.
 * @param elapsed_time the elapsed time in milliseconds.
 */
inline void writeExecutionConfigurationToFile(const int n, const double elapsed_time) {
    std::ofstream file;
    file.open("execution_configuration_cuda.csv", std::ios_base::app);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (file.tellp() == 0) {
        file << "n,elapsed_time" << std::endl;
    }

    file << n << "," << elapsed_time << std::endl;
    file.close();
}



#endif // GEMM_CUH

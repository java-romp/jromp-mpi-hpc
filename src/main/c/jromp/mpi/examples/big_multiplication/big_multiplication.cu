#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "big_multiplication.h"

void init_matrix(double *matrix, const size_t size) {
    assert_non_null(matrix);

    for (size_t i = 0; i < size; i++) {
        matrix[i] = random_in_range(1, 1000);
    }
}

int main(const int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
    }

    const auto N = static_cast<size_t>(strtoll(argv[1], nullptr, 10));

    double *d_A, *d_B, *d_C;
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    cublasHandle_t handle;

    auto *A = static_cast<double *>(malloc(N * N * sizeof(double)));
    auto *B = static_cast<double *>(malloc(N * N * sizeof(double)));
    auto *C = static_cast<double *>(malloc(N * N * sizeof(double)));

    init_matrix(A, N * N);
    init_matrix(B, N * N);

    CUDA_CALL(cudaMalloc(&d_A, N * N * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_B, N * N * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_C, N * N * sizeof(double)));

    CUBLAS_CALL(cublasCreate(&handle));

    CUBLAS_CALL(cublasSetMatrix(N, N, sizeof(double), A, N, d_A, N));
    CUBLAS_CALL(cublasSetMatrix(N, N, sizeof(double), B, N, d_B, N));
    CUBLAS_CALL(cublasSetMatrix(N, N, sizeof(double), C, N, d_C, N));

    START_CUDA_TIMER(gemm)
    // CUBLAS_CALL(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_A, N, d_B, K, &beta, d_C, N));
    CUBLAS_CALL(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, CUDA_R_16F, N, d_B, CUDA_R_16F, N,
        &beta, d_C, CUDA_R_16F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    STOP_CUDA_TIMER(gemm)

    CUBLAS_CALL(cublasGetMatrix(N, N, sizeof(double), d_C, N, C, N));

    printf("Time: %f\n", GET_CUDA_TIMER(gemm));

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}

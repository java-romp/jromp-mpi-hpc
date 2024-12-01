#include "big_multiplication.cuh"

void cublas_multiplication(const size_t N, const double *d_A, const double *d_B, double *d_C, const double alpha,
                           const double beta, const cublasHandle_t *handle) {
    START_CUDA_TIMER(cublasDgemm)
    CUBLAS_CALL(cublasDgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));
    STOP_CUDA_TIMER(cublasDgemm)
    std::cout << "Elapsed time (cublasDgemm): " << GET_CUDA_TIMER(cublasDgemm) << " ms" << std::endl;
}

int main(const int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <N>" << std::endl;
    }

    const auto N = static_cast<size_t>(strtoll(argv[1], nullptr, 10));
    std::cout << "Matrix size: " << N << std::endl;

    double *d_A, *d_B, *d_C;
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    cublasHandle_t cublasH;

    auto *A = static_cast<double *>(malloc(N * N * sizeof(double)));
    auto *B = static_cast<double *>(malloc(N * N * sizeof(double)));
    auto *C = static_cast<double *>(malloc(N * N * sizeof(double)));

    matrixInitialization(A, B, N);

    CUDA_CALL(cudaMalloc(&d_A, N * N * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_B, N * N * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_C, N * N * sizeof(double)));

    CUBLAS_CALL(cublasCreate(&cublasH));

    CUBLAS_CALL(cublasSetMatrix(N, N, sizeof(double), A, N, d_A, N));
    CUBLAS_CALL(cublasSetMatrix(N, N, sizeof(double), B, N, d_B, N));
    CUBLAS_CALL(cublasSetMatrix(N, N, sizeof(double), C, N, d_C, N));

    std::cout << "Start matrix multiplication" << std::endl;

    cublas_multiplication(N, d_A, d_B, d_C, alpha, beta, &cublasH);

    CUBLAS_CALL(cublasGetMatrix(N, N, sizeof(double), d_C, N, C, N));

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(cublasH);

    return 0;
}

void matrixInitialization(double *a, double *b, const size_t n) {
    assert_non_null(a);
    assert_non_null(b);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            a[i * n + j] = randomInRange(1, 1000);
            b[i * n + j] = randomInRange(1, 1000);
        }
    }
}

#include "big_multiplication.cuh"

int main(const int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <N>" << std::endl;
    }

    const auto N = static_cast<size_t>(strtol(argv[1], nullptr, 10));
    std::cout << "Matrix size: " << N << std::endl;

    double *d_A, *d_B, *d_C;
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    cublasHandle_t cublasH = nullptr;
    cudaStream_t stream = nullptr;

    auto *A = static_cast<double *>(malloc(N * N * sizeof(double)));
    auto *B = static_cast<double *>(malloc(N * N * sizeof(double)));
    auto *C = static_cast<double *>(malloc(N * N * sizeof(double)));

    matrixInitialization(A, B, N);

    // Initialize CUBLAS handle and bind the stream
    CUBLAS_CALL(cublasCreate(&cublasH));
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CALL(cublasSetStream(cublasH, stream));

    CUDA_CALL(cudaMalloc(&d_A, N * N * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_B, N * N * sizeof(double)));
    CUDA_CALL(cudaMalloc(&d_C, N * N * sizeof(double)));

    START_CUDA_TIMER(Gemm)
        CUDA_CALL(cudaMemcpyAsync(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CALL(cudaMemcpyAsync(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice, stream));

        CUBLAS_CALL(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));

        CUDA_CALL(cudaMemcpyAsync(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
    STOP_CUDA_TIMER_PRINT_ELAPSED(Gemm)

    free(A);
    free(B);
    free(C);
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));

    CUBLAS_CALL(cublasDestroy(cublasH));
    CUDA_CALL(cudaStreamDestroy(stream));

    CUDA_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
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

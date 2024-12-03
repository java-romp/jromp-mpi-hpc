#include "gemm.cuh"

int main(const int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <N>" << std::endl;
    }

    const auto N = static_cast<int>(strtol(argv[1], nullptr, 10));
    std::cout << "Matrix size: " << N << std::endl;

    const int m = N;
    const int n = N;
    const int k = N;
    const int lda = N;
    const int ldb = N;
    const int ldc = N;
    constexpr cublasOperation_t transA = CUBLAS_OP_N;
    constexpr cublasOperation_t transB = CUBLAS_OP_N;
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    cublasHandle_t cublasH = nullptr;
    cudaStream_t stream = nullptr;
    std::vector<data_type> A(N * N);
    std::vector<data_type> B(N * N);
    std::vector<data_type> C(N * N);
    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    matrixInitialization(A.data(), B.data(), N);

    // Initialize CUBLAS handle and bind the stream
    CUBLAS_CALL(cublasCreate(&cublasH));
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CALL(cublasSetStream(cublasH, stream));

    CUDA_CALL(cudaMalloc(&d_A, A.size() * sizeof(data_type)));
    CUDA_CALL(cudaMalloc(&d_B, B.size() * sizeof(data_type)));
    CUDA_CALL(cudaMalloc(&d_C, C.size() * sizeof(data_type)));

    START_CUDA_TIMER(Gemm)
        CUDA_CALL(cudaMemcpyAsync(d_A, A.data(), A.size() * sizeof(data_type), cudaMemcpyHostToDevice, stream));
        CUDA_CALL(cudaMemcpyAsync(d_B, B.data(), B.size() * sizeof(data_type), cudaMemcpyHostToDevice, stream));

        CUBLAS_CALL(cublasDgemm(cublasH, transA, transB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

        CUDA_CALL(cudaMemcpyAsync(C.data(), d_C, C.size() * sizeof(data_type), cudaMemcpyDeviceToHost, stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
    STOP_CUDA_TIMER_PRINT_ELAPSED(Gemm)

    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));

    CUBLAS_CALL(cublasDestroy(cublasH));
    CUDA_CALL(cudaStreamDestroy(stream));

    CUDA_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}

void matrixInitialization(double *a, double *b, const int n) {
    assert_non_null(a);
    assert_non_null(b);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = randomInRange(1, 1000);
            b[i * n + j] = randomInRange(1, 1000);
        }
    }
}

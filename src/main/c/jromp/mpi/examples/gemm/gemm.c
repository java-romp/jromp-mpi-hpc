#include "gemm.h"

omp_lock_t print_lock;  // Lock to prevent interleaved output
int workers;            // Number of worker processes
int N;                  // Matrix size
int threads;            // Number of threads per process
int optimization_level; // Optimization level
int rank;               // Rank of the current process
int size;               // Number of processes

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <N> <threads> <optimization_level>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int provided;
    const int required = MPI_THREAD_MULTIPLE;

    MPI_Init_thread(&argc, &argv, required, &provided);

    if (provided < required) {
        perror("Error: MPI does not provide the required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    set_random_seed_secure(rank);

    // Initialize globals
    omp_init_lock(&print_lock);
    workers = size - 1;
    N = (int) strtol(argv[1], NULL, 10);
    threads = (int) strtol(argv[2], NULL, 10);
    optimization_level = (int) strtol(argv[3], NULL, 10);

    LOG_MASTER("Information: N = %d, threads per rank = %d, optimization_level = %d\n", N, threads, optimization_level);
    LOG_MASTER("Checking the number of threads in all ranks...\n");

    MPI_Barrier(MPI_COMM_WORLD);

    // Check that all ranks have the same number of threads
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp master
        {
            printf("Number of threads of rank %d: %d\n", rank, omp_get_num_threads());
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const int rows_per_worker = N / workers; // Exclude the master process
    double *A;
    double *B;
    double *C;

    if (rank == MASTER_RANK) {
        // Only master process allocates memory for all complete matrices
        A = malloc(N * N * sizeof(double));
        B = malloc(N * N * sizeof(double));
        C = malloc(N * N * sizeof(double));

        // Check memory allocation
        if (A == NULL || B == NULL || C == NULL) {
            perror("Error: malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        LOG_MASTER("*************************************\n");
        LOG_MASTER("******* Matrix Initialization *******\n");
        LOG_MASTER("*************************************\n");
        START_OMP_TIMER(initialization)
            matrix_initialization(A, B, N);
        STOP_OMP_TIMER(initialization)
        LOG_MASTER("Time to initialize the matrices: %fs\n", GET_OMP_TIMER(initialization));

        LOG_MASTER("*************************************\n");
        LOG_MASTER("****** Sending data to workers ******\n");
        LOG_MASTER("*************************************\n");
        START_MPI_TIMER(send_data)
            MPI_Request *requests = malloc(2 * workers * sizeof(MPI_Request));

            // Distribute rows of A to workers and send matrix B to all workers
            for (int i = 1; i < size; i++) {
                MPI_Isend(&A[(i - 1) * rows_per_worker * N], rows_per_worker * N, MPI_DOUBLE, i, DATA_TAG,
                          MPI_COMM_WORLD, &requests[i - 1]);
                MPI_Isend(B, N * N, MPI_DOUBLE, i, DATA_TAG, MPI_COMM_WORLD, &requests[workers + i - 1]);
            }

            MPI_Waitall(2 * workers, requests, MPI_STATUSES_IGNORE);
            free(requests);
        STOP_MPI_TIMER(send_data)
        LOG_MASTER("Time to send data to workers: %fs\n", GET_MPI_TIMER(send_data));

        // Send a message to workers (without body) to indicate the start of the calculations
        for (int i = 1; i < size; i++) {
            MPI_Send(NULL, 0, MPI_BYTE, i, START_MULTIPLICATION_TAG, MPI_COMM_WORLD);
        }

        LOG_MASTER("*************************************\n");
        LOG_MASTER("******* Matrix Multiplication *******\n");
        LOG_MASTER("*************************************\n");
        START_MPI_TIMER(calculations)
            int ended_workers = 0;
            MPI_Status status;

            do {
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                if (status.MPI_TAG == FINISH_TAG) {
                    MPI_Recv(&C[(status.MPI_SOURCE - 1) * rows_per_worker * N], rows_per_worker * N, MPI_DOUBLE,
                             status.MPI_SOURCE, FINISH_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    LOG_MASTER("Worker %d has finished\n", status.MPI_SOURCE);
                    ended_workers++;
                } else {
                    // Unexpected message
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    LOG_MASTER("Unexpected message\n");

                    // Free memory and exit
                    free(A);
                    free(B);
                    free(C);

                    exit(EXIT_FAILURE);
                }
            } while (ended_workers < workers);
        STOP_MPI_TIMER(calculations)
        LOG_MASTER("Time to do the calculations: %f\n", GET_MPI_TIMER(calculations));

        LOG_MASTER("Writing execution configuration to file\n");
        write_execution_configuration_to_file(N, workers, threads, optimization_level, GET_MPI_TIMER(calculations));
    } else {
        // Workers allocate memory for their part of the matrices
        A = malloc(rows_per_worker * N * sizeof(double));
        B = malloc(N * N * sizeof(double)); // All workers need the matrix B
        C = malloc(rows_per_worker * N * sizeof(double));

        // Check memory allocation
        if (A == NULL || B == NULL || C == NULL) {
            perror("Error: malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Receive rows of A and matrix B
        MPI_Recv(A, rows_per_worker * N, MPI_DOUBLE, MASTER_RANK, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B, N * N, MPI_DOUBLE, MASTER_RANK, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Wait for the start message
        MPI_Recv(NULL, 0, MPI_BYTE, MASTER_RANK, START_MULTIPLICATION_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        LOG_WORKER("Worker %d has started\n", rank);

        // Perform matrix multiplication
        gemm(A, B, C, rows_per_worker);
        // C is filled during multiplication

        // Send results back to master process
        MPI_Send(C, rows_per_worker * N, MPI_DOUBLE, MASTER_RANK, FINISH_TAG, MPI_COMM_WORLD);
    }

    // Free memory
    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

WORKER void gemm(const double *a, const double *b, double *c, const int rows_per_worker) {
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(c);

    #pragma omp parallel shared(a, b, c, rows_per_worker) num_threads(threads)
    {
        double local_sum;
        int i, j, k;

        #pragma omp for
        for (i = 0; i < rows_per_worker; i++) {
            for (j = 0; j < N; j++) {
                local_sum = 0;

                for (k = 0; k < N; k++) {
                    local_sum += a[i * N + k] * b[k * N + j];
                }

                c[i * N + j] = local_sum;
            }
        }
    }
}

MASTER void matrix_initialization(double *a, double *b, const int n) {
    assert_non_null(a);
    assert_non_null(b);

    #pragma omp parallel for shared(a, b, n) num_threads(threads) if(n > 10000)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = random_in_range(1, 1000);
            b[i * n + j] = random_in_range(1, 1000);
        }
    }
}

// Last revision (scastd): 19/01/2025 00:18

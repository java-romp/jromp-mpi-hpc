#include "gemm.h"

omp_lock_t print_lock; // Lock to prevent interleaved output
int workers;
int N;
int threads;
int optimization_level;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    if (argc != 4) {
        printf("Usage: %s <N> <threads> <optimization_level>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    set_random_seed_secure(rank);

    // Initialize globals
    omp_init_lock(&print_lock);
    workers = size - 1;
    N = (int) strtol(argv[1], NULL, 10);
    threads = (int) strtol(argv[2], NULL, 10);
    optimization_level = (int) strtol(argv[3], NULL, 10);

    LOG_MASTER("Information: N = %d, threads = %d, optimization_level = %d\n", N, threads, optimization_level);
    omp_set_num_threads(threads);

    // This block is just for checking the number of threads
    #pragma omp parallel
    {
        LOG_WORKER("I am the thread %d\n", omp_get_thread_num());
    }

    const int rows_per_worker = N / workers; // Exclude the master process
    double *a = NULL, *b = NULL, *c = NULL;

    if (rank == 0) {
        // Only master process allocates memory for all complete matrices
        a = malloc(N * N * sizeof(double));
        b = malloc(N * N * sizeof(double));
        c = calloc(N * N, sizeof(double)); // Initialize to zero

        // Check memory allocation
        if (a == NULL || b == NULL || c == NULL) {
            perror("Error: malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        LOG_MASTER("*************************************\n");
        LOG_MASTER("******* Matrix Initialization *******\n");
        LOG_MASTER("*************************************\n");

        START_OMP_TIMER(initialization);

        // Initialize matrices
        matrix_initialization(a, b, N);

        STOP_OMP_TIMER(initialization);
        LOG_MASTER("Time to initialize the matrices: %fs\n", GET_OMP_TIMER(initialization));

        START_MPI_TIMER(calculations);

        LOG_MASTER("*************************************\n");
        LOG_MASTER("******* Matrix Multiplication *******\n");
        LOG_MASTER("*************************************\n");

        MPI_Request *requests = malloc(2 * workers * sizeof(MPI_Request));

        // Send rows of A to workers and matrix B to all workers
        for (int i = 1; i < size; i++) {
            MPI_Isend(&a[(i - 1) * rows_per_worker * N], rows_per_worker * N, MPI_DOUBLE, i, DATA_TAG, MPI_COMM_WORLD,
                      &requests[i - 1]);
            MPI_Isend(b, N * N, MPI_DOUBLE, i, DATA_TAG, MPI_COMM_WORLD, &requests[workers + i - 1]);
        }

        MPI_Waitall(2 * workers, requests, MPI_STATUSES_IGNORE);
        free(requests);

        int ended_workers = 0;
        MPI_Status status;
        Progress progress = { 0 };
        double row_time_start = calculations_mpi_start;
        double row_time_end = 0;

        do {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (UNLIKELY(status.MPI_TAG == FINISH_TAG)) {
                // ^^ Marked as unlikely because the available ranks are not a big number, so this condition is not expected
                // to happen frequently.
                MPI_Recv(&c[(status.MPI_SOURCE - 1) * rows_per_worker * N], rows_per_worker * N, MPI_DOUBLE,
                         status.MPI_SOURCE, FINISH_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                LOG_MASTER("Worker %d has finished\n", status.MPI_SOURCE);
                ended_workers++;
            } else if (LIKELY(status.MPI_TAG == PROGRESS_TAG)) {
                // ^^ Marked as likely because the progress is sent very frequently during the calculations. This condition
                // is expected to happen frequently.
                MPI_Recv(&progress, sizeof(Progress), MPI_BYTE, status.MPI_SOURCE, PROGRESS_TAG, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                row_time_end = MPI_Wtime();

                // Notation:
                //  - T_r: Time to process a row.
                //  - T_t: Time total (from the beginning of the calculations).
                //  - ETF: Estimated time to finish the calculations.
                LOG_MASTER("Progress of worker %d: %f%% (%d/%d)  ::  T_r: %.5fs   T_t: %.5fs   ETF: %.5fs\n",
                           progress.rank, progress.progress, progress.rows_processed, rows_per_worker,
                           row_time_end - row_time_start, row_time_end - calculations_mpi_start,
                           etf(calculations_mpi_start, progress.progress));

                row_time_start = row_time_end;
            } else {
                // Unexpected message
                MPI_Abort(MPI_COMM_WORLD, 1);
                printf("Unexpected message\n");

                free(a);
                free(b);
                free(c);
                return EXIT_FAILURE;
            }
        } while (ended_workers < workers);

        STOP_MPI_TIMER(calculations);
        LOG_MASTER("Total time to do the calculations: %f\n", GET_MPI_TIMER(calculations));

        // Free memory
        free(a);
        free(b);
        free(c);

        LOG_MASTER("Writing execution configuration to file\n");
        write_execution_configuration_to_file(N, workers, threads, optimization_level, GET_MPI_TIMER(calculations));
    } else {
        LOG_WORKER("Number of threads: %d\n", omp_get_num_threads());

        // Workers allocate memory for their part of the matrices
        a = malloc(rows_per_worker * N * sizeof(double));
        b = malloc(N * N * sizeof(double)); // All workers need the matrix B
        c = malloc(rows_per_worker * N * sizeof(double));
        Progress progress = { rank, 0, 0.0 };

        // Check memory allocation
        if (a == NULL || b == NULL || c == NULL) {
            perror("Error: malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Receive rows of A and matrix B
        MPI_Recv(a, rows_per_worker * N, MPI_DOUBLE, MASTER_RANK, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, N * N, MPI_DOUBLE, MASTER_RANK, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Perform matrix multiplication
        matrix_multiplication(a, b, c, rows_per_worker, &progress);

        // Send results back to master process
        MPI_Send(c, rows_per_worker * N, MPI_DOUBLE, MASTER_RANK, FINISH_TAG, MPI_COMM_WORLD);

        // Free memory
        free(b);
        free(a);
        free(c);
    }

    MPI_Finalize();
    return 0;
}

WORKER void matrix_multiplication(const double *a, const double *b, double *c, const int n, Progress *progress) {
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(c);

    MPI_Request request;
    double local_sum = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < N; j++) {
            local_sum = 0;

            for (int k = 0; k < N; k++) {
                local_sum += a[i * N + k] * b[k * N + j];
            }

            c[i * N + j] = local_sum;
        }

        progress->progress = (i + 1.0) / n * 100.0;
        progress->rows_processed = i + 1;
        // Send asynchronous progress to master process to avoid blocking. No wait for the request to complete
        // because it is not necessary to know if the master process has received the progress
        // (it is only informative).
        MPI_Isend(progress, sizeof(Progress), MPI_BYTE, MASTER_RANK, PROGRESS_TAG, MPI_COMM_WORLD, &request);
    }
}

MASTER void matrix_initialization(double *a, double *b, const int n) {
    assert_non_null(a);
    assert_non_null(b);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = random_in_range(1, 1000);
            b[i * n + j] = random_in_range(1, 1000);
        }
    }
}

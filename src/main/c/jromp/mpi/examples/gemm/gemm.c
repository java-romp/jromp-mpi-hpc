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
        return EXIT_FAILURE;
    }

    int provided;
    const int required = MPI_THREAD_MULTIPLE;

    MPI_Init_thread(&argc, &argv, required, &provided);

    if (provided < required) {
        printf("Error: MPI does not provide the required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
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
    omp_set_num_threads(threads);

    LOG_MASTER("Information: N = %d, threads per rank = %d, optimization_level = %d\n",
               N, threads, optimization_level);
    LOG_MASTER("Checking the number of threads in all ranks...\n");

    MPI_Barrier(MPI_COMM_WORLD);

    // Check that all ranks have the same number of threads
    #pragma omp parallel
    {
        #pragma omp master
        {
            printf("Number of threads of rank %d: %d\n", rank, omp_get_num_threads());
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

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

        LOG_MASTER("*************************************\n");
        LOG_MASTER("****** Sending data to workers ******\n");
        LOG_MASTER("*************************************\n");

        START_MPI_TIMER(send_data);

        MPI_Request *requests = malloc(2 * workers * sizeof(MPI_Request));

        // Send rows of A to workers and matrix B to all workers
        for (int i = 1; i < size; i++) {
            MPI_Isend(&a[(i - 1) * rows_per_worker * N], rows_per_worker * N, MPI_DOUBLE, i, DATA_TAG, MPI_COMM_WORLD,
                      &requests[i - 1]);
            MPI_Isend(b, N * N, MPI_DOUBLE, i, DATA_TAG, MPI_COMM_WORLD, &requests[workers + i - 1]);
        }

        MPI_Waitall(2 * workers, requests, MPI_STATUSES_IGNORE);
        free(requests);

        STOP_MPI_TIMER(send_data);
        LOG_MASTER("Time to send data to workers: %fs\n", GET_MPI_TIMER(send_data));

        LOG_MASTER("*************************************\n");
        LOG_MASTER("******* Matrix Multiplication *******\n");
        LOG_MASTER("*************************************\n");

        START_MPI_TIMER(calculations);

        int ended_workers = 0;
        double row_time_start = calculations_mpi_start;
        double row_time_end = 0;
        MPI_Status status;
        progress_t progress = { 0 };
        progress_t global_progress = { 0 };

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
                MPI_Recv(&progress, sizeof(progress_t), MPI_BYTE, status.MPI_SOURCE, PROGRESS_TAG, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                row_time_end = MPI_Wtime();

                global_progress.rows_processed++;
                global_progress.progress = (float) global_progress.rows_processed / (float) N * 100;

                // Notation:
                //  - T_r: Time to process a row.
                //  - T_t: Time total (from the beginning of the calculations).
                //  - ETF: Estimated time to finish the calculations.
                LOG_MASTER(
                        "Progress of worker %d (Thread %d): %f%% (%d/%d) (overall: %f%% (%d/%d))  ::  T_r: %.5fs   T_t: %.5fs   ETF: %.5fs\n",
                        progress.rank, progress.thread, progress.progress, progress.rows_processed,
                        rows_per_worker / threads, global_progress.progress, global_progress.rows_processed, N,
                        row_time_end - row_time_start, row_time_end - calculations_mpi_start,
                        etf(calculations_mpi_start, global_progress.progress));

                row_time_start = row_time_end;
            } else {
                // Unexpected message
                MPI_Abort(MPI_COMM_WORLD, 1);
                LOG_MASTER("Unexpected message\n");

                free(a);
                free(b);
                free(c);
                return EXIT_FAILURE;
            }
        } while (ended_workers < workers);

        STOP_MPI_TIMER(calculations);
        LOG_MASTER("Time to do the calculations: %f\n", GET_MPI_TIMER(calculations));

        LOG_MASTER("Writing execution configuration to file\n");
        write_execution_configuration_to_file(N, workers, threads, optimization_level, GET_MPI_TIMER(calculations));
    } else {
        // Workers allocate memory for their part of the matrices
        a = malloc(rows_per_worker * N * sizeof(double));
        b = malloc(N * N * sizeof(double)); // All workers need the matrix B
        c = malloc(rows_per_worker * N * sizeof(double));

        // Check memory allocation
        if (a == NULL || b == NULL || c == NULL) {
            perror("Error: malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Receive rows of A and matrix B
        MPI_Recv(a, rows_per_worker * N, MPI_DOUBLE, MASTER_RANK, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, N * N, MPI_DOUBLE, MASTER_RANK, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Perform matrix multiplication
        matrix_multiplication(a, b, c, rows_per_worker, rank);

        // Send results back to master process
        MPI_Send(c, rows_per_worker * N, MPI_DOUBLE, MASTER_RANK, FINISH_TAG, MPI_COMM_WORLD);
    }

    // Free memory
    free(a);
    free(b);
    free(c);

    MPI_Finalize();
    return 0;
}

WORKER void matrix_multiplication(const double *a, const double *b, double *c, const int rows_per_worker,
                                  const int rank) {
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(c);

    const int rows_per_thread = rows_per_worker / threads;
    progress_t *progresses = calloc(threads, sizeof(progress_t));
    MPI_Request ignored_request;

    #pragma omp parallel shared(a, b, c, rows_per_worker, rank, rows_per_thread, progresses, ignored_request)
    {
        double local_sum;
        int current_row, i, j, k;
        progress_t *thread_progress;
        const int thread_num = omp_get_thread_num(); // Prevent multiple calls to the function inside the for loop

        #pragma omp for private(local_sum, current_row, i, j, k)
        for (i = 0; i < rows_per_worker; i++) {
            for (j = 0; j < N; j++) {
                local_sum = 0;

                for (k = 0; k < N; k++) {
                    local_sum += a[i * N + k] * b[k * N + j];
                }

                c[i * N + j] = local_sum;
            }

            current_row = i % rows_per_thread;
            thread_progress = &progresses[thread_num];
            thread_progress->rank = rank;
            thread_progress->rows_processed = current_row + 1;
            thread_progress->thread = thread_num;
            thread_progress->progress = (float) thread_progress->rows_processed / (float) rows_per_thread * 100;

            /*
             * Send asynchronous progress to master rank to avoid blocking.
             * Waiting for the request to complete is not necessary because we don't want to know if the
             * master process has received the progress (it is only informative). If it is lost, the master
             * process will not have the progress of the worker, but the calculations will continue.
             */
            MPI_Isend(thread_progress, sizeof(progress_t), MPI_BYTE, MASTER_RANK, PROGRESS_TAG, MPI_COMM_WORLD,
                      &ignored_request);
        }

        free(progresses);
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

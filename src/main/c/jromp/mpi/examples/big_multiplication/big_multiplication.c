#include "big_multiplication.h"

MPI_Datatype progress_type;
int N;
int threads;
int optimization_level;

int main(int argc, char *argv[]) {
    srand(time(NULL));
    MPI_Init(&argc, &argv);

    if (argc != 4) {
        printf("Usage: %s <N> <threads> <optimization_level>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    N = (int) strtol(argv[1], NULL, 10);
    threads = (int) strtol(argv[2], NULL, 10);
    optimization_level = (int) strtol(argv[3], NULL, 10);

    printf("Information: N = %d, threads = %d, optimization_level = %d\n", N, threads, optimization_level);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    create_progress_type(&progress_type);
    MPI_Type_commit(&progress_type);

    const int rows_per_process = N / (size - 1); // Exclude the master process
    double *a = NULL, *b = NULL, *c = NULL;

    if (rank == 0) {
        // Only master process allocates memory for all complete matrices
        a = malloc(N * N * sizeof(double));
        b = malloc(N * N * sizeof(double));
        c = malloc(N * N * sizeof(double));

        // Check memory allocation
        if (a == NULL || b == NULL || c == NULL) {
            perror("Error: malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        START_OMP_TIMER(initialization);

        // Initialize matrices
        matrix_initialization(a, b, c, N);

        STOP_OMP_TIMER(initialization);
        printf("Time to initialize the matrices: %f\n", GET_OMP_TIMER(initialization));

        START_MPI_TIMER(calculations);

        // Send rows of A to workers and matrix B to all workers
        for (int i = 1; i < size; i++) {
            MPI_Send(&a[(i - 1) * rows_per_process * N], rows_per_process * N, MPI_DOUBLE, i, DATA_TAG, MPI_COMM_WORLD);
            MPI_Send(b, N * N, MPI_DOUBLE, i, DATA_TAG, MPI_COMM_WORLD);
        }

        int ended_workers = 0;
        MPI_Status status;
        progress progress = { 0 };

        do {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == FINISH_TAG) {
                MPI_Recv(&c[(status.MPI_SOURCE - 1) * rows_per_process * N], rows_per_process * N, MPI_DOUBLE,
                         status.MPI_SOURCE, FINISH_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                printf("Worker %d has finished\n", status.MPI_SOURCE);
                ended_workers++;
            } else if (status.MPI_TAG == PROGRESS_TAG) {
                MPI_Recv(&progress, 1, progress_type, status.MPI_SOURCE, PROGRESS_TAG, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                printf("Worker %d has completed %d%%\n", progress.rank, progress.progress);
            } else {
                // Unexpected message
                MPI_Abort(MPI_COMM_WORLD, 1);
                printf("Unexpected message\n");

                free(a);
                free(b);
                free(c);
                return EXIT_FAILURE;
            }
        } while (ended_workers < size - 1);

        STOP_MPI_TIMER(calculations);
        printf("Total time to do the calculations: %f\n", GET_MPI_TIMER(calculations));

        // Free memory
        free(a);
        free(b);
        free(c);
    } else {
        // Workers allocate memory for their part of the matrices
        b = malloc(N * N * sizeof(double)); // All workers need the matrix B
        double *sub_A = malloc(rows_per_process * N * sizeof(double));
        double *sub_C = malloc(rows_per_process * N * sizeof(double));
        progress progress = { rank, 0 };

        // Check memory allocation
        if (b == NULL || sub_A == NULL || sub_C == NULL) {
            perror("Error: malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Receive rows of A and matrix B
        MPI_Recv(sub_A, rows_per_process * N, MPI_DOUBLE, 0, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, N * N, MPI_DOUBLE, 0, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Perform matrix multiplication
        matrix_multiplication(sub_A, b, sub_C, rows_per_process, &progress);

        // Send results back to master process
        MPI_Send(sub_C, rows_per_process * N, MPI_DOUBLE, 0, FINISH_TAG, MPI_COMM_WORLD);

        // Free memory
        free(b);
        free(sub_A);
        free(sub_C);
    }

    MPI_Finalize();
    return 0;
}

WORKER void matrix_multiplication(const double *a, const double *b, double *c, const int n, progress *progress) {
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(c);

    MPI_Request request;

    for (int i = 0; i < n; i++) {
        progress->progress = (int) (i / (double) n * 100);
        // Send asynchronous progress to master process to avoid blocking. No wait for the request to complete
        // because it is not necessary to know if the master process has received the progress
        // (it is only informative).
        MPI_Isend(progress, 1, progress_type, 0, PROGRESS_TAG, MPI_COMM_WORLD, &request);

        for (int j = 0; j < N; j++) {
            c[i * N + j] = 0;

            for (int k = 0; k < N; k++) {
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}

MASTER void matrix_initialization(double *a, double *b, double *c, const int n) {
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(c);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = random_in_range(1, 1000);
            b[i * n + j] = random_in_range(1, 1000);
            c[i * n + j] = 0;
        }
    }
}

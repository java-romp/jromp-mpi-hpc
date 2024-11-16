#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define N 20000
//#define DEBUG

int main(int argc, char *argv[]) {
    int rank;
    int size;
    int i;
    int j;
    int num_elements;
    double *matrix;
    double *matrix_chunk;
    double local_sum;
    double global_sum;
    double mean;
    double start_time;
    double end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL));

    num_elements = (N * N) / size;
    matrix_chunk = (double *) malloc(num_elements * sizeof(double));

    printf("Number of elements: %d\n", num_elements);

    if (rank == 0) {
        matrix = (double *) malloc(N * N * sizeof(double));

        // Measure the initialization time of the matrix
        start_time = MPI_Wtime();

        // Initialize the matrix with random values
        for (i = 0; i < N * N; i++) {
#ifdef DEBUG
			matrix[i] = 10.0;
#else
            matrix[i] = rand() % 101;
#endif
        }

        end_time = MPI_Wtime();
        printf("Initialization time took %f seconds\n", end_time - start_time);
    }

    start_time = MPI_Wtime();

    // Distribute the data among all the processes
    MPI_Scatter(matrix, num_elements, MPI_DOUBLE, matrix_chunk, num_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate the sum of the received elements
    local_sum = 0.0;
    for (i = 0; i < num_elements; i++) {
        local_sum += matrix_chunk[i];
    }

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Calculate the mean
    mean = global_sum / (N * N);

#ifdef DEBUG
	printf("Rank %d: mean = %f\n", rank, mean);
#endif

    printf("Number of elements: %d\n", num_elements);
    // Update the matrix chunk by dividing each element by the mean
    for (i = 0; i < num_elements; i++) {
        matrix_chunk[i] /= mean;
    }

    // Send the updated chunks back to the root process
    MPI_Gather(matrix_chunk, num_elements, MPI_DOUBLE, matrix, num_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print the execution time on the master process
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Execution time: %f seconds\n", end_time - start_time);

        // Write to a file the final execution time. The name depends on the number received as argument
        if (argc == 2) {
            char filename[20];
            sprintf(filename, "results/%s.txt", argv[1]);
            FILE *fp = fopen(filename, "a");
            fprintf(fp, "%f seconds\n", end_time - start_time);
            fclose(fp);
        }

        free(matrix);
    }

    // Free memory
    free(matrix_chunk);

    MPI_Finalize();
    return 0;
}

#include "genome.h"

int main(int argc, string argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #ifdef DEBUG_LOGGING
    // Initialize the print lock, to prevent interleaved output
    omp_init_lock(&print_lock);
    #endif

    SHARED cvector(string) directories = NULL;

    // Only execute on master node
    if (rank == 0) {
        if (argc != 2) {
            fprintf(stderr, "Usage: %s <directory>\n", argv[0]);
            return 1;
        }

        START_MPI_TIMER(total)
        START_MPI_TIMER(get_dirs)
        const int num_dirs = get_dirs(argv[1], &directories);
        STOP_MPI_TIMER_PRINT(get_dirs, LOG_MASTER)

        if (num_dirs == -1) {
            return 1;
        }

        // Divide the directories among the worker nodes (excluding the master node)
        const int num_workers = size - 1;
        const int num_dirs_per_worker = num_dirs / num_workers;
        const int num_dirs_remainder = num_dirs % num_workers;
        int start = 0;
        int end = num_dirs_per_worker;

        // Send the directories to the worker nodes
        for (int i = 1; i < size; i++) {
            // Adjust the end index if there are remaining directories, by sending them to the first workers.
            // This ensures that the directories are evenly distributed among the workers.
            if (i <= num_dirs_remainder) {
                end++;
            }

            const int num_dirs_to_send = end - start;
            LOG_MASTER("Sending %d directories to worker %d\n", num_dirs_to_send, i);

            // Send the number of directories to expect
            MPI_Send(&num_dirs_to_send, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

            // Send the directories
            for (int j = start; j < end; j++) {
                const string directory = directories[j];
                const int directory_size = (int) strlen(directory) + 1;

                MPI_Send(&directory_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(directory, directory_size, MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }

            start = end;
            end += num_dirs_per_worker;
        }

        // Receive the results from the worker nodes
        int total_files = 0;
        struct dna_sequence dna_sequence = { 0 };

        MPI_Status status;
        for (int i = 1; i < size; i++) {
            int files;
            MPI_Recv(&files, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            total_files += files;

            struct dna_sequence worker_dna_sequence;
            MPI_Recv(&worker_dna_sequence, sizeof(struct dna_sequence), MPI_BYTE, i, 0, MPI_COMM_WORLD, &status);

            dna_sequence.A += worker_dna_sequence.A;
            dna_sequence.C += worker_dna_sequence.C;
            dna_sequence.G += worker_dna_sequence.G;
            dna_sequence.T += worker_dna_sequence.T;
            dna_sequence.U += worker_dna_sequence.U;
            dna_sequence.N += worker_dna_sequence.N;
        }

        STOP_MPI_TIMER(total)

        // Print the results
        LOG_MASTER("################## Results ##################\n");
        LOG_MASTER("Files processed: %d\n", total_files);
        LOG_MASTER("Total DNA sequence:\n");
        LOG_MASTER("Time spent in total: %f seconds\n", total_mpi_elapsed);
        pretty_print_dna_sequence(&dna_sequence);
        LOG_MASTER("################## End of Results ##################\n");
    } else {
        // Receive the number of directories to expect
        int num_dirs_to_receive;
        MPI_Recv(&num_dirs_to_receive, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cvector_init(directories, num_dirs_to_receive, NULL);

        // Receive the directories
        for (int i = 0; i < num_dirs_to_receive; i++) {
            int directory_size;
            MPI_Recv(&directory_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            string directory = malloc(directory_size);
            MPI_Recv(directory, directory_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            cvector_push_back(directories, directory);
        }

        cvector_clib_assert(directories); // Ensure that the directories vector is not NULL

        // Process the directories
        SHARED int files = 0;
        PRIVATE string directory;
        SHARED struct dna_sequence dna_sequence = { 0 };
        SHARED bool error = false;
        const size_t size = cvector_size(directories);

        START_OMP_TIMER(process_dirs)

        #pragma omp parallel for \
            reduction(+ : files) \
            private(directory) \
            shared(directories, dna_sequence, error)
        for (int i = 0; i < size; i++) {
            if (LIKELY(!error)) {
                directory = directories[i];

                const int dir_files = process_directory(PRIVATE directory, SHARED &dna_sequence);
                if (dir_files == -1) {
                    #pragma omp atomic write
                    error = true;
                    continue; // Do not update the files counter and exit the loop immediately
                }

                // Process the directory
                files += dir_files;
            }
        }

        STOP_OMP_TIMER_PRINT(process_dirs, LOG_WORKER)

        if (UNLIKELY(error)) {
            LOG_WORKER("An error occurred while processing the directories\n");
            LOG_WORKER("Files processed: %d\n", files);
            cvector_free(directories);
            return EXIT_FAILURE;
        }

        LOG_WORKER("Files processed: %d\n", files);
        pretty_print_dna_sequence(&dna_sequence);

        // Send the results back to the master node
        MPI_Send(&files, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&dna_sequence, sizeof(struct dna_sequence), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }

    cvector_free(directories);

    #ifdef DEBUG_LOGGING
    omp_destroy_lock(&print_lock);
    #endif
    MPI_Finalize();
    return EXIT_SUCCESS;
}

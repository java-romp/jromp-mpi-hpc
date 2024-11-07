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

        const string dir = argv[1];
        const int num_dirs = get_dirs(dir, &directories);

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

        cvector_free(directories);
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

        if (UNLIKELY(error)) {
            LOG_WORKER("An error occurred while processing the directories\n");
            LOG_WORKER("Files processed: %d\n", files);
            cvector_free(directories);
            return EXIT_FAILURE;
        }

        LOG_WORKER("Files processed: %d\n", files);
        pretty_print_dna_sequence(&dna_sequence);
        cvector_free(directories);
    }

    #ifdef DEBUG_LOGGING
    omp_destroy_lock(&print_lock);
    #endif
    MPI_Finalize();
    return EXIT_SUCCESS;
}

#include "genome.h"

int get_dirs(const string directory_path, cvector(string) * directories) {
    DIR *dir;
    struct dirent *dir_entry;
    struct stat dir_stat;

    // Open the directory (if possible). If not, return -1 (error)
    if ((dir = opendir(directory_path)) == NULL) {
        fprintf(stderr, "Error: Could not open directory %s\n", directory_path);
        return -1;
    }

    int num_dirs = 0;
    *directories = NULL;

    // Iterate over the directory entries
    while ((dir_entry = readdir(dir)) != NULL) {
        char full_path[MAX_PATH_SIZE];
        snprintf(full_path, sizeof(full_path), "%s/%s", directory_path, dir_entry->d_name);

        if (stat(full_path, &dir_stat) == 0 && S_ISDIR(dir_stat.st_mode)) {
            // Ignore the current and parent directories
            if (strcmp(dir_entry->d_name, ".") != 0 && strcmp(dir_entry->d_name, "..") != 0) {
                // Add the full directory path to the list
                cvector_push_back(*directories, strdup(full_path));
                num_dirs++;
            }
        }
    }

    closedir(dir);
    return num_dirs;
}

int main(int argc, string argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef DEBUG_LOGGING
    // Initialize the print lock, to prevent interleaved output
    omp_lock_t print_lock;
    omp_init_lock(&print_lock);
#endif

    cvector(string) directories;

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

        // Receive the directories
        for (int i = 0; i < num_dirs_to_receive; i++) {
            int directory_size;
            MPI_Recv(&directory_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            string directory = malloc(directory_size);
            MPI_Recv(directory, directory_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            cvector_push_back(directories, directory);
        }

        LOG_WORKER("Received %d directories\n", num_dirs_to_receive);
    }

#ifdef DEBUG_LOGGING
    omp_destroy_lock(&print_lock);
#endif
    MPI_Finalize();
    return 0;
}

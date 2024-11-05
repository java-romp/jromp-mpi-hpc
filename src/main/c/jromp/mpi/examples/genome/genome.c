#include <dirent.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "cvector.h"

#define MAX_PATH_SIZE 1024
#define string char *
#define print_master(...)                                                                                              \
    if (rank == 0) {                                                                                                   \
        printf("    Master: " __VA_ARGS__);                                                                            \
    }
#define print_worker(...)                                                                                              \
    if (rank != 0) {                                                                                                   \
        printf("Worker %03d: ", rank);                                                                                 \
        printf(__VA_ARGS__);                                                                                           \
    }

int list_directory(const string directory_path, cvector(string) * directories) {
    DIR *dir;
    struct dirent *dir_entry;
    struct stat dir_stat;

    // Open the directory (if possible). If not, return -1 (error)
    if ((dir = opendir(directory_path)) == NULL) {
        fprintf(stderr, "Error: Could not open directory %s\n", directory_path);
        return -1;
    }

    int num_files = 0;
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
                num_files++; // Increment the number of directories
            }
        }
    }

    closedir(dir);
    return num_files;
}

int main(int argc, string argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Only execute on master node
    if (rank == 0) {
        if (argc != 2) {
            fprintf(stderr, "Usage: %s <directory>\n", argv[0]);
            return 1;
        }

        const string dir = argv[1];
        cvector(string) directories;
        const int num_dirs = list_directory(dir, &directories);

        if (num_dirs == -1) {
            return 1;
        }

        // string *it;
        // cvector_for_each_in(it, directories) {
        //     printf("Directory %s\n", *it);
        // }

        print_master("Number of directories: %d\n", num_dirs);
        cvector_free(directories);
    } else {
        print_worker("Hello, world!\n");
    }

    MPI_Finalize();
    return 0;
}

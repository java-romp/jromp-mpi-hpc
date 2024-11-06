#include "genome.h"

int get_dirs(const string directory_path, cvector(string) * directories) {
    DIR *dir;

    // Open the directory (if possible). If not, return -1 (error)
    if ((dir = opendir(directory_path)) == NULL) {
        fprintf(stderr, "Error: Could not open directory %s\n", directory_path);
        return -1;
    }

    int num_dirs = 0;
    struct dirent *dir_entry;
    struct stat dir_stat;

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

void process_directory(const string directory) {
    DIR *dir;
    struct dirent *dir_entry;
    struct stat dir_stat;

    // Open the directory (if possible). If not, return
    if ((dir = opendir(directory)) == NULL) {
        fprintf(stderr, "Error: Could not open directory %s\n", directory);
        return;
    }

    // Iterate over the directory entries
    while ((dir_entry = readdir(dir)) != NULL) {
        char full_path[MAX_PATH_SIZE];
        snprintf(full_path, sizeof(full_path), "%s/%s", directory, dir_entry->d_name);

        if (stat(full_path, &dir_stat) == 0 && S_ISDIR(dir_stat.st_mode)) {
            // Ignore the current and parent directories
            if (strcmp(dir_entry->d_name, ".") != 0 && strcmp(dir_entry->d_name, "..") != 0) {
                // Process the subdirectory
                process_directory(full_path);
            }
        } else {
            // Process the file
            process_file(full_path);
        }
    }

    closedir(dir);
}

void process_file(const string file) {
    // Open the file
    FILE *fp;
    if ((fp = fopen(file, "r")) == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", file);
        return;
    }

    char header[MAX_FASTA_HEADER_LENGTH];
    char line[MAX_FASTA_DNA_SEQUENCE_LENGTH];

    // Ignore the first line
    fgets(header, sizeof(header), fp);

    // Read the remaining lines
    while (fgets(line, sizeof(line), fp) != NULL) {
        // If it ends with a newline character, remove it
        const size_t line_length = strlen(line);
        if (line[line_length - 1] == '\n') {
            line[line_length - 1] = '\0';
        }
    }

    // Close the file
    fclose(fp);
}

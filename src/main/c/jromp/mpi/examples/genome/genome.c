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

PARALLEL_FN int process_directory(PRIVATE const string directory, SHARED struct dna_sequence *dna_sequence) {
    DIR *dir;

    // Open the directory (if possible). If not, return
    if ((dir = opendir(directory)) == NULL) {
        fprintf(stderr, "Error: Could not open directory %s\n", directory);
        return -1;
    }

    int num_files = 0;
    struct dirent *dir_entry;
    struct stat dir_stat;

    // Iterate over the directory entries
    while ((dir_entry = readdir(dir)) != NULL) {
        char full_path[MAX_PATH_SIZE];
        snprintf(full_path, sizeof(full_path), "%s/%s", directory, dir_entry->d_name);

        if (stat(full_path, &dir_stat) == 0 && S_ISDIR(dir_stat.st_mode)) {
            // Ignore the current and parent directories
            if (strcmp(dir_entry->d_name, ".") != 0 && strcmp(dir_entry->d_name, "..") != 0) {
                // Process the subdirectory
                const int dir_files = process_directory(full_path, dna_sequence);

                // If an error occurred, return
                if (dir_files == -1) {
                    return -1;
                }

                // Increment the counter with the number of files in the subdirectory
                num_files += dir_files;
            }
        } else {
            // Process the file
            process_file(full_path, SHARED dna_sequence);
            num_files++;
        }
    }

    closedir(dir);
    return num_files;
}

PARALLEL_FN void process_file(const string file, SHARED struct dna_sequence *dna_sequence) {
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
        const size_t line_length = strlen(line);

        // Iterate over the line
        for (size_t i = 0; i < line_length; i++) {
            const char nucleotide = line[i];

#pragma omp critical(nucleotide_update)
            switch (nucleotide) {
                case 'A':
                    dna_sequence->A++;
                    break;
                case 'C':
                    dna_sequence->C++;
                    break;
                case 'G':
                    dna_sequence->G++;
                    break;
                case 'T':
                    dna_sequence->T++;
                    break;
                case 'U':
                    dna_sequence->U++;
                    break;
                case 'N':
                    dna_sequence->N++;
                    break;
                default:
                    break;
            }
        }
    }

    // Close the file
    fclose(fp);
}

void pretty_print_dna_sequence(const struct dna_sequence *dna_sequence) {
    printf("\tAdenine (A): %ld\n"
           "\tCytosine (C): %ld\n"
           "\tGuanine (G): %ld\n"
           "\tThymine (T): %ld\n"
           "\tUracil (U): %ld\n"
           "\tNucleic acid (N): %ld\n",
           dna_sequence->A, dna_sequence->C, dna_sequence->G, dna_sequence->T, dna_sequence->U, dna_sequence->N);
}

#ifndef GENOME_H
#define GENOME_H

#include <dirent.h>
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#define CVECTOR_LOGARITHMIC_GROWTH
#include "cvector.h"

int rank;
int size;
omp_lock_t print_lock;

#define MAX_PATH_SIZE 1024
#define MAX_FASTA_HEADER_LENGTH 1024
#define MAX_FASTA_DNA_SEQUENCE_LENGTH 80
#define string char *
#define DEBUG_LOGGING 1

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#define PARALLEL_FN
#define SHARED
#define PRIVATE

#if DEBUG_LOGGING == 1
    /**
     * Prints the given message if the current rank is 0 (Master process).
     */
    #define LOG_MASTER(...)                                                                                            \
        if (rank == 0) {                                                                                               \
            printf("       Master: " __VA_ARGS__);                                                                     \
        }

    /**
     * Prints the given message if the current rank is not 0 (Worker process).
     * In addition, if the context is parallel, the thread number is also printed synchronously.
     */
    #define LOG_WORKER(...)                                                                                            \
        if (rank != 0) {                                                                                               \
            if (omp_get_num_threads() == 1) {                                                                          \
                printf("   Worker %03d: ", rank);                                                                      \
                printf(__VA_ARGS__);                                                                                   \
            } else {                                                                                                   \
                omp_set_lock(&print_lock);                                                                             \
                printf("Worker %03d-%02d: ", rank, omp_get_thread_num());                                              \
                printf(__VA_ARGS__);                                                                                   \
                omp_unset_lock(&print_lock);                                                                           \
            }                                                                                                          \
        }
#else
    #define LOG_MASTER(...)
    #define LOG_WORKER(...)
#endif

/**
 * Represents a DNA sequence. It contains the number of occurrences of each nucleotide:
 * - A: Adenine.
 * - C: Cytosine.
 * - G: Guanine.
 * - T: Thymine.
 * - U: Uracil.
 * - N: Nucleic acid (any nucleotide).
 */
struct dna_sequence {
    /** Adenine (A) nucleotide count. */
    ssize_t A;
    /** Cytosine (C) nucleotide count. */
    ssize_t C;
    /** Guanine (G) nucleotide count. */
    ssize_t G;
    /** Thymine (T) nucleotide count. */
    ssize_t T;
    /** Uracil (U) nucleotide count. */
    ssize_t U;
    /** Nucleic acid (N) nucleotide count. */
    ssize_t N;
} __attribute__((aligned(64)));

int get_dirs(const string directory_path, cvector(string) * directories);

PARALLEL_FN int process_directory(PRIVATE const string directory, SHARED struct dna_sequence *dna_sequence);

void process_file(const string file, SHARED struct dna_sequence *dna_sequence);

void pretty_print_dna_sequence(SHARED const struct dna_sequence *dna_sequence);

#endif // GENOME_H

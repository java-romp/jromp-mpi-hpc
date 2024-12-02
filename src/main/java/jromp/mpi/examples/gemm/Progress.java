package jromp.mpi.examples.gemm;

record Progress(int rank, int rowsProcessed, double progress) {
    Progress {
        if (rank < 0) {
            throw new IllegalArgumentException("Rank must be non-negative");
        }

        if (rowsProcessed < 0) {
            throw new IllegalArgumentException("Rows processed must be non-negative");
        }

        if (progress < 0) {
            throw new IllegalArgumentException("Progress must be non-negative");
        }
    }
}

package jromp.mpi.examples.gemm;

final class Tags {
    private Tags() {
        // Prevent instantiation
    }

    public static final int DATA_TAG = 0;
    public static final int FINISH_TAG = 1;
}

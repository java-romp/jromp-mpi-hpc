package jromp.mpi.examples.gemm;

import jromp.JROMP;

class Utils {
    private static final Object printLock = new Object();

    private Utils() {
        throw new IllegalStateException("Utility class");
    }

    public static void LOG_MASTER(int rank, String format, Object... args) {
        if (rank == 0) {
            System.out.printf("      Master: %s", String.format(format, args));
        }
    }

    public static void LOG_WORKER(int rank, String format, Object... args) {
        if (rank != 0) {
            if (JROMP.getNumThreads() == 1) {
                System.out.printf("   Worker %02d: %s", rank, String.format(format, args));
            } else {
                synchronized (printLock) {
                    System.out.printf("Worker %02d-%02d: %s", rank, JROMP.getThreadNum(), String.format(format, args));
                }
            }
        }
    }
}

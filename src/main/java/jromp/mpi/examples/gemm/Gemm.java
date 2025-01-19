package jromp.mpi.examples.gemm;

import jromp.JROMP;
import jromp.var.SharedVariable;
import jromp.var.Variable;
import mpi.MPI;
import mpi.MPIException;
import mpi.Status;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Objects;

@SuppressWarnings({ "java:S100", "java:S106", "java:S117", "java:S1192", "java:S1450", "java:S1659", "java:S3008" })
public class Gemm {
    // Tags for MPI messages
    public static final int DATA_TAG = 0;
    public static final int FINISH_TAG = 1;
    public static final int START_MULTIPLICATION_TAG = 2;

    // Constants/Fields used in program
    private static final SecureRandom random = new SecureRandom();
    private static final Object printLock = new Object();
    private static int workers;
    private static int N;
    private static int threads;
    private static int rank;
    private static int size;
    private static int chunkSize;

    // Global constants
    private static final int MASTER_RANK = 0;
    private static final int EXIT_FAILURE = 1;
    private static final int BATCH_ROWS = 4096;

    public static void main(String[] args) throws MPIException {
        if (args.length != 2) {
            printf("Usage: mpirun ... java ... jromp.mpi.examples.gemm.Gemm <N> <threads>\n");
            System.exit(EXIT_FAILURE);
        }

        int provided;
        final int required = MPI.THREAD_MULTIPLE;

        provided = MPI.InitThread(args, required);

        if (provided < required) {
            System.err.print("Error: MPI does not provide the required thread support\n");
            MPI.COMM_WORLD.abort(EXIT_FAILURE);
            System.exit(EXIT_FAILURE);
        }

        rank = MPI.COMM_WORLD.getRank();
        size = MPI.COMM_WORLD.getSize();

        setRandomSeedSecure(rank);

        // Initialize globals
        workers = size - 1;
        N = Integer.parseInt(args[0]);
        threads = Integer.parseInt(args[1]);
        chunkSize = BATCH_ROWS * N;

        LOG_MASTER("Information: N = %d, threads per rank = %d\n", N, threads);
        LOG_MASTER("Checking the number of threads in all ranks...\n");

        MPI.COMM_WORLD.barrier();

        // Check that all ranks have the same number of threads
        JROMP.withThreads(threads)
             .masked(() -> printf("Number of threads of rank %d: %d\n", rank, JROMP.getNumThreads()))
             .join();

        MPI.COMM_WORLD.barrier();

        final int rowsPerWorker = N / workers; // Exclude the master process
        double[] A;
        double[] B;
        double[] C;

        if (rank == MASTER_RANK) {
            // Only master process allocates memory for all complete matrices
            A = new double[N * N];
            B = new double[N * N];
            C = new double[N * N];

            // No memory allocation check in Java

            LOG_MASTER("*************************************\n");
            LOG_MASTER("******* Matrix Initialization *******\n");
            LOG_MASTER("*************************************\n");

            double initializationStart = JROMP.getWTime();

            matrixInitialization(A, B, N);

            double initializationEnd = JROMP.getWTime();
            LOG_MASTER("Time to initialize the matrices: %fs\n", initializationEnd - initializationStart);

            LOG_MASTER("*************************************\n");
            LOG_MASTER("****** Sending data to workers ******\n");
            LOG_MASTER("*************************************\n");
            double sendDataStart = MPI.wtime();

            // Distribute rows of A to workers and send matrix B to all workers
            for (int i = 1; i < size; i++) {
                double[] offsetA = Arrays.copyOfRange(A, (i - 1) * rowsPerWorker * N, i * rowsPerWorker * N);

                sendBatchedArray(offsetA, i, rowsPerWorker, chunkSize, DATA_TAG);
                sendBatchedArray(B, i, N, chunkSize, DATA_TAG);
            }

            double sendDataEnd = MPI.wtime();
            LOG_MASTER("Time to send data to workers: %fs\n", sendDataEnd - sendDataStart);

            // Send a message to workers (without body) to indicate the start of the calculations
            for (int i = 1; i < size; i++) {
                MPI.COMM_WORLD.send(new byte[0], 0, MPI.BYTE, i, START_MULTIPLICATION_TAG);
            }

            LOG_MASTER("*************************************\n");
            LOG_MASTER("******* Matrix Multiplication *******\n");
            LOG_MASTER("*************************************\n");
            double calculationsStart = MPI.wtime();
            int endedWorkers = 0;
            Status status;

            do {
                status = MPI.COMM_WORLD.probe(MPI.ANY_SOURCE, MPI.ANY_TAG);

                if (status.getTag() == FINISH_TAG) {
                    // When a finish message is received, I can receive all the following messages of the
                    // same worker. Then, the probe call will return a new message from another worker.
                    double[] offsetC = new double[rowsPerWorker * N];
                    recvBatchedArray(offsetC, status.getSource(), rowsPerWorker, chunkSize, FINISH_TAG);

                    System.arraycopy(offsetC, 0, C, offsetFromRank(status.getSource(), rowsPerWorker),
                                     rowsPerWorker * N);

                    LOG_MASTER("Worker %d has finished\n", status.getSource());
                    endedWorkers++;
                } else {
                    // Unexpected message
                    MPI.COMM_WORLD.abort(EXIT_FAILURE);
                    LOG_MASTER("Unexpected message\n");

                    System.exit(EXIT_FAILURE);
                }
            } while (endedWorkers < workers);
            double calculationsEnd = MPI.wtime();
            double calculationsTimer = calculationsEnd - calculationsStart;
            LOG_MASTER("Time to do the calculations: %f\n", calculationsTimer);

            LOG_MASTER("Writing execution configuration to file\n");
            writeExecutionConfigurationToFile(N, workers, threads, calculationsTimer);
        } else {
            // Workers allocate memory for their part of the matrices
            A = new double[rowsPerWorker * N];
            B = new double[N * N]; // All workers need the matrix B
            C = new double[rowsPerWorker * N];

            // No memory allocation check in Java

            // Receive rows of A and matrix B
            recvBatchedArray(A, MASTER_RANK, rowsPerWorker, chunkSize, DATA_TAG);
            recvBatchedArray(B, MASTER_RANK, N, chunkSize, DATA_TAG);

            // Wait for the start message
            MPI.COMM_WORLD.recv(new byte[0], 0, MPI.BYTE, MASTER_RANK, START_MULTIPLICATION_TAG);
            LOG_WORKER("Worker %d has started\n", rank);

            // Perform matrix multiplication
            gemm(A, B, C, rowsPerWorker);
            // C is filled during multiplication

            // Send results back to master process
            sendBatchedArray(C, MASTER_RANK, rowsPerWorker, chunkSize, FINISH_TAG);
        }

        MPI.Finalize();
    }

    private static void gemm(final double[] a, final double[] b, double[] c, final int rowsPerWorker) {
        Objects.requireNonNull(a);
        Objects.requireNonNull(b);
        Objects.requireNonNull(c);

        Variable<double[]> vA = new SharedVariable<>(a);
        Variable<double[]> vB = new SharedVariable<>(b);
        Variable<double[]> vC = new SharedVariable<>(c);
        Variable<Integer> vRowsPerWorker = new SharedVariable<>(rowsPerWorker);

        JROMP.withThreads(threads)
             .registerVariables(vA, vB, vC, vRowsPerWorker)
             .parallelFor(0, rowsPerWorker, (start, end) -> {
                 double localSum;
                 int i, j, k;
                 double[] A = vA.value();
                 double[] B = vB.value();
                 double[] C = vC.value();

                 for (i = start; i < end; i++) {
                     for (j = 0; j < N; j++) {
                         localSum = 0;

                         for (k = 0; k < N; k++) {
                             localSum += A[i * N + k] * B[k * N + j];
                         }

                         C[i * N + j] = localSum;
                     }
                 }
             })
             .join();
    }

    private static void writeExecutionConfigurationToFile(final int n, final int workers,
                                                          final int threads, final double time) {
        File file = new File("execution_configs_java.csv");

        try {
            // If the file does not exist, create it and write the header
            if (file.createNewFile()) {
                try (FileWriter writer = new FileWriter(file, false)) { // Disable append mode
                    writer.write("n,workers,threads,total_cpus,time" + System.lineSeparator());
                }
            }

            // Add the content to the end of the file
            try (FileWriter writer = new FileWriter(file, true)) { // Enable append mode
                writer.write(String.format("%d,%d,%d,%d,%f%n", n, workers, threads, workers * threads, time));
            }
        } catch (IOException e) {
            e.printStackTrace(System.err);
        }
    }

    private static void matrixInitialization(double[] a, double[] b, final int n) {
        Objects.requireNonNull(a);
        Objects.requireNonNull(b);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = randomInRange(1, 1000);
                b[i * n + j] = randomInRange(1, 1000);
            }
        }
    }

    /**
     * Generates a random integer in the range [min, max].
     *
     * @param min The minimum value.
     * @param max The maximum value.
     *
     * @return The random integer.
     */
    private static int randomInRange(final int min, final int max) {
        return random.nextInt(min, max + 1);
    }

    /**
     * Sets the random seed securely.
     * The seed is generated using the current time and the rank of the process.
     * <p>
     * NOTE: <a href="https://wiki.sei.cmu.edu/confluence/display/c/MSC32-C.+Properly+seed+pseudorandom+number+generators">MSC32-C</a>
     *
     * @param rank The rank of the process.
     */
    private static void setRandomSeedSecure(final int rank) {
        random.setSeed(System.nanoTime() ^ (System.currentTimeMillis() / 1000) ^ rank);
    }

    public static void printf(String format, Object... args) {
        System.out.printf(format, args);
    }

    public static void LOG_MASTER(String format, Object... args) {
        if (rank == 0) {
            printf("      Master: %s", String.format(format, args));
        }
    }

    public static void LOG_WORKER(String format, Object... args) {
        if (rank != 0) {
            if (JROMP.getNumThreads() == 1) {
                printf("   Worker %02d: %s", rank, String.format(format, args));
            } else {
                synchronized (printLock) {
                    printf("Worker %02d-%02d: %s", rank, JROMP.getThreadNum(), String.format(format, args));
                }
            }
        }
    }

    private static void sendBatchedArray(double[] arr, int toRank, int rowsPerWorker, int chunkSize, int tag) throws MPIException {
        for (int offset = 0; offset < rowsPerWorker * N; offset += chunkSize) {
            int sendSize = Math.min(chunkSize, rowsPerWorker * N - offset);
            double[] offsetArray = Arrays.copyOfRange(arr, offset, offset + sendSize);

            MPI.COMM_WORLD.send(offsetArray, sendSize, MPI.DOUBLE, toRank, tag);
        }
    }

    private static void recvBatchedArray(double[] arr, int fromRank, int rowsPerWorker, int chunkSize, int tag) throws MPIException {
        for (int offset = 0; offset < rowsPerWorker * N; offset += chunkSize) {
            int recvSize = Math.min(chunkSize, rowsPerWorker * N - offset);
            double[] local = Arrays.copyOfRange(arr, offset, offset + recvSize);

            MPI.COMM_WORLD.recv(local, recvSize, MPI.DOUBLE, fromRank, tag);

            System.arraycopy(local, 0, arr, offset, recvSize);
        }
    }

    private static int offsetFromRank(int rank, int rowsPerWorker) {
        return (rank - 1) * rowsPerWorker * N;
    }
}

// Last revision (scastd): 19/01/2025 13:57

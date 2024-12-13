package jromp.mpi.examples.gemm;

import jromp.JROMP;
import jromp.var.SharedVariable;
import jromp.var.Variable;
import mpi.MPI;
import mpi.MPIException;
import mpi.Request;
import mpi.Status;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.security.SecureRandom;
import java.util.Objects;

import static jromp.JROMP.getNumThreads;
import static jromp.JROMP.getThreadNum;
import static jromp.JROMP.getWTime;
import static jromp.mpi.examples.gemm.Tags.DATA_TAG;
import static jromp.mpi.examples.gemm.Tags.FINISH_TAG;
import static jromp.mpi.examples.gemm.Tags.PROGRESS_TAG;

@SuppressWarnings({
        "java:S100", // Rename this method name to match the regular expression '^[a-z][a-zA-Z0-9]*$'.
        "java:S106", // Replace the use of System.out by a logger
        "java:S117", // Local variable should match the regular expression '^[a-z][a-zA-Z0-9]*$'
        "java:S1192", // String literals should not be duplicated
        "java:S1450", // Field can be converted to a local variable
        "java:S1659", // Declare variables on separate lines
        "java:S3008" // Field name should match the regular expression '^[a-z][a-zA-Z0-9]*$'
})
public class Gemm {
    private static final SecureRandom random = new SecureRandom();
    private static final Object printLock = new Object();
    private static int workers;
    private static int N;
    private static int threads;
    private static int rank;
    private static int size;

    private static final int MASTER_RANK = 0;
    private static final int EXIT_SUCCESS = 0;
    private static final int EXIT_FAILURE = 1;

    public static void main(String[] args) throws MPIException {
        if (args.length != 2) {
            printf("Usage: mpirun ... java ... jromp.mpi.examples.gemm.Gemm <N> <threads>\n");
            System.exit(EXIT_FAILURE);
        }

        int provided;
        final int required = MPI.THREAD_MULTIPLE;

        provided = MPI.InitThread(args, required);

        if (provided < required) {
            printf("Error: MPI does not provide the required thread support\n");
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

        LOG_MASTER("Information: N = %d, threads per rank = %d\n", N, threads);
        LOG_MASTER("Checking the number of threads in all ranks...\n");

        MPI.COMM_WORLD.barrier();

        // Check that all ranks have the same number of threads
        JROMP.withThreads(threads)
             .masked(() -> printf("Number of threads of rank %d: %d\n", rank, getNumThreads()))
             .join();

        MPI.COMM_WORLD.barrier();

        int rowsPerWorker = N / workers; // Exclude the master process
        double[] A;
        double[] B;
        double[] C;

        if (rank == MASTER_RANK) {
            // Only master process allocates memory for all complete matrices
            A = new double[N * N];
            B = new double[N * N];
            C = new double[N * N]; // All matrices are initialized to zero (no calloc)

            // No memory allocation check in Java

            LOG_MASTER("*************************************\n");
            LOG_MASTER("******* Matrix Initialization *******\n");
            LOG_MASTER("*************************************\n");

            double initializationStart = getWTime();

            // Initialize matrices
            matrixInitialization(A, B, N);

            double initializationEnd = getWTime();
            LOG_MASTER("Time to initialize the matrices: %fs\n", initializationEnd - initializationStart);

            LOG_MASTER("*************************************\n");
            LOG_MASTER("****** Sending data to workers ******\n");
            LOG_MASTER("*************************************\n");

            double sendDataStart = getWTime();

            Request[] requests = new Request[2 * workers];

            DoubleBuffer bufferA = MPI.newDoubleBuffer(N * N).put(A);
            DoubleBuffer bufferB = MPI.newDoubleBuffer(N * N).put(B);

            // Distribute rows of A to workers and send matrix B to all workers
            for (int i = 1; i < size; i++) {
                DoubleBuffer aSubMatrix = bufferA.slice((i - 1) * rowsPerWorker * N, rowsPerWorker * N);

                requests[i - 1] = MPI.COMM_WORLD.iSend(aSubMatrix, rowsPerWorker * N, MPI.DOUBLE, i, DATA_TAG);
                requests[workers + i - 1] = MPI.COMM_WORLD.iSend(bufferB, N * N, MPI.DOUBLE, i, DATA_TAG);
            }

            Request.waitAll(requests);

            double sendDataEnd = getWTime();
            LOG_MASTER("Time to send data to workers: %fs\n", sendDataEnd - sendDataStart);

            LOG_MASTER("*************************************\n");
            LOG_MASTER("******* Matrix Multiplication *******\n");
            LOG_MASTER("*************************************\n");

            double calculationsStart = MPI.wtime();

            int endedWorkers = 0;
            double timeEnd;
            Status status;
            Progress progress = new Progress();
            Progress globalProgress = new Progress();
            ByteBuffer progressBuffer = MPI.newByteBuffer(Progress.size());

            do {
                status = MPI.COMM_WORLD.probe(MPI.ANY_SOURCE, MPI.ANY_TAG);

                switch (status.getTag()) {
                    case FINISH_TAG -> {
                        DoubleBuffer recvBuffer = MPI.newDoubleBuffer(rowsPerWorker * N);
                        MPI.COMM_WORLD.recv(recvBuffer, rowsPerWorker * N, MPI.DOUBLE, status.getSource(), FINISH_TAG);

                        recvBuffer.get(C, (status.getSource() - 1) * rowsPerWorker * N, rowsPerWorker * N);
                        recvBuffer.clear();

                        LOG_MASTER("Worker %d has finished\n", status.getSource());
                        endedWorkers++;
                    }
                    case PROGRESS_TAG -> {
                        MPI.COMM_WORLD.recv(progressBuffer, Progress.size(), MPI.BYTE, status.getSource(),
                                            PROGRESS_TAG);
                        timeEnd = MPI.wtime();

                        progress.readBuffer(progressBuffer);
                        progressBuffer.clear();

                        globalProgress.incrementRowsProcessed();
                        globalProgress.progress((float) globalProgress.rowsProcessed() / N * 100.0f);

                        // Notation:
                        //  - T_r: Time to process a row.
                        //  - T_t: Time total (from the beginning of the calculations).
                        //  - ETF: Estimated time to finish the calculations.
                        LOG_MASTER(
                                "Progress of worker %d (Thread %d): %f%% (%d/%d) (overall: %f%% (%d/%d))  ::  T_r: %.5fs   T_t: %.5fs   ETF: %.5fs\n",
                                progress.rank(), progress.thread(), progress.progress(), progress.rowsProcessed(),
                                rowsPerWorker / threads, globalProgress.progress(), globalProgress.rowsProcessed(), N,
                                progress.rowTime(), timeEnd - calculationsStart,
                                etf(calculationsStart, globalProgress.progress()));
                    }
                    default -> {
                        // Unexpected message
                        MPI.COMM_WORLD.abort(EXIT_FAILURE);
                        LOG_MASTER("Unexpected message\n");

                        System.exit(EXIT_FAILURE);
                    }
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

            DoubleBuffer bufferA = DoubleBuffer.wrap(A);
            DoubleBuffer bufferB = DoubleBuffer.wrap(B);
            DoubleBuffer bufferC = DoubleBuffer.wrap(C);

            // Receive rows of A and matrix B
            MPI.COMM_WORLD.recv(bufferA, rowsPerWorker * N, MPI.DOUBLE, MASTER_RANK, DATA_TAG);
            MPI.COMM_WORLD.recv(bufferB, N * N, MPI.DOUBLE, MASTER_RANK, DATA_TAG);

            // Perform matrix multiplication
            matrixMultiplication(A, B, C, rowsPerWorker, rank);
            // bufferC is filled when the matrix multiplication is done

            // Send results back to master process
            MPI.COMM_WORLD.send(bufferC, rowsPerWorker * N, MPI.DOUBLE, MASTER_RANK, FINISH_TAG);

            // Free memory
            bufferA.clear();
            bufferB.clear();
            bufferC.clear();
        }

        MPI.Finalize();
    }

    private static void matrixMultiplication(double[] a, double[] b, double[] c, int rowsPerWorker, int rank) {
        Objects.requireNonNull(a);
        Objects.requireNonNull(b);
        Objects.requireNonNull(c);

        Variable<double[]> vA = new SharedVariable<>(a);
        Variable<double[]> vB = new SharedVariable<>(b);
        Variable<double[]> vC = new SharedVariable<>(c);
        Variable<Integer> vRowsPerWorker = new SharedVariable<>(rowsPerWorker);
        Variable<Integer> vRank = new SharedVariable<>(rank);
        Variable<Integer> rowsPerThread = new SharedVariable<>(rowsPerWorker / threads);
        Variable<Progress[]> progresses = new SharedVariable<>(new Progress[threads]);

        JROMP.withThreads(threads)
             .registerVariables(vA, vB, vC, vRowsPerWorker, vRank, rowsPerThread, progresses)
             .parallelFor(0, rowsPerWorker, (start, end) -> {
                 double localSum;
                 int i, j, k;
                 final int threadNum = getThreadNum(); // Prevent multiple calls to the function inside the for loop
                 ByteBuffer progressBuffer = MPI.newByteBuffer(Progress.size());
                 Progress threadProgress = new Progress(rank, 0, threadNum, 0.0f, 0.0);
                 progresses.value()[threadNum] = threadProgress;
                 double[] A = vA.value();
                 double[] B = vB.value();
                 double[] C = vC.value();

                 for (i = start; i < end; i++) {
                     threadProgress.rowTime(getWTime());

                     for (j = 0; j < N; j++) {
                         localSum = 0;

                         for (k = 0; k < N; k++) {
                             localSum += A[i * N + k] * B[k * N + j];
                         }

                         C[i * N + j] = localSum;
                     }

                     threadProgress.rowTime(getWTime() - threadProgress.rowTime());
                     threadProgress.incrementRowsProcessed();
                     threadProgress.progress((float) threadProgress.rowsProcessed() / rowsPerThread.value() * 100.0f);
                     threadProgress.fillBuffer(progressBuffer);

                     try {
                         /*
                          * Send asynchronous progress to master rank to avoid blocking.
                          * Waiting for the request to complete is not necessary because we don't want to know if the
                          * master process has received the progress (it is only informative). If it is lost, the master
                          * process will not have the progress of the worker, but the calculations will continue.
                          */
                         MPI.COMM_WORLD.iSend(progressBuffer, Progress.size(), MPI.BYTE, MASTER_RANK, PROGRESS_TAG);
                     } catch (MPIException e) {
                         e.printStackTrace(System.err);
                     }
                 }
             })
             .join();
    }

    private static void writeExecutionConfigurationToFile(int n, int workers, int threads, double time) {
        File file = new File("execution_configs_java.csv");

        try {
            // If the file does not exist, create it and write the header
            if (file.createNewFile()) {
                try (FileWriter writer = new FileWriter(file, false)) { // Disable append mode
                    writer.write("n,workers,threads,total_cpus,time" + System.lineSeparator());
                }
            }

            // Add the content to the end of the file
            try (FileWriter writer = new FileWriter(file, true)) { // Append mode
                writer.write(String.format("%d,%d,%d,%d,%f%n", n, workers, threads, workers * threads, time));
            }
        } catch (IOException e) {
            e.printStackTrace(System.err);
        }
    }

    private static void matrixInitialization(double[] a, double[] b, int n) {
        Objects.requireNonNull(a);
        Objects.requireNonNull(b);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = random.nextInt(1, 1000);
                b[i * n + j] = random.nextInt(1, 1000);
            }
        }
    }

    private static double etf(double startTime, double progress) throws MPIException {
        return (100.0 - progress) * (MPI.wtime() - startTime) / progress;
    }

    private static void setRandomSeedSecure(int rank) {
        random.setSeed(System.nanoTime() ^ System.currentTimeMillis() / 1000 ^ rank);
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
            if (getNumThreads() == 1) {
                printf("   Worker %02d: %s", rank, String.format(format, args));
            } else {
                synchronized (printLock) {
                    printf("Worker %02d-%02d: %s", rank, getThreadNum(), String.format(format, args));
                }
            }
        }
    }
}

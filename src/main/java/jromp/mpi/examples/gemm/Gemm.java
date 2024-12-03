package jromp.mpi.examples.gemm;

import jromp.JROMP;
import mpi.Datatype;
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
import static jromp.mpi.examples.gemm.Utils.LOG_MASTER;
import static jromp.mpi.examples.gemm.Utils.LOG_WORKER;

@SuppressWarnings("all")
public class Gemm {
    private static final SecureRandom random = new SecureRandom();
    private static Datatype progressType;
    private static int workers;
    private static int N;
    private static int threads;
    private static final int MASTER_RANK = 0;

    public static void main(String[] args) throws MPIException {
        MPI.Init(args);

        if (args.length != 2) {
            if (MPI.COMM_WORLD.getRank() == 0) {
                System.out.println("Usage: mpirun ... java ... jromp.mpi.examples.gemm.Gemm <N> <threads>");
            }

            MPI.COMM_WORLD.abort(1);
            return;
        }

        int rank = MPI.COMM_WORLD.getRank();
        int size = MPI.COMM_WORLD.getSize();

        progressType = createProgressType();
        progressType.commit();
        workers = size - 1;
        N = Integer.parseInt(args[0]);
        threads = Integer.parseInt(args[1]);

        LOG_MASTER(rank, "Information: N = %d, threads = %d\n", N, threads);

        JROMP.withThreads(threads)
             .parallel(() -> LOG_WORKER(rank, "I am the thread %d\n", getThreadNum()))
             .join();

        int rowsPerWorker = N / workers; // Exclude the master process
        double[] A;
        double[] B;
        double[] C = null;

        if (rank == MASTER_RANK) {
            // Only master process allocates memory for all complete matrices
            A = new double[N * N];
            B = new double[N * N];
            C = new double[N * N];

            LOG_MASTER(rank, "*************************************\n");
            LOG_MASTER(rank, "******* Matrix Initialization *******\n");
            LOG_MASTER(rank, "*************************************\n");

            double initializationStart = getWTime();

            // Initialize matrices
            matrixInitialization(A, B, N);

            double initializationEnd = getWTime();
            LOG_MASTER(rank, "Time to initialize the matrices: %fs\n", initializationEnd - initializationStart);

            double calculationsStart = MPI.wtime();

            LOG_MASTER(rank, "*************************************\n");
            LOG_MASTER(rank, "******* Matrix Multiplication *******\n");
            LOG_MASTER(rank, "*************************************\n");

            Request[] requests = new Request[2 * workers];

            DoubleBuffer bufferA = MPI.newDoubleBuffer(N * N).put(A);
            DoubleBuffer bufferB = MPI.newDoubleBuffer(N * N).put(B);

            // Send rows of A to workers and matrix B to all workers
            for (int i = 1; i < size; i++) {
                DoubleBuffer aSubMatrix = bufferA.slice((i - 1) * rowsPerWorker * N, rowsPerWorker * N);

                requests[i - 1] = MPI.COMM_WORLD.iSend(aSubMatrix, rowsPerWorker * N, MPI.DOUBLE, i, DATA_TAG);
                requests[workers + i - 1] = MPI.COMM_WORLD.iSend(bufferB, N * N, MPI.DOUBLE, i, DATA_TAG);
            }

            Request.waitAll(requests);
            requests = null; // Free memory

            int endedWorkers = 0;
            Status status;
            Progress progress = new Progress(0, 0, 0.0);
            double rowTimeStart = calculationsStart;
            double rowTimeEnd;
            ByteBuffer progressBuffer = MPI.newByteBuffer(Progress.size());

            do {
                status = MPI.COMM_WORLD.probe(MPI.ANY_SOURCE, MPI.ANY_TAG);

                if (status.getTag() == FINISH_TAG) {
                    DoubleBuffer recvBuffer = MPI.newDoubleBuffer(rowsPerWorker * N);
                    MPI.COMM_WORLD.recv(recvBuffer, rowsPerWorker * N, MPI.DOUBLE, status.getSource(), FINISH_TAG);

                    recvBuffer.get(C, (status.getSource() - 1) * rowsPerWorker * N, rowsPerWorker * N);

                    LOG_MASTER(rank, "Worker %d has finished\n", status.getSource());
                    endedWorkers++;
                } else if (status.getTag() == PROGRESS_TAG) {
                    progressBuffer.clear();
                    MPI.COMM_WORLD.recv(progressBuffer, Progress.size(), MPI.BYTE, status.getSource(), PROGRESS_TAG);
                    rowTimeEnd = MPI.wtime();

                    progress.readBuffer(progressBuffer);

                    LOG_MASTER(rank, "Progress of worker %d: %f%% (%d/%d)  ::  T_r: %.5fs   T_t: %.5fs   ETF: %.5fs\n",
                               progress.rank(), progress.progress(), progress.rowsProcessed(), rowsPerWorker,
                               rowTimeEnd - rowTimeStart, rowTimeEnd - calculationsStart,
                               etf(calculationsStart, progress.progress()));

                    rowTimeStart = rowTimeEnd;
                } else {
                    // Unexpected message
                    MPI.COMM_WORLD.abort(1);
                    System.out.println("Unexpected message");
                    System.exit(1);
                }
            } while (endedWorkers < workers);

            double calculationsEnd = MPI.wtime();
            double calculationsTimer = calculationsEnd - calculationsStart;
            LOG_MASTER(rank, "Total time to do the calculations: %f\n", calculationsTimer);

            LOG_MASTER(rank, "Writing execution configuration to file\n");
            writeExecutionConfigurationToFile(N, workers, threads, calculationsTimer);
        } else {
            LOG_WORKER(rank, "Number of threads: %d\n", getNumThreads());

            A = new double[rowsPerWorker * N];
            B = new double[N * N];
            C = new double[rowsPerWorker * N];
            DoubleBuffer bufferA = MPI.newDoubleBuffer(rowsPerWorker * N).wrap(A);
            DoubleBuffer bufferB = MPI.newDoubleBuffer(N * N).wrap(B);
            DoubleBuffer bufferC = MPI.newDoubleBuffer(rowsPerWorker * N).wrap(C);
            Progress progress = new Progress(rank, 0, 0.0);

            // Receive rows of A and matrix B
            MPI.COMM_WORLD.recv(bufferA, rowsPerWorker * N, MPI.DOUBLE, MASTER_RANK, DATA_TAG);
            MPI.COMM_WORLD.recv(bufferB, N * N, MPI.DOUBLE, MASTER_RANK, DATA_TAG);

            // Perform matrix multiplication
            matrixMultiplication(A, B, C, rowsPerWorker, progress);

            // Send results back to master process
            MPI.COMM_WORLD.send(bufferC, rowsPerWorker * N, MPI.DOUBLE, MASTER_RANK, FINISH_TAG);

            // Free memory
            bufferA = null;
            bufferB = null;
            bufferC = null;
        }

        MPI.Finalize();
    }

    private static void matrixMultiplication(double[] a, double[] b, double[] c, int n, Progress progress) throws MPIException {
        Objects.requireNonNull(a);
        Objects.requireNonNull(b);
        Objects.requireNonNull(c);

        ByteBuffer progressBuffer = MPI.newByteBuffer(progressType.getSize());
        double localSum = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < N; j++) {
                localSum = 0;

                for (int k = 0; k < N; k++) {
                    localSum += a[i * N + k] * b[k * N + j];
                }

                c[i * N + j] = localSum;
            }

            progress.progress((i + 1.0) / n * 100.0);
            progress.rowsProcessed(i + 1);
            // Send asynchronous progress to master process to avoid blocking. No wait for the request to complete
            // because it is not necessary to know if the master process has received the progress
            // (it is only informative).
            progress.fillBuffer(progressBuffer);
            MPI.COMM_WORLD.iSend(progressBuffer, progressType.getSize(), MPI.BYTE, MASTER_RANK, PROGRESS_TAG);
        }
    }

    private static void writeExecutionConfigurationToFile(int n, int workers, int threads, double time) {
        File file = new File("execution_configuration.csv");

        try {
            // If the file does not exist, creates it and writes the header
            if (!file.exists()) {
                file.createNewFile();
                try (FileWriter writer = new FileWriter(file, false)) { // Disable append mode
                    writer.write("n,workers,threads,total_cpus,time" + System.lineSeparator());
                }
            }

            // Adds the content to the end of the file
            try (FileWriter writer = new FileWriter(file, true)) { // Append mode
                writer.write(String.format("%d,%d,%d,%d,%f%n", n, workers, threads, workers * threads, time));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Datatype createProgressType() throws MPIException {
        Datatype[] types = { MPI.INT, MPI.INT, MPI.DOUBLE };
        int[] blockLengths = { 1, 1, 1 };
        int[] offsets = { 0, MPI.INT.getSize(), 2 * MPI.INT.getSize() };

        return Datatype.createStruct(blockLengths, offsets, types);
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
}

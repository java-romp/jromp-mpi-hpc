package jromp.mpi.examples.gemm;

import jromp.JROMP;
import mpi.Datatype;
import mpi.MPI;
import mpi.MPIException;
import mpi.Request;

import java.nio.DoubleBuffer;
import java.security.SecureRandom;
import java.util.Objects;

import static jromp.JROMP.getThreadNum;
import static jromp.JROMP.getWTime;
import static jromp.mpi.examples.gemm.Tags.DATA_TAG;
import static jromp.mpi.examples.gemm.Utils.LOG_MASTER;
import static jromp.mpi.examples.gemm.Utils.LOG_WORKER;

@SuppressWarnings("all")
public class Gemm {
    private static final SecureRandom random = new SecureRandom();
    private static int workers;
    private static int N;
    private static int threads;

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

        Datatype progressType = createProgressType();
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

        if (rank == 0) {
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
        } else {

        }

        MPI.Finalize();
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
}

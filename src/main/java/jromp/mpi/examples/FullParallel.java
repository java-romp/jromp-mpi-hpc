package jromp.mpi.examples;

import jromp.JROMP;
import jromp.operation.Operations;
import jromp.var.ReductionVariable;
import jromp.var.SharedVariable;
import jromp.var.reduction.ReductionOperations;
import mpi.MPI;
import mpi.MPIException;

import static jromp.mpi.examples.Utils.printf;

@SuppressWarnings("all")
public class FullParallel {
    private static final int N = 2000;

    public static void main(String[] args) throws MPIException {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.getRank();
        int size = MPI.COMM_WORLD.getSize();

        // Print the available number of threads
        JROMP.allThreads()
             .single(false, () -> {
                 printf("Number of threads: %d\n", JROMP.getNumThreads());
             })
             .join();

        double[] A = new double[N * N];
        double[] B = new double[N * N];
        double[] C = new double[N * N];

        if (rank == 0) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A[i * N + j] = 1.0;
                    B[i * N + j] = 1.0;
                }
            }
        }

        MPI.COMM_WORLD.barrier();
        double start_time = MPI.wtime();

        MPI.COMM_WORLD.bcast(A, N * N, MPI.DOUBLE, 0);
        MPI.COMM_WORLD.bcast(B, N * N, MPI.DOUBLE, 0);

        SharedVariable<double[]> a = new SharedVariable<>(A);
        SharedVariable<double[]> b = new SharedVariable<>(B);
        SharedVariable<double[]> c = new SharedVariable<>(C);
        ReductionVariable<Double> sum = new ReductionVariable<>(ReductionOperations.sum(), 0.0);

        JROMP.allThreads()
             .registerVariables(a, b, c, sum)
             .parallelFor(0, N, false, (start, end) -> {
                 double[] localA = a.value();
                 double[] localB = b.value();
                 double[] localC = c.value();

                 for (int i = start; i < end; i++) {
                     for (int j = 0; j < N; j++) {
                         localC[i * N + j] = 0.0;

                         for (int k = 0; k < N; k++) {
                             localC[i * N + j] += localA[i * N + k] * localB[k * N + j];
                         }
                     }
                 }
             })
             .parallelFor(0, N * N, false, (start, end) -> {
                 double[] cInternal = c.value();

                 for (int i = start; i < end; i++) {
                     sum.update(Operations.add(cInternal[i]));
                 }
             })
             .join();

        System.out.println(String.format("Total sum: %f", sum.value()));

        MPI.COMM_WORLD.barrier();
        double end_time = MPI.wtime();

        if (rank == 0) {
            printf("Time: %f\n", end_time - start_time);
        }

        MPI.Finalize();
    }
}

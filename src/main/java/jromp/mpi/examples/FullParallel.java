package jromp.mpi.examples;

import jromp.JROMP;
import jromp.var.SharedVariable;
import jromp.var.Variables;
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
             .single(false, vars -> {
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

        printf("Rank %d: Before bcast\n", rank);

        MPI.COMM_WORLD.barrier();
        double start_time = MPI.wtime();

        MPI.COMM_WORLD.bcast(A, N * N, MPI.DOUBLE, 0);
        MPI.COMM_WORLD.bcast(B, N * N, MPI.DOUBLE, 0);

        printf("Rank %d: After bcast\n", rank);

        Variables variables = Variables.create()
                                       .add("A", new SharedVariable<>(A))
                                       .add("B", new SharedVariable<>(B))
                                       .add("C", new SharedVariable<>(C));

        JROMP.allThreads()
             .withVariables(variables)
             .single(false, vars -> printf("Rank %d: Inside single\n", rank))
             .parallelFor(0, N, false, (start, end, vars) -> {
                 double[] localA = vars.<double[]>get("A").value();
                 double[] localB = vars.<double[]>get("B").value();
                 double[] localC = vars.<double[]>get("C").value();
                 int count = 0;

                 for (int i = start; i < end; i++) {
                     for (int j = 0; j < N; j++) {
                         localC[i * N + j] = 0.0;

                         for (int k = 0; k < N; k++) {
                             // localC[i * N + j] += localA[i * N + k] * localB[k * N + j];
                             count++;
                         }
                     }
                 }

                 printf("Rank %d: Count = %d\n", rank, count);
             })
             .single(false, vars -> printf("Rank %d: After parallel for\n", rank))
             .join();

        MPI.COMM_WORLD.barrier();
        double end_time = MPI.wtime();

        if (rank == 0) {
            printf("Time: %f\n", end_time - start_time);
        }

        MPI.Finalize();
    }
}

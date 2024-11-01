package jromp.mpi.examples;

import jromp.JROMP;
import mpi.MPI;
import mpi.MPIException;

public class JrompMPI {
    public static void main(String[] args) throws MPIException {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.getRank();
        int size = MPI.COMM_WORLD.getSize();

        JROMP.withThreads(4)
             .parallelFor(0, 16, false, (start, end, vars) -> {
                 for (int i = start; i < end; i++) {
                     System.out.println(String.format("Rank %d: i = %d (Thread %d)", rank, i, JROMP.getThreadNum()));
                 }
             })
             .join();

        MPI.Finalize();
    }
}

package jromp.mpi.examples;

import jromp.JROMP;

public class SimpleJROMP {
    public static void main(String[] args) {
        JROMP.withThreads(2)
             .parallel(vars -> System.out.println(
                     "Hello from thread " + JROMP.getThreadNum() + " of " + JROMP.getNumThreads()))
             .join();
    }
}

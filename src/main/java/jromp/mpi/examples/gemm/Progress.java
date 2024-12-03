package jromp.mpi.examples.gemm;

import java.nio.ByteBuffer;
import java.util.Objects;

final class Progress {
    private int rank;
    private int rowsProcessed;
    private double progress;

    Progress(int rank, int rowsProcessed, double progress) {
        if (rank < 0) {
            throw new IllegalArgumentException("Rank must be non-negative");
        }

        if (rowsProcessed < 0) {
            throw new IllegalArgumentException("Rows processed must be non-negative");
        }

        if (progress < 0) {
            throw new IllegalArgumentException("Progress must be non-negative");
        }

        this.rank = rank;
        this.rowsProcessed = rowsProcessed;
        this.progress = progress;
    }

    public int rank() {
        return rank;
    }

    public void rank(int rank) {
        this.rank = rank;
    }

    public int rowsProcessed() {
        return rowsProcessed;
    }

    public void rowsProcessed(int rowsProcessed) {
        this.rowsProcessed = rowsProcessed;
    }

    public double progress() {
        return progress;
    }

    public void progress(double progress) {
        this.progress = progress;
    }

    public void fillBuffer(ByteBuffer buffer) {
        buffer.clear()
              .putInt(rank)
              .putInt(rowsProcessed)
              .putDouble(progress)
              .rewind();
    }

    public void readBuffer(ByteBuffer buffer) {
        buffer.rewind();
        rank = buffer.getInt();
        rowsProcessed = buffer.getInt();
        progress = buffer.getDouble();
    }

    public static int size() {
        return Integer.BYTES + Integer.BYTES + Double.BYTES;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (obj == null || obj.getClass() != this.getClass()) return false;
        var that = (Progress) obj;
        return this.rank == that.rank &&
                this.rowsProcessed == that.rowsProcessed &&
                Double.doubleToLongBits(this.progress) == Double.doubleToLongBits(that.progress);
    }

    @Override
    public int hashCode() {
        return Objects.hash(rank, rowsProcessed, progress);
    }

    @Override
    public String toString() {
        return "Progress[" +
                "rank=" + rank + ", " +
                "rowsProcessed=" + rowsProcessed + ", " +
                "progress=" + progress + ']';
    }
}

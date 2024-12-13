package jromp.mpi.examples.gemm;

import java.nio.ByteBuffer;
import java.util.Objects;

final class Progress {
    private int rank;
    private int rowsProcessed;
    private int thread;
    private float progress;
    private double rowTime;

    Progress() {
        this(0, 0, (int) Thread.currentThread().threadId(), 0.0f, 0.0);
    }

    Progress(int rank, int rowsProcessed, int thread, float progress, double rowTime) {
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
        this.thread = thread;
        this.progress = progress;
        this.rowTime = rowTime;
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

    public int thread() {
        return thread;
    }

    public void thread(int thread) {
        this.thread = thread;
    }

    public double progress() {
        return progress;
    }

    public void progress(float progress) {
        this.progress = progress;
    }

    public double rowTime() {
        return rowTime;
    }

    public void rowTime(double rowTime) {
        this.rowTime = rowTime;
    }

    public void incrementRowsProcessed() {
        rowsProcessed++;
    }

    public void fillBuffer(ByteBuffer buffer) {
        buffer.clear()
              .putInt(rank)
              .putInt(rowsProcessed)
              .putInt(thread)
              .putFloat(progress)
              .putDouble(rowTime)
              .rewind();
    }

    public void readBuffer(ByteBuffer buffer) {
        buffer.rewind();
        rank = buffer.getInt();
        rowsProcessed = buffer.getInt();
        thread = buffer.getInt();
        progress = buffer.getFloat();
        rowTime = buffer.getDouble();
    }

    public static int size() {
        return 3 * Integer.BYTES + Float.BYTES + Double.BYTES;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Progress progress1)) return false;
        return rank == progress1.rank
                && rowsProcessed == progress1.rowsProcessed
                && thread == progress1.thread
                && Float.compare(progress, progress1.progress) == 0
                && Double.compare(progress1.rowTime, rowTime) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(rank, rowsProcessed, thread, progress, rowTime);
    }

    @Override
    public String toString() {
        return "Progress{" +
                "rank=" + rank +
                ", rowsProcessed=" + rowsProcessed +
                ", thread=" + thread +
                ", progress=" + progress +
                ", rowTime=" + rowTime +
                '}';
    }
}

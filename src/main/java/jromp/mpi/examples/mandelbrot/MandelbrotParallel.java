package jromp.mpi.examples.mandelbrot;

import jromp.JROMP;
import jromp.var.PrivateVariable;
import jromp.var.ReductionVariable;
import jromp.var.SharedVariable;
import jromp.var.Variable;
import jromp.var.Variables;
import jromp.var.reduction.ReductionOperations;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Date;

import static java.lang.Math.sqrt;
import static jromp.mpi.examples.mandelbrot.Mandelbrot.COUNT_MAX;
import static jromp.mpi.examples.mandelbrot.Mandelbrot.IMAGE_SIZE;
import static jromp.mpi.examples.mandelbrot.Mandelbrot.X_MAX;
import static jromp.mpi.examples.mandelbrot.Mandelbrot.X_MIN;
import static jromp.mpi.examples.mandelbrot.Mandelbrot.Y_MAX;
import static jromp.mpi.examples.mandelbrot.Mandelbrot.Y_MIN;

@SuppressWarnings("all") // Hide warnings in the IDE.
public class MandelbrotParallel {
    public static void main(String[] args) throws IOException {
        new MandelbrotParallel().$main(args);
    }

    public void $main(String[] args) throws IOException {
        int[] r;
        int[] g;
        int[] b;
        int c;
        int cMax;
        int[] count;
        final int countMax = COUNT_MAX;
        String filename = "mandelbrot_java_parallel.ppm";
        int i;
        int j;
        final int m = IMAGE_SIZE;
        final int n = IMAGE_SIZE;
        double x;
        double y;
        double[] tv1 = new double[1];
        double[] tv2 = new double[1];

        timestamp();

        System.out.printf("\n");
        System.out.printf("MANDELBROT\n");
        System.out.printf("  C version\n");
        System.out.printf("\n");
        System.out.printf("  Create an ASCII PPM image of the Mandelbrot set.\n");
        System.out.printf("\n");
        System.out.printf("  For each point C = X + i*Y\n");
        System.out.printf("  with X range [%f,%f]\n", X_MIN, X_MAX);
        System.out.printf("  and  Y range [%f,%f]\n", Y_MIN, Y_MAX);
        System.out.printf("  carry out %d iterations of the map\n", countMax);
        System.out.printf("  Z(n+1) = Z(n)^2 + C.\n");
        System.out.printf("  If the iterates stay bounded (norm less than 2)\n");
        System.out.printf("  then C is taken to be a member of the set.\n");
        System.out.printf("\n");
        System.out.printf("  An ASCII PPM image of the set is created using\n");
        System.out.printf("    N = %d pixels in the X direction and\n", n);
        System.out.printf("    N = %d pixels in the Y direction.\n", n);

        /* Carry out the iteration for each pixel, determining COUNT. */
        count = new int[m * n];

        Variables variables = Variables.create();

        try {
            variables.add("i", new PrivateVariable<>(0))
                     .add("j", new PrivateVariable<>(0))
                     .add("x", new PrivateVariable<>(0.0d))
                     .add("y", new PrivateVariable<>(0.0d))
                     .add("count", new SharedVariable<>(new int[m * n]))
                     .add("m", new SharedVariable<>(m))
                     .add("n", new SharedVariable<>(n))
                     .add("xMax", new SharedVariable<>(X_MAX))
                     .add("xMin", new SharedVariable<>(X_MIN))
                     .add("yMax", new SharedVariable<>(Y_MAX))
                     .add("yMin", new SharedVariable<>(Y_MIN))
                     .add("countMax", new SharedVariable<>(countMax))
                     .add("cMax", new ReductionVariable<>(ReductionOperations.max(), 0));

            JROMP.allThreads()
                 .withVariables(variables)
                 .single(false, (vars) -> tv1[0] = System.nanoTime())
                 .parallelFor(0, m, false, (start, end, vars) -> {
                     for (int i_i = start; i_i < end; i_i++) {
                         Variable<Double> x1 = vars.get("x");
                         x1.set((i_i * X_MAX + (m - i_i - 1) * X_MIN) / (m - 1));

                         for (int j_j = 0; j_j < vars.<Integer>get("n").value(); j_j++) {
                             Variable<Double> y1 = vars.get("y");
                             y1.set((j_j * Y_MAX + (n - j_j - 1) * Y_MIN) / (n - 1));

                             int explode = explode(x1.value(), y1.value(), countMax);
                             vars.<int[]>get("count").value()[i_i + j_j * m] = explode;
                         }
                     }
                 })
                 .parallelFor(0, m, false, (start, end, vars) -> {
                     for (int i_i = start; i_i < end; i_i++) {
                         for (int j_j = 0; j_j < vars.<Integer>get("n").value(); j_j++) {
                             Variable<Integer> cMax_ = vars.get("cMax");

                             int count1 = vars.<int[]>get("count").value()[i_i + j_j * m];
                             if (cMax_.value() < count1) {
                                 cMax_.set(count1);
                             }
                         }
                     }
                 })
                 .single(false, (vars) -> tv2[0] = System.nanoTime())
                 .join();
        } catch (IllegalStateException e) {
            e.printStackTrace();
        }

        final double executionSeconds = (tv2[0] - tv1[0]) / 1.0e9;
        System.out.printf("Wall clock time = %12.4g sec\n", executionSeconds);

        /* Set the image data. */
        r = new int[m * n];
        g = new int[m * n];
        b = new int[m * n];

        count = variables.<int[]>get("count").value();
        cMax = variables.<Integer>get("cMax").value();

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                if (count[i + j * m] % 2 == 1) {
                    r[i + j * m] = 255;
                    g[i + j * m] = 255;
                    b[i + j * m] = 255;
                } else {
                    c = (int) (255.0 * sqrt(sqrt(sqrt((double) count[i + j * m] / (double) cMax))));
                    r[i + j * m] = 3 * c / 5;
                    g[i + j * m] = 3 * c / 5;
                    b[i + j * m] = c;
                }
            }
        }

        /* Write an image file. */
        ppmWrite(filename, m, n, r, g, b);

        System.out.printf("\n");
        System.out.printf("  ASCII PPM image data stored in \"%s\".\n", filename);

        /* Free memory. */
        // Not needed in Java

        /* Terminate. */
        System.out.printf("\n");
        System.out.printf("MANDELBROT\n");
        System.out.printf("  Normal end of execution.\n");
        System.out.printf("\n");
        timestamp();
    }

    int explode(double x, double y, int countMax) {
        int k;
        int explosionStep;
        double x1, x2;
        double y1, y2;

        explosionStep = 0;
        x1 = x;
        y1 = y;

        for (k = 1; k <= countMax; k++) {
            x2 = x1 * x1 - y1 * y1 + x;
            y2 = 2.0 * x1 * y1 + y;

            if (x2 < -2.0 || 2.0 < x2 || y2 < -2.0 || 2.0 < y2) {
                explosionStep = k;
                break;
            }

            x1 = x2;
            y1 = y2;
        }

        return explosionStep;
    }

    int ppmWrite(String outputFilename, int xSize, int ySize, int[] r, int[] g, int[] b) throws IOException {
        File fileOut;
        int rIndex;
        int gIndex;
        int bIndex;
        int rgbMax;
        int error;
        int i;
        int j;

        /* Open the output file. */
        fileOut = new File(outputFilename);

        if (fileOut.exists()) {
            fileOut.delete();
        }

        boolean created = fileOut.createNewFile();

        if (!created) {
            System.out.printf("\n");
            System.out.printf("PPM_WRITE - Fatal error!\n");
            System.out.printf("  Cannot open the output file \"%s\".\n", outputFilename);
            return 1;
        }

        PrintStream stream = new PrintStream(fileOut, "UTF-8");

        /* Compute the maximum. */
        rgbMax = 0;
        rIndex = 0;
        gIndex = 0;
        bIndex = 0;

        for (j = 0; j < ySize; j++) {
            for (i = 0; i < xSize; i++) {
                if (rgbMax < r[rIndex]) {
                    rgbMax = r[rIndex];
                }
                rIndex++;

                if (rgbMax < g[gIndex]) {
                    rgbMax = g[gIndex];
                }
                gIndex++;

                if (rgbMax < b[bIndex]) {
                    rgbMax = b[bIndex];
                }
                bIndex++;
            }
        }

        /* Write the header. */
        error = ppmWriteHeader(stream, xSize, ySize, rgbMax);

        if (error == 1) {
            System.out.printf("\n");
            System.out.printf("PPM_WRITE - Fatal error!\n");
            System.out.printf("  PPM_WRITE_HEADER failed.\n");
            return 1;
        }

        /* Write the data. */
        error = ppmWriteData(stream, xSize, ySize, r, g, b);

        if (error == 1) {
            System.out.printf("\n");
            System.out.printf("PPM_WRITE - Fatal error!\n");
            System.out.printf("  PPM_WRITE_DATA failed.\n");
            return 1;
        }

        /* Close the file. */
        stream.close();
        return 0;
    }

    int ppmWriteData(PrintStream stream, int xSize, int ySize, int[] r, int[] g, int[] b) throws IOException {
        int rIndex;
        int gIndex;
        int bIndex;
        int rgbNum;
        int i;
        int j;

        rIndex = 0;
        gIndex = 0;
        bIndex = 0;
        rgbNum = 0;

        StringBuilder sb = new StringBuilder();

        for (j = 0; j < ySize; j++) {
            for (i = 0; i < xSize; i++) {
                sb.append(String.format("%d %d %d", r[rIndex], g[gIndex], b[bIndex]));

                rgbNum += 3;
                rIndex++;
                gIndex++;
                bIndex++;

                if (rgbNum % 12 == 0 || i == xSize - 1 || rgbNum == 3 * xSize * ySize) {
                    sb.append("\n");
                } else {
                    sb.append(" ");
                }
            }
        }

        stream.print(sb.toString());
        return 0;
    }

    int ppmWriteHeader(PrintStream stream, int xSize, int ySize, int rgbMax) throws IOException {
        stream.printf("P3\n");
        stream.printf("%d %d\n", xSize, ySize);
        stream.printf("%d\n", rgbMax);

        return 0;
    }

    void timestamp() {
        System.out.printf("%s\n", new Date(System.currentTimeMillis()));
    }
}

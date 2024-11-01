package jromp.mpi.examples.mandelbrot;

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
public class MandelbrotOriginal {
    public static void main(String[] args) throws IOException {
        new MandelbrotOriginal().$main(args);
    }

    public void $main(String[] args) throws IOException {
        int[] r;
        int[] g;
        int[] b;
        int c;
        int cMax;
        int[] count;
        final int countMax = COUNT_MAX;
        String filename = "mandelbrot_java_original.ppm";
        int i;
        int j;
        final int m = IMAGE_SIZE;
        final int n = IMAGE_SIZE;
        double x;
        double y;
        double tv1, tv2;

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
        tv1 = System.nanoTime();

        for (i = 0; i < m; i++) {
            x = ((double) i * X_MAX + (double) (m - i - 1) * X_MIN) / (double) (m - 1);

            for (j = 0; j < n; j++) {
                y = ((double) j * Y_MAX + (double) (n - j - 1) * Y_MIN) / (double) (n - 1);

                count[i + j * m] = explode(x, y, countMax);
            }
        }

        /* Set CMAX to the maximum count. */
        cMax = 0;
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                if (cMax < count[i + j * m]) {
                    cMax = count[i + j * m];
                }
            }
        }

        tv2 = System.nanoTime();
        final double executionSeconds = (tv2 - tv1) / 1.0e9;
        System.out.printf("Wall clock time = %12.4g sec\n", executionSeconds);

        /* Set the image data. */
        r = new int[m * n];
        g = new int[m * n];
        b = new int[m * n];

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

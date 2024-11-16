#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX 3000  // Tamaño de la placa en X
#define NY 3000  // Tamaño de la placa en Y
#define ITER 100 // Número de iteraciones

void initialize(double *plate, int nx, int ny);

void print_plate(double *plate, int nx, int ny);

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Dividir la placa entre los procesos
    int local_ny = NY / size; // Cada proceso maneja un bloque de filas
    if (rank == size - 1)
        local_ny += NY % size; // Último proceso toma filas restantes

    // Reservar memoria para la subplaca local
    double *local_plate = (double *) malloc((NX * (local_ny + 2)) * sizeof(double));
    double *new_local_plate = (double *) malloc((NX * (local_ny + 2)) * sizeof(double));

    // Inicializar la subplaca
    initialize(local_plate, NX, local_ny + 2);

    for (int iter = 0; iter < ITER; iter++) {
        // Intercambiar filas adyacentes entre procesos MPI
        if (rank > 0) {
            MPI_Sendrecv(&local_plate[NX], NX, MPI_DOUBLE, rank - 1, 0,
                         &local_plate[0], NX, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&local_plate[NX * local_ny], NX, MPI_DOUBLE, rank + 1, 0,
                         &local_plate[NX * (local_ny + 1)], NX, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Actualizar la subplaca local con OpenMP
        #pragma omp parallel for
        for (int y = 1; y <= local_ny; y++) {
            for (int x = 1; x < NX - 1; x++) {
                new_local_plate[y * NX + x] = 0.25 * (
                                                  local_plate[(y - 1) * NX + x] +
                                                  local_plate[(y + 1) * NX + x] +
                                                  local_plate[y * NX + (x - 1)] +
                                                  local_plate[y * NX + (x + 1)]
                                              );
            }
        }

        // Copiar los nuevos valores a la subplaca
        #pragma omp parallel for
        for (int i = 0; i < NX * (local_ny + 2); i++) {
            local_plate[i] = new_local_plate[i];
        }
    }

    // Imprimir o recolectar los datos (opcional)
    if (rank == 0) {
        printf("Simulación completada.\n");
    }

    free(local_plate);
    free(new_local_plate);
    MPI_Finalize();
    return 0;
}

void initialize(double *plate, int nx, int ny) {
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            if (y == 0 || y == ny - 1 || x == 0 || x == nx - 1) {
                plate[y * nx + x] = 100.0; // Bordes calientes
            } else {
                plate[y * nx + x] = 0.0; // Interior frío
            }
        }
    }
}

void print_plate(double *plate, int nx, int ny) {
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            printf("%.2f ", plate[y * nx + x]);
        }
        printf("\n");
    }
}

# Configuración Caléndula

## Compilación de OpenMPI

Se han probado diferentes configuraciones de compilación:

### Usando el compilador icelake/gcc_9.4.0 y bibliotecas internas

Este compilador tiene todas las dependencias instaladas, por lo que no es necesario usar ningún módulo más.

### Usando el compilador GCC/11.2.0 y bibliotecas internas

Compilando con esta versión y dependencias, se obtuvieron distintos errores en tiempo de ejecución debido
a que no se integraban bien con la biblioteca de Infiniband. Se llegó a ejecutar con éxito en un nodo, pero
al intentar ejecutar en varios nodos, se producía un error de comunicación.

### Usando el compilador GCC/11.2.0, bibliotecas externas y parámetros de ejecución en Java y MPI

Esta configuración es la que ha resultado más exitosa. Se han usado las bibliotecas que están instaladas
en Caléndula, que son las siguientes:

- `libevent/2.1.12-GCCcore-11.2.0`
- `hwloc/2.5.0-GCCcore-11.2.0`
- `PMIx/4.1.0-GCCcore-11.2.0`
- `libfabric/1.13.2-GCCcore-11.2.0`
- `UCX/1.11.2-GCCcore-11.2.0`

Además, se han usado los siguientes parámetros de ejecución:

#### Parámetros de ejecución en Java

- `-XX:+UseParallelGC`: para usar el recolector de basura paralelo (https://www.javaperformancetuning.com/news/qotm026.shtml).
- `-XX:-TieredCompilation`: para desactivar la compilación por niveles (https://www.baeldung.com/jvm-tiered-compilation).

#### Parámetros de ejecución en MPI

- `--mca pml ob1`: para usar el módulo de comunicación ob1.

Además de las variables de entorno siguientes (solo se usan para evitar warnings):

- `PRTE_MCA_plm_slurm_disable_warning = true`.
- `PRTE_MCA_plm_slurm_ignore_args = true`.

# Scripts utilizados

Se han programado diferentes scripts en Bash para facilitar la compilación y ejecución de los programas en Caléndula.

## Compilación de MPI

El script [prepare_libs_calendula.bash](prepare_libs_calendula.bash) se encarga de compilar el programa MPI.
Se debe ejecutar en una sesión interactiva en un nodo comprendido entre los ids cn6009 y cn6026
(icelake y Rocky Linux 8).

```bash
salloc --account=ule_formacion_9 --partition=formacion --qos=formacion --time=01:00:00 --nodelist=cn6009 --cpus-per-task=16
```

Una vez dentro del nodo ya se puede ejecutar el script.

```bash
./prepare_libs_calendula.bash
```

## Ejecución de MPI

Para ejecutar el programa de Java que utiliza MPI, debemos usar el programa `mpirun` que ha resultado
de la compilación anterior. Para ello, se han programado los scripts [run_calendula.bash](run_calendula.bash) y
[run.slurm](run.slurm), que se encargan de establecer los parámetros de ejecución necesarios.

```bash
./run_calendula.bash
```

plugins {
    id("java")
}

group = "io.github.mpi"
version = "1.0-SNAPSHOT"

val ompiLibPath = "${project.projectDir}/libs/ompi"
val dependencyJarsPath = "${project.projectDir}/libs/java"
val mpiBinPath = "$ompiLibPath/bin"
val mpiLibPath = "$ompiLibPath/lib"
val mpiRun = "$mpiBinPath/mpirun"
val mpiJavac = "$mpiBinPath/mpijavac.pl"

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(
        files(
            "$dependencyJarsPath/commons-lang3-3.16.0.jar",
            "$dependencyJarsPath/jromp-2.2.0.jar",
            "$mpiLibPath/mpi.jar"
        )
    )
}

tasks.compileJava {
    options.forkOptions.executable = mpiJavac
}

fun createTaskWithNumProcesses(name: String, processes: Int, debug: Boolean) {
    tasks.register<Exec>("run$name") {
        dependsOn("classes")

        group = "application"
        description = "Run $name with mpirun"

        val classpath = sourceSets.main.get().runtimeClasspath.asPath
        val mpiRunParameters = mutableListOf("--mca", "pml", "ob1", "--bind-to", "none")
        val jvmParameters = listOf("-XX:+UseParallelGC", "-XX:-TieredCompilation")

        if (debug) {
            mpiRunParameters.add("--report-bindings")
        }

        // Disable SLURM warnings (not really needed in local environment)
        environment("PRTE_MCA_plm_slurm_disable_warning", true)
        environment("PRTE_MCA_plm_slurm_ignore_args", true)

        commandLine =
            listOf(
                mpiRun,
                *mpiRunParameters.toTypedArray(),
                "-np", "$processes",
                "java", *jvmParameters.toTypedArray(), "-cp", classpath, "jromp.mpi.examples.$name"
            )

        standardOutput = System.out
        errorOutput = System.err
        isIgnoreExitValue = false

        if (debug) {
            doLast {
                val cmd = environment.map { (key, value) -> "$key=$value" }.toMutableList()
                cmd.addAll(commandLine)
                println()
                println()
                println("\u001B[33mExecuted command:\n  ${cmd.joinToString(" ")}\u001B[0m")
            }
        }
    }
}

createTaskWithNumProcesses("HelloMPI", 4, true)
createTaskWithNumProcesses("Blocking", 6, true)
createTaskWithNumProcesses("Burro", 6, true)
createTaskWithNumProcesses("Cross", 4, true)
createTaskWithNumProcesses("FullParallel", 3, true)
createTaskWithNumProcesses("SimpleJROMP", 1, true)
createTaskWithNumProcesses("JrompMPI", 2, true)

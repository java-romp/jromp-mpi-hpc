import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI

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

fun checkInternetConnection(): Boolean {
    try {
        val url = URI.create("https://www.google.com").toURL()
        val connection = url.openConnection() as HttpURLConnection

        connection.connectTimeout = 3000
        connection.connect()

        val okResponse = connection.responseCode == 200

        if (okResponse) {
            println("Internet connection is available")
        } else {
            println("Internet connection is not available")
        }

        return okResponse
    } catch (e: IOException) {
        println("Internet connection is not available")
        return false
    }
}

dependencies {
    if (checkInternetConnection()) {
        implementation("org.apache.commons:commons-lang3:3.16.0")
        implementation("io.github.java-romp:jromp:3.0.0")
        implementation(files("$mpiLibPath/mpi.jar"))
    } else {
        implementation(
            files(
                "$dependencyJarsPath/commons-lang3-3.16.0.jar",
                "$dependencyJarsPath/jromp-3.0.0.jar",
                "$mpiLibPath/mpi.jar"
            )
        )
    }
}

tasks.compileJava {
    options.forkOptions.executable = mpiJavac
}

fun createTaskWithNumProcesses(
    name: String,
    processes: Int,
    debug: Boolean,
    subpackage: String = "",
    params: List<String> = emptyList()
) {
    tasks.register<Exec>("run$name") {
        dependsOn("classes")

        group = "application"
        description = "Run $name with mpirun"

        val classpath = sourceSets.main.get().runtimeClasspath.asPath
        val mpiRunParameters = mutableListOf("--bind-to", "none")
        val jvmParameters = listOf("-XX:+UseParallelGC", "-XX:-TieredCompilation")

        if (debug) {
            mpiRunParameters.add("--report-bindings")
        }

        // Disable SLURM warnings (not really needed in local environment)
        environment("PRTE_MCA_plm_slurm_disable_warning", true)
        environment("PRTE_MCA_plm_slurm_ignore_args", true)
        environment("UCX_NET_DEVICES", "mlx5_0:1")

        val pkg = StringBuilder("jromp.mpi.examples")
            .append(if (subpackage.isNotEmpty()) ".$subpackage" else "")
            .append(".")
            .append(name)

        commandLine =
            listOf(
                mpiRun,
                *mpiRunParameters.toTypedArray(),
                "-np", "$processes",
                "java", *jvmParameters.toTypedArray(), "-cp", classpath, "-ea", pkg.toString(),
                *params.toTypedArray()
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
createTaskWithNumProcesses("Gemm", 5, false, "gemm", listOf("2000", "4"))

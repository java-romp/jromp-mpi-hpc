import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if not os.path.exists("diagrams"):
    os.makedirs("diagrams")


# Función para crear todos los gráficos de las ejecuciones en C (múltiples optimizaciones)
# columnas: n,workers,threads,total_cpus,opt_level,time
def plot_graphs_c(data):
    if not os.path.exists("diagrams/c"):
        os.makedirs("diagrams/c")

    sns.set_theme(style="whitegrid")

    pivot_table = np.log1p(data.pivot_table(values="time", index="workers", columns="threads"))
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Heatmap C (log scale): Time depending on Workers and Threads")
    plt.xlabel("Threads")
    plt.ylabel("Workers")
    plt.savefig("diagrams/c/heatmap_time.png", dpi=300)
    plt.close()


# Función para crear todos los gráficos de las ejecuciones en Java (sin optimizaciones)
# columnas: n,workers,threads,total_cpus,time
def plot_graphs_java(data):
    if not os.path.exists("diagrams/java"):
        os.makedirs("diagrams/java")

    sns.set_theme(style="whitegrid")

    pivot_table = np.log1p(data.pivot_table(values="time", index="workers", columns="threads"))
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title("Heatmap Java (log scale): Time depending on Workers and Threads")
    plt.xlabel("Threads")
    plt.ylabel("Workers")
    plt.savefig("diagrams/java/heatmap_time.png", dpi=300)
    plt.close()


# Función para crear todos los gráficos de las ejecuciones en CUDA (múltiples optimizaciones)
# columnas: n,optimization,elapsed_time
def plot_graphs_cuda(data):
    if not os.path.exists("diagrams/cuda"):
        os.makedirs("diagrams/cuda")

    sns.set_theme(style="whitegrid")
    custom_palette = sns.color_palette("bright")
    lw = 1.5

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="n", y="elapsed_time", hue="optimization", style="optimization", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Time vs. N")
    plt.xlabel("N")
    plt.ylabel("Time (s)")
    plt.legend(title="Optimization")
    plt.yscale("log")
    plt.savefig("diagrams/cuda/time_vs_n_log.png", dpi=300)
    plt.close()


def plot_graphs_c_java_combined(c_data, java_data):
    if not os.path.exists("diagrams/c_java"):
        os.makedirs("diagrams/c_java")

    sns.set_theme(style="whitegrid")
    custom_palette = sns.color_palette("bright")
    lw = 1.5

    # Definir factor de margen
    margin_factor = 0.05  # 5% de margen

    # Make plots with 2 subplots (left C, right Java)
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    # C
    sns.lineplot(data=c_data, x="threads", y="time", hue="workers", style="workers", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None, ax=axs[0])
    axs[0].set_title("Time vs. Thread configuration (C)")
    axs[0].set_xlabel("Threads")
    axs[0].set_ylabel("Time (s)")
    axs[0].legend(title="Workers")

    # Java
    sns.lineplot(data=java_data, x="threads", y="time", hue="workers", style="workers", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None, ax=axs[1])
    axs[1].set_title("Time vs. Thread configuration (Java)")
    axs[1].set_xlabel("Threads")
    axs[1].set_ylabel("Time (s)")
    axs[1].legend(title="Workers")

    # Ajustar ejes iguales con margen
    x_min = min(c_data["threads"].min(), java_data["threads"].min())
    x_max = max(c_data["threads"].max(), java_data["threads"].max())
    y_min = min(c_data["time"].min(), java_data["time"].min())
    y_max = max(c_data["time"].max(), java_data["time"].max())

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_ticks = c_data["threads"].unique()

    for ax in axs:
        ax.set_xlim(x_min - x_range * margin_factor, x_max + x_range * margin_factor)
        ax.set_ylim(y_min - y_range * margin_factor, y_max + y_range * margin_factor)
        ax.set_xticks(x_ticks)

    plt.savefig("diagrams/c_java/time_vs_threads.png", dpi=300)
    plt.close()

    # Single plot with Java data appended to the C data (adding a new column to the java one: opt_level = "Java")
    # Prepend "C" to the opt_level column of the C data
    java_data["opt_level"] = "Java (No Optimized)"
    c_data.sort_values(by="opt_level", inplace=True)
    c_data["opt_level"] = "C-" + c_data["opt_level"].astype(str)
    data_ = [c_data, java_data]
    combined_data = pd.concat(data_, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_data, x="total_cpus", y="time", hue="opt_level", style="opt_level", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Time vs. Total CPUs (C & Java)")
    plt.xlabel("Total CPUs")
    plt.ylabel("Time (s)")
    plt.legend(title="Optimization Level")
    plt.yscale("log")
    plt.savefig("diagrams/c_java/time_vs_total_cpus_log_combined.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Cargar datos
    data_c = pd.read_csv("datos/execution_configs_c.csv")
    data_java = pd.read_csv("datos/execution_configs_java.csv")
    data_cuda = pd.read_csv("datos/execution_configs_cuda.csv")

    # Crear gráficos
    plot_graphs_c(data_c)
    plot_graphs_java(data_java)
    plot_graphs_cuda(data_cuda)
    plot_graphs_c_java_combined(data_c, data_java)

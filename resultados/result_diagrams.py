import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if not os.path.exists("diagrams"):
    os.makedirs("diagrams")


def plot_graphs_c(data):
    if not os.path.exists("diagrams/c"):
        os.makedirs("diagrams/c")

    sns.set_theme(style="whitegrid")

    pivot_table = np.log1p(data.pivot_table(values="time", index="workers", columns="threads"))
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="rocket_r")
    plt.title("Heatmap C (log scale)", fontsize=14)
    plt.xlabel("Threads")
    plt.ylabel("Workers")
    plt.tight_layout()
    plt.savefig("diagrams/c/heatmap_time.png", dpi=300)
    plt.close()


def plot_graphs_java(data):
    if not os.path.exists("diagrams/java"):
        os.makedirs("diagrams/java")

    sns.set_theme(style="whitegrid")

    pivot_table = np.log1p(data.pivot_table(values="time", index="workers", columns="threads"))
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="rocket_r")
    plt.title("Heatmap Java (log scale)", fontsize=14)
    plt.xlabel("Threads")
    plt.ylabel("Workers")
    plt.tight_layout()
    plt.savefig("diagrams/java/heatmap_time.png", dpi=300)
    plt.close()


def plot_graphs_all_combined(c_data, java_data, cuda_data):
    if not os.path.exists("diagrams/combined"):
        os.makedirs("diagrams/combined")

    sns.set_theme(style="whitegrid")
    custom_palette = sns.color_palette("bright")
    lw = 1.5

    # Definir factor de margen
    margin_factor = 0.05  # 5% de margen

    # Make plots with 2 subplots (left C, right Java)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.tight_layout(pad=5.0)

    # C -O0
    sns.lineplot(data=c_data[c_data["opt_level"] == 0], x="threads", y="time", hue="workers", style="workers", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None, ax=axs[0, 0])
    axs[0, 0].set_title("Time vs. Thread configuration (C -O0)", fontsize=14)
    axs[0, 0].set_xlabel("Threads")
    axs[0, 0].set_ylabel("Time (s)")
    axs[0, 0].legend(title="Workers")

    # C -O1
    sns.lineplot(data=c_data[c_data["opt_level"] == 1], x="threads", y="time", hue="workers", style="workers", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None, ax=axs[0, 1])
    axs[0, 1].set_title("Time vs. Thread configuration (C -O1)", fontsize=14)
    axs[0, 1].set_xlabel("Threads")
    axs[0, 1].set_ylabel("Time (s)")
    axs[0, 1].legend(title="Workers")

    # C -O2
    sns.lineplot(data=c_data[c_data["opt_level"] == 2], x="threads", y="time", hue="workers", style="workers", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None, ax=axs[1, 0])
    axs[1, 0].set_title("Time vs. Thread configuration (C -O2)", fontsize=14)
    axs[1, 0].set_xlabel("Threads")
    axs[1, 0].set_ylabel("Time (s)")
    axs[1, 0].legend(title="Workers")

    # C -O3
    sns.lineplot(data=c_data[c_data["opt_level"] == 3], x="threads", y="time", hue="workers", style="workers", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None, ax=axs[1, 1])
    axs[1, 1].set_title("Time vs. Thread configuration (C -O3)", fontsize=14)
    axs[1, 1].set_xlabel("Threads")
    axs[1, 1].set_ylabel("Time (s)")
    axs[1, 1].legend(title="Workers")

    # Ajustar ejes iguales con margen
    x_min = c_data["threads"].min()
    x_max = c_data["threads"].max()
    y_min = c_data["time"].min()
    y_max = c_data["time"].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_ticks = c_data["threads"].unique()

    for ax in axs:
        for a in ax:
            a.set_xlim(x_min - x_range * margin_factor, x_max + x_range * margin_factor)
            a.set_ylim(y_min - y_range * margin_factor, y_max + y_range * margin_factor)
            a.set_xticks(x_ticks)

    plt.savefig("diagrams/combined/time_vs_threads_all_c.png", dpi=300)
    plt.close()

    # (time_vs_threads_c0_java) Single plot with C opt level 0 and Java data
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.tight_layout(pad=5.0)

    # C -O0
    sns.lineplot(data=c_data[c_data["opt_level"] == 0], x="threads", y="time", hue="workers", style="workers", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None, ax=axs[0])
    axs[0].set_title("Time vs. Thread configuration (C -O0)", fontsize=14)
    axs[0].set_xlabel("Threads")
    axs[0].set_ylabel("Time (s)")
    axs[0].legend(title="Workers")

    # Java
    sns.lineplot(data=java_data, x="threads", y="time", hue="workers", style="workers", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None, ax=axs[1])
    axs[1].set_title("Time vs. Thread configuration (Java)", fontsize=14)
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

    plt.savefig("diagrams/combined/time_vs_threads_c0_java.png", dpi=300)
    plt.close()

    # Time vs. Total CPUs log scale
    include_cuda = True
    c_data.sort_values(by="opt_level", inplace=True)
    c_data_ = c_data.copy()
    c_data_["opt_level"] = "C -O" + c_data_["opt_level"].astype(str)
    data_ = [c_data_, java_data]
    combined_data = pd.concat(data_, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_data, x="total_cpus", y="time", hue="opt_level", style="opt_level", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None)
    if include_cuda:
        fastest_time_cuda = cuda_data["elapsed_time"].min() / 1000  # Convert to seconds
        plt.scatter(x=1, y=fastest_time_cuda, color="g", label="CUDA", s=50, marker="x")
    plt.title("Time vs. Total CPUs (C and Java)", fontsize=14)
    plt.xlabel("Total CPUs")
    plt.ylabel("Time (s)")
    plt.legend(title="Execution type")
    plt.yscale("log")
    plt.tight_layout()
    plt.xticks([1, 200, 400, 600, 800, 960])
    plt.savefig("diagrams/combined/time_vs_total_cpus_log_combined.png", dpi=300)
    plt.close()


def plot_graph_all_combined_one_c_optimization(c_data, java_data, cuda_data):
    if not os.path.exists("diagrams/combined"):
        os.makedirs("diagrams/combined")

    custom_palette = sns.color_palette("bright")
    lw = 1.5

    optimization = 3
    c_data.sort_values(by="opt_level", inplace=True)
    c_data_ = c_data.copy()
    c_data_ = c_data_[c_data_["opt_level"] == optimization]
    c_data_["opt_level"] = f"C -O{optimization}"
    data_ = [c_data_, java_data]
    combined_data = pd.concat(data_, ignore_index=True)
    fastest_time_cuda = cuda_data["elapsed_time"].min() / 1000  # Convert to seconds

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_data, x="total_cpus", y="time", hue="opt_level", style="opt_level", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None)
    plt.scatter(x=1, y=fastest_time_cuda, color="g", label="CUDA", s=50, marker="x")
    plt.title("Time vs. Total CPUs (C, Java and CUDA)", fontsize=14)
    plt.xlabel("Total CPUs")
    plt.ylabel("Time (s)")
    plt.legend(title="Execution type")
    plt.yscale("log")
    plt.tight_layout()
    plt.xticks([1, 200, 400, 600, 800, 960])
    plt.savefig(f"diagrams/combined/time_vs_total_cpus_log_combined_o{optimization}.png", dpi=300)
    plt.close()


def c_speedup(data):
    if not os.path.exists("diagrams/c"):
        os.makedirs("diagrams/c")

    sns.set_theme(style="whitegrid")

    # Calcular speedup respecto a la ejecución secuencial (total_cpus = 1, opt_level = 0)
    sequential_time = data[(data["total_cpus"] == 1) & (data["opt_level"] == 0)]["time"].min()
    data["speedup"] = sequential_time / data["time"]
    data_copy = data.copy()
    data_copy["opt_level"] = "-O" + data_copy["opt_level"].astype(str)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_copy, x="total_cpus", y="speedup", hue="opt_level", style="opt_level", markers=True, dashes=False,
                 palette="bright", linewidth=1.5, errorbar=None)
    plt.title("Speedup vs. Total CPUs (C)", fontsize=14)
    plt.xlabel("Total CPUs")
    plt.ylabel("Speedup")
    plt.legend(title="Optimization level")
    plt.tight_layout()
    plt.savefig("diagrams/c/speedup_vs_total_cpus.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Cargar datos
    data_c = pd.read_csv("datos/execution_configs_c.csv")
    data_java = pd.read_csv("datos/execution_configs_java.csv")
    data_cuda = pd.read_csv("datos/execution_configs_cuda.csv")

    data_java["opt_level"] = "Java"

    # Crear gráficos
    plot_graphs_c(data_c)
    plot_graphs_java(data_java)
    plot_graphs_all_combined(data_c, data_java, data_cuda)
    plot_graph_all_combined_one_c_optimization(data_c, data_java, data_cuda)
    c_speedup(data_c)

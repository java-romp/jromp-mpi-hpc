import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Función para crear gráficos
def plot_graphs(data):
    sns.set_theme(style="whitegrid")
    custom_palette = sns.color_palette("bright")
    lw = 1.5

    # 1. Tiempo vs. Número de Workers
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="workers", y="time", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Tiempo vs. Número de Workers")
    plt.xlabel("Número de Workers")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("time_vs_workers.png", dpi=300)
    plt.close()

    # 2. Tiempo vs. Total CPUs
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="total_cpus", y="time", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Tiempo vs. Total CPUs")
    plt.xlabel("Total CPUs")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("time_vs_total_cpus.png", dpi=300)
    plt.close()

    # 2.1 Tiempo vs. Total CPUs (log scale)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="total_cpus", y="time", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Tiempo vs. Total CPUs")
    plt.xlabel("Total CPUs")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Nivel de Optimización")
    plt.yscale("log")
    plt.savefig("time_vs_total_cpus_log.png", dpi=300)
    plt.close()

    # 2.2 Tiempo vs. Total CPUs (total cpus limit 40), remove from data to aux variable and work with the dataframe
    aux = data[data["total_cpus"] <= 40]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=aux, x="total_cpus", y="time", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Tiempo vs. Total CPUs (Total CPUs <= 40)")
    plt.xlabel("Total CPUs")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("time_vs_total_cpus_40.png", dpi=300)
    plt.close()

    # 2.3 Tiempo vs. Total CPUs (total cpus from 40 to last), remove from data to aux variable and work with the dataframe
    aux = data[data["total_cpus"] > 40]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=aux, x="total_cpus", y="time", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Tiempo vs. Total CPUs (Total CPUs > 40)")
    plt.xlabel("Total CPUs")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("time_vs_total_cpus_40_last.png", dpi=300)
    plt.close()

    # 4. Eficiencia Escalable
    base_time = data[data["workers"] == 1]["time"].min()
    data["efficiency"] = base_time / (data["time"] * data["total_cpus"])
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="total_cpus", y="efficiency", markers=True,
                 dashes=False, palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Eficiencia Escalable")
    plt.xlabel("Total CPUs")
    plt.ylabel("Eficiencia")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("efficiency_vs_cpus.png", dpi=300)
    plt.close()

    # 5. Comparación por Configuración de Threads
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="threads", y="time", hue="workers", style="workers", markers=True, dashes=False,
                 palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Tiempo vs. Configuración de Threads")
    plt.xlabel("Número de Threads")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Workers")
    plt.savefig("time_vs_threads.png", dpi=300)
    plt.close()

    # 6. Heatmap de Tiempo
    pivot_table = data.pivot_table(values="time", index="workers", columns="threads")
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title("Heatmap: Tiempo en función de Workers y Threads")
    plt.xlabel("Threads")
    plt.ylabel("Workers")
    plt.savefig("heatmap_time.png", dpi=300)
    plt.close()

    # 8. Porcentaje de Optimización vs. Total CPUs
    max_cpus = data["total_cpus"].max()
    data["optimization_percentage"] = (data["total_cpus"] / max_cpus) * 100
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="optimization_percentage", y="time", markers=True,
                 dashes=False, palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Tiempo vs. Porcentaje de Optimización")
    plt.xlabel("Porcentaje de Optimización (%)")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("time_vs_optimization_percentage.png", dpi=300)
    plt.close()

    # 10. Speedup vs. Total CPUs
    sequential_time = data[data["total_cpus"] == 1]["time"].min()
    data["speedup"] = sequential_time / data["time"]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="total_cpus", y="speedup", markers=True,
                 dashes=False, palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Speedup vs. Total CPUs")
    plt.xlabel("Total CPUs")
    plt.ylabel("Speedup")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("speedup_vs_total_cpus.png", dpi=300)
    plt.close()

    # 11. Coste Computacional Absoluto
    data["computational_cost"] = data["time"] * data["total_cpus"]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="total_cpus", y="computational_cost", markers=True,
                 dashes=False, palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Coste Computacional Absoluto vs. Total CPUs")
    plt.xlabel("Total CPUs")
    plt.ylabel("Coste Computacional Absoluto")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("computational_cost_vs_total_cpus.png", dpi=300)
    plt.close()


if __name__ == '__main__':
    # Configurar Matplotlib para usar el backend "tkagg" (Tkinter)
    plt.switch_backend('tkagg')

    # Cargar datos
    data = pd.read_csv("../resultados/data/execution_configs_java.csv")

    plot_graphs(data)

    print("Gráficos generados y guardados en el directorio actual.")

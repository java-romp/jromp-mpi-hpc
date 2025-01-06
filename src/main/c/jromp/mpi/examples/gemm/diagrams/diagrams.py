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
    sns.lineplot(data=data, x="workers", y="time", hue="opt_level", marker="o", palette=custom_palette, linewidth=lw,
                 errorbar=None)
    plt.title("Tiempo vs. Número de Workers")
    plt.xlabel("Número de Workers")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("time_vs_workers.png")
    plt.close()

    # 2. Tiempo vs. Total CPUs
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="total_cpus", y="time", hue="opt_level", marker="o", palette=custom_palette, linewidth=lw,
                 errorbar=None)
    plt.title("Tiempo vs. Total CPUs")
    plt.xlabel("Total CPUs")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("time_vs_total_cpus.png")
    plt.close()

    # 2.1 Tiempo vs. Total CPUs (log scale)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="total_cpus", y="time", hue="opt_level", marker="o", palette=custom_palette, linewidth=lw,
                 errorbar=None)
    plt.title("Tiempo vs. Total CPUs")
    plt.xlabel("Total CPUs")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Nivel de Optimización")
    plt.yscale("log")
    plt.savefig("time_vs_total_cpus_log.png")
    plt.close()

    # 2.2 Tiempo vs. Total CPUs (total cpus limit 40), remove from data to aux variable and work with the dataframe
    aux = data[data["total_cpus"] <= 40]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=aux, x="total_cpus", y="time", hue="opt_level", marker="o", palette=custom_palette, linewidth=lw,
                 errorbar=None)
    plt.title("Tiempo vs. Total CPUs (Total CPUs <= 40)")
    plt.xlabel("Total CPUs")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("time_vs_total_cpus_40.png")
    plt.close()

    # 2.3 Tiempo vs. Total CPUs (total cpus from 40 to last), remove from data to aux variable and work with the dataframe
    aux = data[data["total_cpus"] > 40]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=aux, x="total_cpus", y="time", hue="opt_level", marker="o", palette=custom_palette, linewidth=lw,
                 errorbar=None)
    plt.title("Tiempo vs. Total CPUs (Total CPUs > 40)")
    plt.xlabel("Total CPUs")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("time_vs_total_cpus_40_last.png")
    plt.close()

    # 3. Tiempo vs. Nivel de Optimización
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="opt_level", y="time", hue="workers", marker="o", palette=custom_palette, linewidth=lw,
                 errorbar=None)
    plt.title("Tiempo vs. Nivel de Optimización")
    plt.xlabel("Nivel de Optimización")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Workers")
    plt.savefig("time_vs_opt_level.png")
    plt.close()

    # 4. Eficiencia Escalable
    base_time = data[data["workers"] == 1]["time"].min()
    data["efficiency"] = base_time / (data["time"] * data["total_cpus"])
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="total_cpus", y="efficiency", hue="opt_level", marker="o", palette=custom_palette,
                 linewidth=lw, errorbar=None)
    plt.title("Eficiencia Escalable")
    plt.xlabel("Total CPUs")
    plt.ylabel("Eficiencia")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("efficiency_vs_cpus.png")
    plt.close()

    # 5. Comparación por Configuración de Threads
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="threads", y="time", hue="workers", marker="o", palette=custom_palette, linewidth=lw,
                 errorbar=None)
    plt.title("Tiempo vs. Configuración de Threads")
    plt.xlabel("Número de Threads")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Workers")
    plt.savefig("time_vs_threads.png")
    plt.close()

    # 6. Heatmap de Tiempo
    pivot_table = data.pivot_table(values="time", index="workers", columns="threads")
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title("Heatmap: Tiempo en función de Workers y Threads")
    plt.xlabel("Threads")
    plt.ylabel("Workers")
    plt.savefig("heatmap_time.png")
    plt.close()

    # 7. Tiempo Relativo a Caso Base
    case_base = data[data["opt_level"] == 0]["time"].min()
    data["relative_time"] = data["time"] / case_base
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="opt_level", y="relative_time", hue="workers", marker="o", palette=custom_palette,
                 linewidth=lw, errorbar=None)
    plt.title("Tiempo Relativo vs. Nivel de Optimización")
    plt.xlabel("Nivel de Optimización")
    plt.ylabel("Tiempo Relativo")
    plt.legend(title="Workers")
    plt.savefig("relative_time_vs_opt_level.png")
    plt.close()

    # 8. Porcentaje de Optimización vs. Total CPUs
    max_cpus = data["total_cpus"].max()
    data["optimization_percentage"] = (data["total_cpus"] / max_cpus) * 100
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="optimization_percentage", y="time", hue="opt_level", marker="o", palette=custom_palette,
                 linewidth=lw, errorbar=None)
    plt.title("Tiempo vs. Porcentaje de Optimización")
    plt.xlabel("Porcentaje de Optimización (%)")
    plt.ylabel("Tiempo (s)")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("time_vs_optimization_percentage.png")
    plt.close()

    # 9. Porcentaje Ajustado de Optimización vs. Total CPUs
    max_cpus_opt3 = data[data["opt_level"] == 3]["total_cpus"].max()
    data["adjusted_optimization_percentage"] = (data["total_cpus"] / max_cpus_opt3) * (data["opt_level"] / 3) * 100
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="total_cpus", y="adjusted_optimization_percentage", hue="opt_level", marker="o",
                 palette=custom_palette, linewidth=lw, errorbar=None)
    plt.title("Porcentaje Ajustado de Optimización vs. Total CPUs")
    plt.xlabel("Total CPUs")
    plt.ylabel("Porcentaje Ajustado de Optimización (%)")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("adjusted_optimization_percentage_vs_total_cpus.png")
    plt.close()

    # 10. Speedup vs. Total CPUs
    sequential_time = data[data["total_cpus"] == 1 & (data["opt_level"] == 0)]["time"].min()
    data["speedup"] = sequential_time / data["time"]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="total_cpus", y="speedup", hue="opt_level", marker="o", palette=custom_palette,
                 linewidth=lw, errorbar=None)
    plt.title("Speedup vs. Total CPUs")
    plt.xlabel("Total CPUs")
    plt.ylabel("Speedup")
    plt.legend(title="Nivel de Optimización")
    plt.savefig("speedup_vs_total_cpus.png")
    plt.close()


if __name__ == '__main__':
    # Configurar Matplotlib para usar el backend "tkagg" (Tkinter)
    plt.switch_backend('tkagg')

    # Cargar datos
    data = pd.read_csv("../execution_configs_c.csv")

    plot_graphs(data)

    print("Gráficos generados y guardados en el directorio actual.")

import os
import psutil


def get_free_cpus():
    total_cpus = os.cpu_count() or 1
    used_cpus = psutil.cpu_percent(percpu=True)  # Список загрузки каждого ядра (%)

    # Считаем "свободные" ядра (где загрузка < 50%)
    free_cpus = sum(1 for usage in used_cpus if usage < 30)

    return max(1, free_cpus)  # Минимум 1 процесс


def get_optimal_workers(task_type="cpu"):
    free_cpus = get_free_cpus()

    if task_type == "cpu":
        return max(1, free_cpus - 1)  # CPU-интенсивные задачи
    elif task_type == "io":
        return max(1, 2 * free_cpus)  # I/O-интенсивные задачи
    else:
        return 1
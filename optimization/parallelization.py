import multiprocessing


def division_processes(file_paths, cpus, method, output=False):
    # Разбиваем пути на части для многопроцессорной обработки
    chunk_size = len(file_paths) // cpus
    chunks = [file_paths[i:i + chunk_size] for i in range(0, len(file_paths), chunk_size)]

    if output:
        # Запускаем method параллельно
        with multiprocessing.Pool(processes=cpus) as pool:
            results = pool.map(method, chunks)

        return results
    else:
        with multiprocessing.Pool(processes=cpus) as pool:
            pool.map(method, chunks)

import json
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import quad


def create_new_gh_json(input_path, output_path, n_points=10):

    def _get_uniform_arc_points(tck, n_points=10):
        """Возвращает n_points, равномерно распределённых по длине кривой"""

        def _arc_length(tck, u_start=0.0, u_end=1.0):
            """Вычисляет длину сплайна между двумя параметрами u"""

            def _integrand(u):
                dx, dy, dz = splev(u, tck, der=1)
                return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            length, _ = quad(_integrand, u_start, u_end)
            return length

        total_length = _arc_length(tck)

        # Найдём точки по длине (не по параметру!)
        target_lengths = np.linspace(0, total_length, n_points)

        # Сопоставим длине параметры u методом бисекции
        u_vals = [0.0]
        for s in target_lengths[1:]:
            # Бинарный поиск параметра u, дающего длину ≈ s
            u_low, u_high = 0.0, 1.0
            for _ in range(20):
                u_mid = (u_low + u_high) / 2.0
                length_mid = _arc_length(tck, 0.0, u_mid)
                if length_mid < s:
                    u_low = u_mid
                else:
                    u_high = u_mid
            u_vals.append((u_low + u_high) / 2.0)

        points = np.vstack(splev(u_vals, tck)).T
        return points.tolist()

    with open(input_path, 'r') as f:
        data = json.load(f)

    needs_keys = ["RGH", "LGH", "NGH"]
    result = {}
    for key, point_list in data.items():
        if not key in needs_keys:
            continue
        points = np.array(point_list)
        if len(points) < 2:
            continue  # Нельзя построить кривую

        # Построим сплайн
        s = 0.1 * len(points)
        tck, _ = splprep(points.T, s=0)

        # Получим равномерно распределённые точки по длине кривой
        sampled_points = _get_uniform_arc_points(tck, n_points=n_points)
        result[key] = sampled_points

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
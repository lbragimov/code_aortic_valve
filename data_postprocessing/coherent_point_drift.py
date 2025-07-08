import json
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import quad
from pathlib import Path
from pycpd import AffineRegistration, DeformableRegistration
from data_preprocessing.text_worker import add_info_logging


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


def find_new_curv(cur_case, list_train_cases, train_curv_folder, result_curv_folder, org_cases_folder):
    def _load_landmarks(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return np.array([data['R'], data['L'], data['N'], data['RLC'], data['RNC'], data['LNC'], data['GH']])

    def _load_curves(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return {k: np.array(v) for k, v in data.items()}

    def _find_deformable(source_points,target_points,
                         # beta_range=(0.5, 2.5, 0.1),  # (start, stop, step)
                         # beta_range=(0.5, 10.0, 0.5),
                         # alpha_range=(0.01, 0.3, 0.02),  # (start, stop, step)
                         # alpha_range=(0.05, 1.0, 0.05),
                         beta_range=(0.5, 3.0, 0.25),
                         alpha_range = (0.01, 0.3, 0.02),
                         mse_threshold=1.0, max_trials=600):

        best_TY = None
        best_reg = None
        best_mse = float("inf")
        is_acceptable = False
        best_alpha = None
        best_beta = None

        beta_vals = np.arange(*beta_range)
        alpha_vals = np.arange(*alpha_range)

        trial_count = 0

        for beta in beta_vals:
            for alpha in alpha_vals:
                if trial_count >= max_trials:
                    break

                try:
                    reg = DeformableRegistration(X=target_points, Y=source_points, beta=beta, alpha=alpha)
                    TY, _ = reg.register()
                    mse = np.mean(np.sum((TY - target_points) ** 2, axis=1))

                    if mse < best_mse:
                        best_mse = mse
                        best_TY = TY
                        best_reg = reg
                        is_acceptable = mse < mse_threshold

                    if mse < mse_threshold:
                        return best_TY, best_reg, True

                except Exception:
                    continue

                trial_count += 1

        return best_TY, best_reg, is_acceptable

    def _find_affine(source_points, target_points):
        reg_aff = AffineRegistration(X=target_points, Y=source_points)
        TY_aff, _ = reg_aff.register()
        return TY_aff, reg_aff

    def _apply_deformable_transform(points, reg):
        # return reg.transform_point_cloud(Y=curve)
        # Исходные точки, на которых обучалась деформация
        Y = reg.Y
        TY = reg.TY  # трансформированные исходные точки

        # Смещения
        D = TY - Y  # (N, 3)

        # Гауссовское ядро между новыми точками и Y
        def gaussian_kernel(x, y, beta):
            diff = x[:, None, :] - y[None, :, :]  # (M, N, 3)
            dist2 = np.sum(diff ** 2, axis=2)  # (M, N)
            return np.exp(-dist2 / (2 * beta ** 2))

        G = gaussian_kernel(points, Y, reg.beta)  # (M, N)
        curve_transformed = points + G @ D  # (M, 3)
        return curve_transformed

    def _apply_affine_transform(points, reg):
        A = reg.B  # матрица преобразования (например, 3x3 в 3D)
        t = reg.t  # вектор сдвига
        return points @ A.T + t

    def _average_curves(curves_list):
        return np.mean(np.stack(curves_list), axis=0)

    test_pts = _load_landmarks(cur_case)
    # org_pts = _load_landmarks(Path(org_cases_folder) / cur_case.name)
    collected_curves = {"RGH": [], "LGH": [], "NGH": []}
    weights = []
    errors = []  # сюда будем сохранять ошибки
    results_meta = [] # для сохранения информации по каждому кейсу

    for train_case in list_train_cases:
        train_pts = _load_landmarks(train_case)
        train_curves = _load_curves(Path(train_curv_folder) / train_case.name)

        try:
            TY_def, reg_def, is_acceptable = _find_deformable(train_pts, test_pts)
            if not is_acceptable:
                add_info_logging(f"CPD failed for {train_case.name}", "work_logger")
            # TY_aff, reg_aff = _find_affine(org_pts, TY_def)
        except Exception as e:
            add_info_logging(f"CPD failed for {train_case.name}: {e}", "work_logger")
            continue

        # # Сохраняем параметры для этого кейса
        # results_meta.append({
        #     "case_name": train_case.name,
        #     "beta": beta,
        #     "alpha": alpha
        # })

        # Вычисляем ошибку между трансформированными точками и тестовыми
        # diff = TY - test_pts
        diff = TY_def - test_pts
        mse = np.mean(np.square(diff))
        # errors.append(mse)
        # weight = 1.0 / (mse + 1e-6)
        alpha = 0.05
        weight = np.exp(-alpha * np.array(mse))
        weights.append(weight)

        for name in collected_curves.keys():
            transformed = _apply_deformable_transform(train_curves[name], reg_def)
            # transformed_finish = _apply_affine_transform(transformed, reg_aff)
            collected_curves[name].append(transformed)
            # collected_curves[name].append(transformed_finish)

    # Нормализация весов
    weights = np.array(weights)
    weights /= weights.sum()

    # # Преобразуем ошибки в веса
    # errors = np.array(errors)
    # weights = 1.0 / (errors + 1e-6)  # обратные веса
    # weights /= weights.sum()  # нормализация

    # Взвешенное усреднение
    averaged = {}
    for name, curves_list in collected_curves.items():
        stacked_curves = np.stack(curves_list)
        weighted_avg = np.tensordot(weights, stacked_curves, axes=1)
        averaged[name] = weighted_avg

    # averaged = {name: _average_curves(curves) for name, curves in collected_curves.items()}

    result = {}
    for key, point_coord in averaged.items():
        if isinstance(point_coord, np.ndarray):
            point_coord = point_coord.tolist()
        result[key] = point_coord

    with open(Path(result_curv_folder) / cur_case.name, 'w') as f:
        json.dump(result, f, indent=4)

    # df = pd.DataFrame(results_meta)
    # csv_path = Path(result_curv_folder) / f"{cur_case.stem}_deformable_params.csv"
    # df.to_csv(csv_path, index=False)

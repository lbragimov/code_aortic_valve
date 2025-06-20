from data_preprocessing.text_worker import add_info_logging
import os
from pathlib import Path
import json
import numpy as np
from pycpd import AffineRegistration


def find_new_curv(cur_case, list_train_cases, train_curv_folder, result_curv_folder):
    def _load_landmarks(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return np.array([data['R'], data['L'], data['N'], data['RLC'], data['RNC'], data['LNC']])

    def _load_curves(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return {k: np.array(v) for k, v in data.items()}

    def _apply_cpd(source_points, target_points):
        reg = AffineRegistration(X=target_points, Y=source_points)
        TY, _ = reg.register()
        return TY, reg

    def _apply_transform_to_curve(curve, reg):
        # Для affine: просто домножаем на матрицу и прибавляем сдвиг
        A = reg.B
        t = reg.t
        return curve @ A.T + t

    def _average_curves(curves_list):
        return np.mean(np.stack(curves_list), axis=0)

    test_pts = _load_landmarks(cur_case)
    collected_curves = {"RGH": [], "LGH": [], "NGH": []}

    for train_case in list_train_cases:
        train_pts = _load_landmarks(train_case)
        train_curves = _load_curves(Path(train_curv_folder) / train_case.name)

        _, reg = _apply_cpd(train_pts, test_pts)

        for name in collected_curves.keys():
            transformed = _apply_transform_to_curve(train_curves[name], reg)
            collected_curves[name].append(transformed)

    averaged = {name: _average_curves(curves) for name, curves in collected_curves.items()}

    result = {}
    for key, point_coord in averaged.items():
        if isinstance(point_coord, np.ndarray):
            point_coord = point_coord.tolist()
        result[key] = point_coord

    with open(Path(result_curv_folder) / cur_case.name, 'w') as f:
        json.dump(result, f, indent=4)



def controller(data_path):
    ds_folder_name = "Dataset499_AortaLandmarks"
    json_duplication_g_h_path = os.path.join(data_path, "json_duplication_geometric_heights")
    json_land_mask_coord_path = os.path.join(data_path, "json_landmarks_mask_coord")
    test_cases_path = os.path.join(json_land_mask_coord_path, "test")
    train_cases_path = os.path.join(json_land_mask_coord_path, "train")
    list_test_cases = list(Path(test_cases_path).glob("*.json"))
    list_train_cases = list(Path(train_cases_path).glob("*.json"))
    test_curv_cases_path = os.path.join(json_duplication_g_h_path, "test")
    train_curv_cases_path = os.path.join(json_duplication_g_h_path, "train")
    result_curv_cases_path = os.path.join(json_duplication_g_h_path, "result")
    for cur_test_case in list_test_cases:
        find_new_curv(cur_test_case, list_train_cases, train_curv_cases_path, result_curv_cases_path)
    add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)
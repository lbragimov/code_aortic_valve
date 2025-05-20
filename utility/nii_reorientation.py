import json
import io
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
from pathlib import Path
import numpy as np
import SimpleITK as sitk


# --- Перехват print в лог ---
print_buffer = io.StringIO()
original_stdout = sys.stdout

class PrintLogger:
    def write(self, message):
        print_buffer.write(message)
        original_stdout.write(message)
    def flush(self):
        pass

sys.stdout = PrintLogger()

# --- Показ лога и завершение ---
def show_log_and_exit(timeout_seconds=20):
    """
    Показывает все print-сообщения в окне и завершает скрипт через timeout_seconds.
    """
    log_text = print_buffer.getvalue()

    def close_after_timeout():
        root.after(timeout_seconds * 1000, root.destroy)

    root = tk.Tk()
    root.title("Лог выполнения скрипта")

    text_area = scrolledtext.ScrolledText(root, width=100, height=30)
    text_area.insert(tk.END, log_text)
    text_area.config(state=tk.DISABLED)
    text_area.pack(padx=10, pady=10)

    close_after_timeout()
    root.mainloop()
    sys.exit()


def reorient_image_by_plane(image_path, points_dict):
    """
    Поворачивает 3D-изображение так, чтобы ось Z была направлена по нормали к плоскости,
    заданной тремя точками A, B, C (в индексных координатах).

    :param image: SimpleITK изображение
    :param points_dict: словарь {'A': [i, j, k], 'B': [...], 'C': [...]}
    :return: повернутое изображение (SimpleITK Image)
    """
    image = sitk.ReadImage(image_path)

    # Переводим индексные координаты в физические
    A_phys = np.array(image.TransformIndexToPhysicalPoint(list(map(int, points_dict['RLC']))))
    B_phys = np.array(image.TransformIndexToPhysicalPoint(list(map(int, points_dict['RNC']))))
    C_phys = np.array(image.TransformIndexToPhysicalPoint(list(map(int, points_dict['LNC']))))

    # Векторы в плоскости
    v1 = B_phys - A_phys
    v2 = C_phys - A_phys

    # Ортонормальная система координат
    z_axis = np.cross(v1, v2)
    z_axis /= np.linalg.norm(z_axis)

    x_axis = v1 / np.linalg.norm(v1)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Переортогонализация x_axis (на случай неточного вычисления)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Сборка матрицы поворота (столбцы — новые оси x, y, z)
    R = np.vstack([x_axis, y_axis, z_axis]).T  # shape (3, 3)

    # Центр поворота — центр треугольника ABC
    center_point = (A_phys + B_phys + C_phys) / 3.0

    # Создаём аффинную трансформацию
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(R.flatten())
    affine.SetCenter(A_phys.tolist())  # Центр поворота — точка A

    # Ресэмплинг
    resampled = sitk.Resample(
        image,
        image,  # та же сетка, просто повёрнутая
        affine.GetInverse(),  # обратное преобразование!
        sitk.sitkLinear,
        0.0,
        image.GetPixelID()
    )

    return resampled


def save_image_with_same_name(image, original_path, output_dir):
    """
    Сохраняет изображение с тем же именем, что и у original_path, но в другую папку.

    :param image: SimpleITK.Image — изображение для сохранения
    :param original_path: str — путь к исходному файлу (чтобы взять имя)
    :param output_dir: str — путь к выходной папке
    """
    filename = os.path.basename(original_path)  # например: HOM_M19_H217_W96_YA.nii.gz
    output_path = os.path.join(output_dir, filename)

    sitk.WriteImage(image, output_path)
    print(f"Save: {output_path}")


def select_nii_file():
    # Скрываем основное окно Tkinter
    root = tk.Tk()
    root.withdraw()

    # 1. Выбор папки
    folder_path = filedialog.askdirectory(title="Select folder")

    if folder_path:
        # 2. Выбор файла в этой папке
        file_path = filedialog.askopenfilename(
            title="Select a file in the selected folder",
            initialdir=folder_path,
            filetypes=(("nii", "*.nii.gz;*.nii"),)  # можно задать фильтры
        )

        if file_path:
            return file_path
        else:
            print("File not selected")
            show_log_and_exit()
    else:
        print("folder not selected")
        show_log_and_exit()


def get_file_name(file_path):
    filename = Path(file_path).name

    # Удаляем расширение .nii или .nii.gz
    if filename.endswith('.nii.gz'):
        filename = filename[:-7]
    elif filename.endswith('.nii'):
        filename = filename[:-4]

    # Удаляем префикс до первой буквы H, n или p
    match = re.search(r'[Hnp].*', filename)
    if match:
        filename = match.group(0)
    else:
        print("Внимание: имя файла не содержит ожидаемых символов 'H', 'n' или 'p'.")

    # Удаляем окончание _0000, если есть
    if filename.endswith('_0000'):
        filename = filename[:-5]

    return filename


def get_json_dict(json_folder_path, nii_file_name):
    def read_json(json_path):
        if os.path.exists(json_path):
            with open(json_path, 'r') as read_file:
                return json.load(read_file)
        else:
            print("A file with that name and extension was not found in the json folder.")
            show_log_and_exit()

    if nii_file_name[0] == "H":
        json_path = os.path.join(json_folder_path, "Homburg pathology", f"{nii_file_name}.json")
        return read_json(json_path)
    elif nii_file_name[0] == "n":
        json_path = os.path.join(json_folder_path, "Normal", f"{nii_file_name}.json")
        return read_json(json_path)
    elif nii_file_name[0] == "p":
        json_path = os.path.join(json_folder_path, "Pathology", f"{nii_file_name}.json")
        return read_json(json_path)
    else:
        print("A file with that name and extension was not found in the json folder.")
        show_log_and_exit()


def controller(data_path):
    result_path = os.path.join(data_path, "result")
    json_folder_path = os.path.join(data_path, "json_markers_info")
    output_folder_path = os.path.join(data_path, "temp_reorient_nii_images")
    nii_file_path = select_nii_file()
    input_nii_file_name = get_file_name(nii_file_path)
    points_cord = get_json_dict(json_folder_path=json_folder_path,
                                nii_file_name=input_nii_file_name)
    reorient_nii_img = reorient_image_by_plane(image_path=nii_file_path, points_dict=points_cord)
    save_image_with_same_name(image=reorient_nii_img, original_path=nii_file_path, output_dir=output_folder_path)
    print("Done! The image is reoriented and saved.")
    show_log_and_exit()


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)
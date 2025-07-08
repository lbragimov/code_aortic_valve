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
import nibabel as nib


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

    # if nii_file_name[0] == "H":
    #     json_path = os.path.join(json_folder_path, "Homburg pathology", f"{nii_file_name}.json")
    #     return read_json(json_path)
    # elif nii_file_name[0] == "n":
    #     json_path = os.path.join(json_folder_path, "Normal", f"{nii_file_name}.json")
    #     return read_json(json_path)
    # elif nii_file_name[0] == "p":
    #     json_path = os.path.join(json_folder_path, "Pathology", f"{nii_file_name}.json")
    #     return read_json(json_path)
    # else:
    #     print("A file with that name and extension was not found in the json folder.")
    #     show_log_and_exit()
    # return read_json(os.path.join("C:/Users/Kamil/Aortic_valve/data/json_duplication_geometric_heights/test",
    #                               f"{nii_file_name}.json"))
    return read_json(os.path.join("C:/Users/Kamil/Aortic_valve/data/json_duplication_geometric_heights/result",
                                  f"{nii_file_name}.json"))


# === Функция: мировые координаты -> воксельные индексы ===
def world_to_voxel_sitk(point, image):
    return image.TransformPhysicalPointToIndex(point)

# === Добавление точек в маску ===
def add_point_to_mask_sitk(mask, voxel_coord, radius=3):
    x, y, z = voxel_coord
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            for k in range(-radius, radius + 1):
                if i**2 + j**2 + k**2 <= radius**2:
                    xi, yi, zi = x + i, y + j, z + k
                    if (0 <= xi < mask.GetWidth() and
                        0 <= yi < mask.GetHeight() and
                        0 <= zi < mask.GetDepth()):
                        mask[xi, yi, zi] = 1


def controller(data_path):
    result_path = os.path.join(data_path, "result")
    # json_folder_path = os.path.join(data_path, "json_markers_info")
    # json_folder_path = os.path.join(data_path, "json_duplication_geometric_heights/test")
    json_folder_path = os.path.join(data_path, "json_duplication_geometric_heights/result")
    output_folder_path = os.path.join(data_path, "temp_all_points_mask")
    nii_file_path = select_nii_file()
    input_nii_file_name = get_file_name(nii_file_path)
    points_cord = get_json_dict(json_folder_path=json_folder_path,
                                nii_file_name=input_nii_file_name)

    radius = 1  # радиус сферы вокселей

    # === Загрузка данных ===
    img = sitk.ReadImage(nii_file_path)
    mask_shape = img.GetSize()  # (x, y, z)
    mask = sitk.Image(img.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(img)  # сохраняем направление, origin, spacing

    # Создание 4 масок
    mask_dict = {
        "group1": sitk.Image(mask_shape, sitk.sitkUInt8),  # R, L, N, RLC, RNC, LNC
        "group2": sitk.Image(mask_shape, sitk.sitkUInt8),  # BR - closed
        "group3": sitk.Image(mask_shape, sitk.sitkUInt8),  # RGH, LGH, NGH
        "group4": sitk.Image(mask_shape, sitk.sitkUInt8),  # RCI, LCI, NCI
        "group5": sitk.Image(mask_shape, sitk.sitkUInt8),
    }

    # Копирование информации
    for m in mask_dict.values():
        m.CopyInformation(img)

    key_to_group = {
        "R": "group1", "L": "group1", "N": "group1",
        "RLC": "group1", "RNC": "group1", "LNC": "group1",
        "BR - closed": "group2",
        "RGH": "group3", "LGH": "group3", "NGH": "group3",
        "RCI": "group4", "LCI": "group4", "NCI": "group4",
        "RLS - closed": "group5", "RNS - closed": "group5", "LNS - closed": "group5"
    }

    # Заполнение масок
    for key, value in points_cord.items():
        if key not in key_to_group:
            print(f"❗ Пропущен ключ: {key}")
            continue

        group = key_to_group[key]
        target_mask = mask_dict[group]

        points = value if isinstance(value[0], list) else [value]

        for point in points:
            try:
                voxel = world_to_voxel_sitk(point, img)
                add_point_to_mask_sitk(target_mask, voxel, radius)
            except Exception as e:
                print(f"⚠️ Ошибка в точке {point}: {e}")

    # Сохранение всех масок
    base_name = os.path.basename(nii_file_path).replace('.nii.gz', '').replace('.nii', '')
    for group, mask in mask_dict.items():
        out_path = os.path.join(output_folder_path, f"{base_name}_{group}.nii.gz")
        sitk.WriteImage(mask, out_path)
        print(f"✅ Сохранено: {out_path}")

    print("✅ Готово. Все маски сохранены.")


    # save_image_with_same_name(image=reorient_nii_img, original_path=nii_file_path, output_dir=output_folder_path)
    print("Done! The image is reoriented and saved.")
    show_log_and_exit()


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)
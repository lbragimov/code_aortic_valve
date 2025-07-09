import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import os


def controller(data_path):
    csv_path = os.path.join(data_path, "result/cases_info.csv")
    # Загрузка CSV
    df = pd.read_csv(csv_path)

    # Преобразуем строковые представления кортежей в списки
    df["ImageSpacing"] = df["ImageSpacing"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Преобразуем в numpy-массив N×3
    spacing_array = np.vstack(df["ImageSpacing"].values)

    # Разделяем по осям
    x_spacing = spacing_array[:, 0]
    y_spacing = spacing_array[:, 1]
    z_spacing = spacing_array[:, 2]

    # Выводим статистику
    print("Image Spacing Statistics:")
    for axis, values in zip(["X", "Y", "Z"], [x_spacing, y_spacing, z_spacing]):
        print(f"  Axis {axis}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}, std={values.std():.4f}")

    # Визуализация
    plt.figure(figsize=(12, 4))
    for i, (spacing, label) in enumerate(zip([x_spacing, y_spacing, z_spacing], ["X", "Y", "Z"])):
        plt.subplot(1, 3, i + 1)
        plt.hist(spacing, bins=20, color='skyblue', edgecolor='black')
        plt.title(f"Spacing {label}-axis")
        plt.xlabel("Spacing (mm)")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
    print("Finish")

if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)
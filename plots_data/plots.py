import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==== 1. Загрузка данных ====
file_path = "gnn_vs_center_comparison.csv"  # путь к твоему файлу
df = pd.read_csv(file_path)

# Проверка наличия нужных столбцов
required_cols = ["measurement", "abs_err_gnn", "abs_err_center"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"В файле нет столбца: {col}")

# ==== 2. Подготовка данных ====
measurements = sorted(df["measurement"].unique())

data = []
positions = []
labels = []

pos = 1
for m in measurements:
    subset = df[df["measurement"] == m]

    gnn_vals = subset["abs_err_gnn"].dropna().values
    center_vals = subset["abs_err_center"].dropna().values

    data.append(gnn_vals)
    data.append(center_vals)

    positions.append(pos)
    positions.append(pos + 1)

    labels.append(f"{m}\nGNN")
    labels.append(f"{m}\nCenter")

    pos += 3  # расстояние между группами

# ==== 3. Построение графика ====
plt.figure(figsize=(14, 6))
plt.boxplot(data, positions=positions)
plt.xticks(positions, labels, rotation=45, ha="right")
plt.ylabel("Absolute Error")
plt.title("GNN vs Center Absolute Error per Measurement")
plt.tight_layout()

# ==== 4. Сохранение ====
plt.savefig("gnn_vs_center_boxplot.png", dpi=300)
plt.show()

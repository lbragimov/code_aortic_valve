import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os


def _save_combined_mean_sd(df, measurements, save_txt_path):
    """
    Считает mean ± sd для объединённого списка measurements
    и сохраняет результат в txt.
    """

    # фильтрация по списку measurement
    subset = df[df["measurement"].isin(measurements)]

    if subset.empty:
        raise ValueError("Нет данных для переданных measurement.")

    gnn_vals = subset["abs_err_gnn"].dropna()
    center_vals = subset["abs_err_center"].dropna()

    if len(gnn_vals) == 0 and len(center_vals) == 0:
        raise ValueError("Нет числовых значений для расчёта.")

    gnn_mean = gnn_vals.mean()
    gnn_sd = gnn_vals.std()

    center_mean = center_vals.mean()
    center_sd = center_vals.std()

    # --- запись в txt ---
    with open(save_txt_path, "w", encoding="utf-8") as f:

        f.write("Combined measurements\n")
        f.write(f"Number of rows: {len(subset)}\n\n")

        f.write("GNN (mean ± sd): ")
        f.write(f"{gnn_mean:.3f} ± {gnn_sd:.3f}\n")

        f.write("Center (mean ± sd): ")
        f.write(f"{center_mean:.3f} ± {center_sd:.3f}\n")


def _create_summary_table_plot(df, parameter_keys, save_path, save_txt_path, title):

    rows = []

    for m in parameter_keys:
        subset = df[df["measurement"] == m]

        if subset.empty:
            continue

        gnn_vals = subset["abs_err_gnn"].dropna()
        center_vals = subset["abs_err_center"].dropna()

        if len(gnn_vals) == 0 and len(center_vals) == 0:
            continue

        gnn_mean = gnn_vals.mean()
        gnn_sd = gnn_vals.std()

        center_mean = center_vals.mean()
        center_sd = center_vals.std()

        label = parameter_keys[m][0]
        full_name = parameter_keys[m][1]

        rows.append([
            label,
            full_name,
            f"{gnn_mean:.2f} ± {gnn_sd:.2f}",
            f"{center_mean:.2f} ± {center_sd:.2f}"
        ])

    # --- создаём figure ---
    fig, ax = plt.subplots(figsize=(7, 0.21 * len(rows) + 0.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Label", "Measurement", "GNN\n(mean ± sd)", "Center of mass\n(mean ± sd)"],
        loc="center",
        #cellLoc="left"
        bbox=[0, 0, 1, 1]  # растянуть на всю область
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(4)))

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(ha='center')
        else:
            cell.set_text_props(ha='left')
            cell.PAD = 0.05

    # увеличить высоту строки заголовков
    for col in range(4):
        header_cell = table[(0, col)]
        header_cell.set_height(header_cell.get_height() * 1.8)

    # plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # --- сохранение в txt ---
    with open(save_txt_path, "w", encoding="utf-8") as f:

        # заголовки
        headers = ["Label", "Measurement", "GNN (mean ± sd)", "Center of mass (mean ± sd)"]
        f.write("\t".join(headers) + "\n")

        # строки
        for row in rows:
            f.write("\t".join(row) + "\n")


def _plot_group(df, parameter_keys, save_path, title):
    data = []
    positions = []
    labels = []
    group_centers = []
    separator_positions = []

    pos = 1
    box_widths = 0.2 # ширина бара
    pair_spacing = 0.3  # расстояние внутри пары
    group_spacing = 0.4  # расстояние между группами
    width_per_group = 0.5  # ширина на одну пару (в дюймах)
    base_margin = 1.2  # боковые поля
    fixed_height = 6

    for m in parameter_keys.keys():
        labels.append(parameter_keys[m][0])  # берём label
        subset = df[df["measurement"] == m]

        if subset.empty:
            continue

        gnn_vals = subset["abs_err_gnn"].dropna().values
        center_vals = subset["abs_err_center"].dropna().values

        if len(gnn_vals) == 0 and len(center_vals) == 0:
            continue

        data.append(gnn_vals)
        data.append(center_vals)

        positions.append(pos)
        positions.append(pos + pair_spacing)

        # центр группы (между двумя боксами)
        group_centers.append(pos + pair_spacing / 2)

        separator_positions.append(pos + pair_spacing + group_spacing / 2)

        pos += pair_spacing + group_spacing

    fig_width = len(labels) * width_per_group + base_margin
    fig, ax = plt.subplots(figsize=(fig_width, fixed_height))

    box = ax.boxplot(
        data,
        positions=positions,
        widths=box_widths,
        patch_artist=True,
        showmeans=False,
        meanline=True,
        showfliers=False,
        medianprops = dict(color="black", linewidth=1.5)
    )

    # --- покраска ---
    for i, patch in enumerate(box["boxes"]):
        if i % 2 == 0:
            patch.set_facecolor("#4C72B0")  # GNN
        else:
            patch.set_facecolor("#DD8452")  # Center

    # --- разделительные линии ---
    for sep in separator_positions[:-1]:
        ax.axvline(sep, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    # --- легенда ---
    legend_elements = [
        Patch(facecolor="#4C72B0", label="GNN"),
        Patch(facecolor="#DD8452", label="Center of mass")
    ]
    ax.legend(handles=legend_elements)

    # --- подписи оси X ---
    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels=labels)#, rotation=45, ha="right")

    # ax.set_ylabel("Absolute Error")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def creat_box_plot(result_path, file_name):
    data_file_path = os.path.join(result_path, file_name)
    plot_length_img_path = os.path.join(result_path, 'boxplot_length.png')
    plot_angle_img_path = os.path.join(result_path, 'boxplot_angle.png')
    length_table_path = os.path.join(result_path, "table_length.png")
    angle_table_path = os.path.join(result_path, "table_angle.png")
    length_txt_path = os.path.join(result_path, "table_length.txt")
    angle_txt_path = os.path.join(result_path, "table_angle.txt")
    angle_mean_txt_path = os.path.join(result_path, "mean_angle.txt")
    df = pd.read_csv(str(data_file_path))

    # measurements = sorted(df["measurement"].unique())

    length_par_dict ={
        'IC_R': ["lm1","Right intercommissural distance"],
        'IC_L': ["lm2", "Left intercommissural distance"],
        'IC_N': ["lm3", "Non-coronary intercommissural distance"],
        'IC_distance': ["lm4", "Mean intercommissural distance"],
        'BR_perimeter': ["lm5", "Basal ring perimeter"],
        'BR_max': ["lm6", "Maximum basal ring diameter"],
        'BR_min': ["lm7", "Minimum basal ring diameter"],
        'BR_diameter': ["lm8", "Mean basal ring diameter"],
        'RL_comm_height': ["lm9", "Commissural height between right and left leaflets"],
        'RN_comm_height': ["lm10", "Commissural height between right and non-coronary leaflets"],
        'LN_comm_height': ["lm11", "Commissural height between left and non-coronary leaflets"],
        'mean_comm_heigh': ["lm12", "Mean commissural height"],
        'ST_perimeter': ["lm13", "Sino-tubular junction perimeter"],
        'ST_max': ["lm14", "Maximum STJ diameter"],
        'ST_min': ["lm15", "Minimum STJ diameter"],
        'ST_diameter': ["lm16", "Mean STJ diameter"],
        'commissural_diameter': ["lm17", "Commissural diameter"],
        'centroid_valve_height': ["lm18", "Centroid valve height"]
    }

    angle_par_dict = {
        'R_flat_angle': ["am1", "Right leaflet flat angle"],
        'L_flat_angle': ["am2", "Left leaflet flat angle"],
        'N_flat_angle': ["am3", "Non-coronary leaflet flat angle"],
        'R_vertical_angle': ["am4", "Right leaflet vertical angle"],
        'L_vertical_angle': ["am5", "Left leaflet vertical angle"],
        'N_vertical_angle': ["am6", "Non-coronary leaflet vertical angle"],
        'mean_vertical_angle': ["am7", "Mean vertical leaflet angle"],
        'RL_angle': ["am8", "Angle between right and left leaflets"],
        'RN_angle': ["am9", "Angle between right and non-coronary leaflets"],
        'LN_angle': ["am10", "Angle between left and non-coronary leaflets"],
        'BR_C_plane_angle': ["am11", "Angle between the basal ring plane and the commissural plane"]
    }
    # 'mean_commissural_angle': "Mean commissural angle",

    angle_list = [
        'R_flat_angle', 'L_flat_angle', 'N_flat_angle', 'R_vertical_angle', 'L_vertical_angle', 'N_vertical_angle',
        'RL_angle', 'RN_angle', 'LN_angle', 'BR_C_plane_angle'
    ]

    _save_combined_mean_sd(df, angle_list, angle_mean_txt_path)

    _plot_group(df,
                length_par_dict,
                plot_length_img_path,
               "Length Measurements, mm")

    _plot_group(df,
                angle_par_dict,
                plot_angle_img_path,
               "Angle Measurements, °")

    _create_summary_table_plot(df, length_par_dict,
                               length_table_path,
                               length_txt_path,
                               "Length Measurements")

    _create_summary_table_plot(df, angle_par_dict,
                               angle_table_path,
                               angle_txt_path,
                               "Angle Measurements")

if __name__ == "__main__":

    result_path = r'C:\Users\Kamil\Aortic_valve\data\gnn_folder\results'
    file_name = "gnn_vs_center_comparison.csv"
    creat_box_plot(result_path, file_name)

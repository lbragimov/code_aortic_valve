import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os


def _create_summary_table_plot(df, parameter_keys, save_path, title):

    rows = []

    for m in sorted(parameter_keys):
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

        rows.append([
            m,
            f"{gnn_mean:.3f} ± {gnn_sd:.3f}",
            f"{center_mean:.3f} ± {center_sd:.3f}"
        ])

    # --- создаём figure ---
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 2))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Measurement", "GNN (mean ± sd)", "Center (mean ± sd)"],
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(3)))

    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_group(df, parameter_keys, save_path, title):
    data = []
    positions = []
    # labels = []
    group_centers = []

    pos = 1

    for m in sorted(parameter_keys):
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
        positions.append(pos + 1)

        # labels.append(f"{m}\nGNN")
        # labels.append(f"{m}\nCenter")

        # центр группы (между двумя боксами)
        group_centers.append(pos + 0.5)

        pos += 3

    # plt.figure(figsize=(14, 6))
    # plt.boxplot(data, positions=positions)
    # plt.xticks(positions, labels, rotation=45, ha="right")
    # plt.ylabel("Absolute Error")
    # plt.title(title)
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300)
    # plt.close()

    fig, ax = plt.subplots(figsize=(14, 6))

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanline=True
    )

    # --- покраска ---
    for i, patch in enumerate(box["boxes"]):
        if i % 2 == 0:
            patch.set_facecolor("#4C72B0")  # GNN
        else:
            patch.set_facecolor("#DD8452")  # Center

    # --- легенда ---
    legend_elements = [
        Patch(facecolor="#4C72B0", label="GNN"),
        Patch(facecolor="#DD8452", label="Center")
    ]
    ax.legend(handles=legend_elements)

    # --- подписи оси X ---
    ax.set_xticks(group_centers)
    ax.set_xticklabels(sorted(parameter_keys), rotation=45, ha="right")

    ax.set_ylabel("Absolute Error")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def creat_box_plot(result_path, file_name):
    data_file_path = os.path.join(result_path, file_name)
    plot_length_img_path = os.path.join(result_path, 'gnn_vs_center_boxplot_length.png')
    plot_angle_img_path = os.path.join(result_path, 'gnn_vs_center_boxplot_angle.png')
    length_table_path = os.path.join(result_path, "length_summary_table.png")
    angle_table_path = os.path.join(result_path, "angle_summary_table.png")
    df = pd.read_csv(str(data_file_path))

    measurements = sorted(df["measurement"].unique())

    length_par_dict ={
        'IC_R': "Right intercommissural distance",
        'IC_L': "Left intercommissural distance",
        'IC_N': "Non-coronary intercommissural distance",
        'IC_distance': "Mean intercommissural distance",
        'BR_perimeter': "Basal ring perimeter",
        'BR_max': "Maximum basal ring diameter",
        'BR_min': "Minimum basal ring diameter",
        'BR_diameter': "Mean basal ring diameter",
        'RL_comm_height': "Commissural height between right and left leaflets",
        'RN_comm_height': "Commissural height between right and non-coronary leaflets",
        'LN_comm_height': "Commissural height between left and non-coronary leaflets",
        'mean_comm_heigh': "Mean commissural height",
        'ST_perimeter': "Sino-tubular junction perimeter",
        'ST_max': "Maximum STJ diameter",
        'ST_min': "Minimum STJ diameter",
        'ST_diameter': "Mean STJ diameter",
        'commissural_diameter': "Commissural diameter",
        'centroid_valve_height': "Centroid valve height"
    }

    angle_par_dict = {
        'R_flat_angle': "Right leaflet flat angle",
        'L_flat_angle': "Left leaflet flat angle",
        'N_flat_angle': "Non-coronary leaflet flat angle",
        'R_vertical_angle': "Right leaflet vertical angle",
        'L_vertical_angle': "Left leaflet vertical angle",
        'N_vertical_angle': "Non-coronary leaflet vertical angle",
        'mean_vertical_angle': "Mean vertical leaflet angle",
        'RL_angle': "Angle between right and left leaflets",
        'RN_angle': "Angle between right and non-coronary leaflets",
        'LN_angle': "Angle between left and non-coronary leaflets",
        'BR_C_plane_angle': "Angle between the basal ring plane and the commissural plane"
    }
    # 'mean_commissural_angle': "Mean commissural angle",

    _plot_group(df,
                length_par_dict.keys(),
                plot_length_img_path,
               "GNN vs Center Absolute Error (Length Measurements)")

    _plot_group(df,
                angle_par_dict.keys(),
                plot_angle_img_path,
               "GNN vs Center Absolute Error (Angle Measurements)")

    _create_summary_table_plot(df, length_par_dict.keys(),
                               length_table_path,
                               "Length Measurements")

    _create_summary_table_plot(df, angle_par_dict.keys(),
                               angle_table_path,
                               "Angle Measurements")

if __name__ == "__main__":

    result_path = r'C:\Users\Kamil\Aortic_valve\data\gnn_folder\results'
    file_name = "gnn_vs_center_comparison.csv"
    creat_box_plot(result_path, file_name)

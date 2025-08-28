import os
import json
import pandas as pd

def collect_point_counts(data):
    """
    Собирает количество точек для каждого case/ключа в виде словаря.
    """
    rows = []
    for case_name, case_data in data.items():
        row = {"case": case_name}
        for key, points in case_data.items():
            if isinstance(points, list):  # только массивы координат
                row[key] = len(points)
        rows.append(row)
    return pd.DataFrame(rows).set_index("case")


def validate_point_counts(data, save_csv=None):
    df = collect_point_counts(data)

    # эталонное количество точек для каждого ключа (берём первую строку)
    reference_counts = df.iloc[0].to_dict()

    errors = []
    for col in df.columns:
        expected = reference_counts[col]
        mismatches = df[df[col] != expected]
        if not mismatches.empty:
            for case_name, count in mismatches[col].items():
                errors.append(
                    f"Case '{case_name}': key '{col}' has {count} points "
                    f"(expected {expected})"
                )

    # сохранить таблицу в CSV (если задан путь)
    if save_csv:
        df.to_csv(save_csv, encoding="utf-8", index=True)

    return df, reference_counts, errors

def main(data_path):
    dict_all_case_path = os.path.join(data_path, "dict_all_case.json")
    output_csv_file = os.path.join(data_path, "result", "dict_all_case_check.csv")
    with open(dict_all_case_path, "r") as f:
        data = json.load(f)

    df, ref_counts, errors = validate_point_counts(data, output_csv_file)

    print("Reference counts:")
    for key, cnt in ref_counts.items():
        print(f" - {key}: {cnt}")

    # print("\nTable of point counts:")
    # print(df)

    if errors:
        print("\nMismatches found:")
        for e in errors:
            print(" -", e)
    else:
        print("\nAll cases match the reference counts ✅")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    main(data_path)
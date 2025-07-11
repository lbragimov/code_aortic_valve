import json
import yaml
from pathlib import Path
import logging
import numpy as np
from datetime import datetime


def add_info_logging(text_info, type_logger="work_logger"):
    current_time = datetime.now()
    str_time = current_time.strftime("%H:%M")
    if type_logger == "work_logger":
        logger = logging.getLogger("work_logger")
        logger.info(f"time:  {str_time} {text_info}")
    elif type_logger == "result_logger":
        logger = logging.getLogger("result_logger")
        logger.info(f"time:  {str_time} {text_info}")


def json_reader(path):
    with open(path, 'r') as read_file:
        return json.load(read_file)


def json_save(current_dict, path):
    with open(path, 'w') as json_file:
        json.dump(current_dict, json_file)

def create_new_json(output_file, dict_data):
    labels = {
        1: "R",
        2: "L",
        3: "N",
        4: "RLC",
        5: "RNC",
        6: "LNC",
        7: "GH"
    }
    result = {}
    for key, point_coord in dict_data.items():
        if isinstance(point_coord, np.ndarray):
            point_coord = point_coord.tolist()
        result[labels[key]] = point_coord

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)


def yaml_reader(path):
    with open(path, "r") as read_file:
        return yaml.safe_load(read_file)


def yaml_save(current_dict, path):
    with open(path, "w") as file:
        yaml.dump(current_dict, file, default_flow_style=False, allow_unicode=True)


def txt_json_convert(txt_file_path: str, json_file_path: str):

    original_file_name = Path(txt_file_path).parts[-1]

    # Read the contents of a text file
    with open(txt_file_path, 'r') as file:
        current_key = None  # Variable to store the current section
        # Initialize an empty dictionary to store data
        data = {}

        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            elif line in original_file_name:
                continue
            elif "Failed" in line:
                data[current_key] = []
                continue

            # If the line contains the section name
            if line.isalpha() or "closed" in line:
                current_key = line
                data[current_key] = []
            else:
                # Convert the line with numbers to a list of numbers and add to the current section
                values = list(map(float, line.split()))
                # If the section has only one row of data, save as a list, otherwise as a list of lists
                data[current_key].append(values)

    # Convert single-element lists to one-dimensional arrays to match the structure of the second file
    for key in data:
        if len(data[key]) == 1:
            data[key] = data[key][0]  # If the section has one row, convert the list of lists to a list

    # Save the result to a .json file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    return data


def parse_txt_file(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    data = {}
    current_label = None
    points = []

    for line in lines[1:]:  # пропускаем первую строку — это имя кейса
        if not any(c.isdigit() for c in line):  # это заголовок (например, R, NCI и т.п.)
            if current_label and points:
                data[current_label] = points
            current_label = line
            points = []
        else:
            coords = list(map(float, line.split()))
            points.append(coords)

    if current_label and points:
        data[current_label] = points  # добавить последнюю группу

    return data

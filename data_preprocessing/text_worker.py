import json
import yaml
from pathlib import Path
import logging
from datetime import datetime


def add_info_logging(text_info):
    current_time = datetime.now()
    str_time = current_time.strftime("%d:%H:%M")
    logging.info(f"time:  {str_time} {text_info}")


def json_reader(path):
    with open(path, 'r') as read_file:
        return json.load(read_file)


def json_save(current_dict, path):
    with open(path, 'w') as json_file:
        json.dump(current_dict, json_file)


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

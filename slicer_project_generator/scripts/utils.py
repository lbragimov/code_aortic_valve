import os
import json


def json_reader(path):
    with open(path, 'r') as read_file:
        return json.load(read_file)


def json_save(current_dict, path):
    with open(path, 'w') as json_file:
        json.dump(current_dict, json_file)


def get_available_cases(cases_dir="cases"):
    if not os.path.exists(cases_dir):
        return []
    return [d for d in os.listdir(cases_dir) if os.path.isdir(os.path.join(cases_dir, d))]
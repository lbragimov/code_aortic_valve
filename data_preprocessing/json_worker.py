import json


def json_reader(path):
    with open(path, 'r') as read_file:
        return json.load(read_file)


def json_save(current_dict, path):
    with open(path, 'w') as json_file:
        json.dump(current_dict, json_file)

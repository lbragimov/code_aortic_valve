import json

from pathlib import Path


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
    print("hi")


if __name__ == "__main__":
    txt_file_path = "C:/Users/Kamil/Aortic_valve/data/marker_info/Homburg pathology/o_HOM_M19_H217_W96_YA.txt"
    json_file_path = "C:/Users/Kamil/Aortic_valve/data/HOM_M19_H217_W96_YA.json"
    txt_json_convert(txt_file_path, json_file_path)



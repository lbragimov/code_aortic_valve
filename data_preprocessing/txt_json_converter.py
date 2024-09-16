import json

# Read the file
file_path = "/mnt/data/o_HOM_M19_H217_W96_YA.txt"
data = {}


def txt_json_convert(dir: str, f_name: str):
    with open(dir + "/" + f_name, 'r') as file:
        current_key = None

        for line in file:
            line = line.strip()

            if not line:
                continue  # Skip empty lines

            # Check if the line is a section header
            if line.isalpha():
                current_key = line.replace(" - closed", "")
                data[current_key] = []
            else:
                # Split the line into float values and add to the current key
                values = list(map(float, line.split()))
                data[current_key].append(values)

    # Convert to JSON
    json_data = json.dumps(data, indent=4)
    print("hi")


if __name__ == "__main__":
    txt_json_convert("C:/Users/Kamil/Aortic_valve/data/Homburg pathology txt files",
                    "o_HOM_M19_H217_W96_YA.txt")



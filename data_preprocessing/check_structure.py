import os


# Function for creating folder structure
def create_directory_structure(base_path, structure):
    for key, value in structure.items():
        # Full path to the current folder
        current_path = os.path.join(base_path, key)

        # Check and create the current folder
        if not os.path.exists(current_path):
            os.makedirs(current_path)

        # If the value is a list, create nested folders
        if isinstance(value, list):
            for item in value:
                # If the element is a string, create a folder
                if isinstance(item, str):
                    sub_path = os.path.join(current_path, item)
                    if not os.path.exists(sub_path):
                        os.makedirs(sub_path)
                # If the element is a dictionary, process it recursively
                elif isinstance(item, dict):
                    create_directory_structure(current_path, item)

        # If the element is a dictionary, process it recursively
        elif isinstance(value, dict):
            create_directory_structure(current_path, value)
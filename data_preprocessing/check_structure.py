import os


# Function for creating folder structure
def create_directory_structure(base_path, structure):
    """
    Recursively creates a folder structure based on a dictionary.
    If the folder exists, checks subfolders.

    :param base_path: Base path to create the structure.
    :param structure: Dictionary describing the folder structure.
    """
    for folder, substructure in structure.items():
        # Full path of the current folder
        current_path = os.path.join(base_path, folder)

        # Create current folder if it does not exist
        if not os.path.exists(current_path):
            os.makedirs(current_path)

        # If the value is a dictionary, we process nested folders
        if isinstance(substructure, dict):
            create_directory_structure(current_path, substructure)
        # If the value is a list, create nested folders from the list
        elif isinstance(substructure, list):
            for subfolder in substructure:
                if isinstance(subfolder, str):
                    subfolder_path = os.path.join(current_path, subfolder)
                    if not os.path.exists(subfolder_path):
                        os.makedirs(subfolder_path)
                elif isinstance(subfolder, dict):
                    create_directory_structure(current_path, subfolder)
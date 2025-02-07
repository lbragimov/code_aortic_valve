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
        os.makedirs(current_path, exist_ok=True)

        # If the value is a dictionary, we process nested folders
        if isinstance(substructure, dict):
            create_directory_structure(current_path, substructure)
        # If the value is a list, create nested folders from the list
        elif isinstance(substructure, list):
            for subfolder in substructure:
                if isinstance(subfolder, str):
                    subfolder_path = os.path.join(current_path, subfolder)
                    os.makedirs(subfolder_path, exist_ok=True)
                elif isinstance(subfolder, dict):
                    create_directory_structure(current_path, subfolder)


def collect_file_paths(folder_path, structure, output_folder=None):
    file_paths = []
    output_file_paths = [] if output_folder else None  # Создаём список только если он нужен

    for sub_dir in structure:
        for case in os.listdir(os.path.join(folder_path, sub_dir)):
            file_paths.append(os.path.join(folder_path, sub_dir, case))
            if output_folder:
                output_file_paths.append(os.path.join(output_folder, sub_dir, case))

    return (file_paths, output_file_paths) if output_folder else file_paths

from datetime import datetime
import os


def get_subfolder_names(path: str):
    """
    Returns the names of all subfolders in the given path
    """
    return [f.name for f in os.scandir(path) if f.is_dir()]


def create_folder_recursive(path: str) -> None:
    """
    Creates all dirs given by path
    """
    os.makedirs(path, exist_ok=True)


def create_unique_folder(path: str, description: str):
    """
    Formats the current time to a string and creates a unique folder in the given path
    """

    # folder name
    now = datetime.now()
    now_string = now.strftime("%y-%m-%d_%H-%M-%S_%f")
    folder_name = f"{now_string}-{description}"

    experiment_folder_path = os.path.join(path, folder_name)
    create_folder_recursive(experiment_folder_path)
    return experiment_folder_path

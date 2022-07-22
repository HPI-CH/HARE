from datetime import datetime
import os
import utils.settings as settings

def create_folders_in_path(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def new_saved_experiment_folder(experiment_name):
    """
    Creates a new folder in the saved_experiments
    :return:s the path -> you can save the model, evaluations (report, conf matrix, ...)
    """

    # folder name
    currentDT = datetime.now()
    currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
    folder_name = currentDT_str + "-" + experiment_name

    experiment_folder_path = os.path.join(settings.SAVED_EXPERIMENTS_PATH, folder_name)
    create_folders_in_path(experiment_folder_path)
    return experiment_folder_path

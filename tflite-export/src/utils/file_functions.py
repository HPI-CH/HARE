import os


def get_subfolder_names(path):
    return [f.name for f in os.scandir(path) if f.is_dir()]

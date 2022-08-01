from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings
import os
import time

settings.init()

test_folder_name = 'test_folder_07' # CAUTION: no clash with exisiting name, otherwise deleted!

# function to test
new_saved_experiment_folder(test_folder_name)
# time.sleep(5) # if you want to see something

# check, if folder exists
experiment_folder_names = os.listdir(settings.saved_experiments_path)
filter_func = lambda folder_name: folder_name.endswith(test_folder_name)
test_folder_names = list(filter(filter_func, experiment_folder_names))
assert len(test_folder_names) > 0, 'folder was not created!'

# clean up: remove created folders
for folder_name in test_folder_names:
    os.rmdir(os.path.join(settings.saved_experiments_path, folder_name))
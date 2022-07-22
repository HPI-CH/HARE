"""
File that gets executed!
Only import from experiments and tests

Execute this file from repo root dir
"""
from utils import settings

# TESTS
# import tests.test_original_jens_windowize
# import tests.test_new_experiment_folder
# import tests.test_compare_model_input
# import experiments.leave_subject_out

# EXPERIMENTS
# import experiments.hello_world
# import experiments.opportunity_jens_cnn

# settings.init("sonar")
# import experiments.sonar_cnn


if __name__ == "__main__":
    import experiments.export_res_net_exp
    #import experiments.transfer_learning_opportunity
    # import tests.test_nan_replace
    #import experiments.transfer_learning_felix
    #import experiments.choose_best_model_exp_felix
    #import utils.save_all_recordings
    #import experiments.bachelorpodium_export
    #import experiments.test_exp.py

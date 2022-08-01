import argparse
"""
File that gets executed!
Only import from experiments and tests

Execute this file from tflite-export dir
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Execute experiments as described in thesis or export tflite baseline model for on-device use.')
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['gait', 'nursing', 'opportunity'],
                        help='the data set to execute the action on', dest="dataset")
    parser.add_argument('-a', '--action', type=str, required=True, choices=['export', 'experiment'], dest="action",
                        help='the action to execute on the chosen data set. \nexport: For each subject s of the chosen data set: Train a model on the entire data set excluding data from s, then export as tflite model after freezing all non-dense layers.\nexperiment: run personalization experiment for chosen data set as described in thesis.')

    args = parser.parse_args()

    dataset = args.dataset
    action = args.action

    if dataset == "gait":
        if action == "export":
            import experiments.export_gait_cnns
        else:
            import experiments.transfer_learning_gait
    elif dataset == "nursing":
        if action == "export":
            import experiments.export_nursing_resnets
        else:
            import experiments.transfer_learning_nursing
    elif dataset == "opportunity":
        if action == "export":
            print("Since the opportunity data set has not been made compatible with our app, we do not allow export to tflite. Choose another data set or the experiment action for the opportunity data set.")
            exit(0)
        else:
            import experiments.transfer_learning_opportunity

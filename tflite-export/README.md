# UnicornML

This repository supports training deep learning models using HAR datasets such as [Opportunity dataset](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition), our SonarLab dataset and the GAIT dataset.
The configurations for loading the datasets are subclasses of [data_config.py](./src/data_configs/data_config.py).

After training the corresponding model can be exported as a tflite model including metadata about the output label strings, the required input features and the variances and means of their underlying distribution as well as function signatures (`train`, `infer`, `save` and `restore`). The exported functions and metadata are specified in the `export` function in [RainbowModel.py](./src/models/RainbowModel.py) which is also the base class for all concrete model implementations in this repo.
The exported model usually also includes a NormalizationLayer (e.g. see [ResNetModel.py](./src/models/ResNetModel.py)) that contains the mean and variance of the distribution of each input feature gathered from the whole dataset in `load_dataset` in [data_config.py](./src/data_configs/data_config.py). If the specific model contains such a normalization layer depends on the implementation of the [RainbowModel.py](./src/models/RainbowModel.py) subclass.

Opportunity HAR Dataset - "Higher level"-Activity Recognition

## How to

- all executable files are stored under src/experiments (and src/tests)
- to run something:
  - conda env required
  - in [src/runner.py](./src/runner.py) comment in the experiment- or test-python file you want to run (`import experiments.example_pythonfile` or `import tests.test_example_pythonfile2`)
    - for exporting a tflite model of ResNet after training it on the `SONAR-lab` dataset uncomment `import experiment.export_res_net_exp` in the body of the `if __name__ == "__main__":` condition in [src/runner.py](./src/runner.py)
  - run `python3 src/runner.py` from inside the repo root folder.
    - Note that you will need a data set stored according to the `dataset_path` passed to the DataConfig subclass used in your experiment. For the `export_res_net_exp` you need the SONAR-lab dataset at `../data/lab_data_filtered_without_null` from inside the repo directory. It can be downloaded from [here](https://nextcloud.hpi.de/s/fSKsgwQ2bx2DRWs).
      The model exported from this experiment will be stored under `saved_experiments`. Before exporting the model from this particular experiment, all non dense layers are frozen for training. This means that running the `train` signature of the exported model will only adapt the dense layers of the model.

## Environment setup

We use a Formatter (Black) and a Linter (PyLint) for the code. The included vscode configuration lets them run on every save.

- Black: In VSCode, open this folder and save a file. If black is not installed, it will ask you what to do. Choose "Yes" for installing it.
- PyLint: A similar dialog should appear when PyLint is not installed. Install it.

- Required Packages, run the commands below in the same order:
  - tensorflow (2.7.0) `conda install -c conda-forge tensorflow==2.7` (extra steps might be needed for different PCs, but this should work on the lab servers)
  - pandas (any) `conda install pandas`
  - sklearn (any) `conda install -c anaconda scikit-learn`
  - matplotlib (any) `conda install -c conda-forge matplotlib`
  - Linter: autopep8 `pip install autopep8`
  - TFlite-support: `pip install tflite-support`

## Guidelines

- ml coding is based on experiments
  - we explicitly allow to copy code (break the software development rule) in some cases
    - like the k-fold cross validation, there is no good modularity possible as it changes too often

# UnicornML

This repository supports training deep learning models using HAR datasets such as [Opportunity dataset](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition), our SonarLab dataset and the GAIT dataset.
The configurations for loading the datasets are subclasses of [data_config.py](./src/data_configs/data_config.py).

After training the corresponding model can be exported as a tflite model including metadata about the output label strings, the required input features and the variances and means of their underlying distribution as well as function signatures (`train`, `infer`, `save` and `restore`). The exported functions and metadata are specified in the `export` function in [RainbowModel.py](./src/models/RainbowModel.py) which is also the base class for all concrete model implementations in this repo.
The exported model usually also includes a NormalizationLayer (e.g. see [ResNetModel.py](./src/models/ResNetModel.py)) that contains the mean and variance of the distribution of each input feature gathered from the whole dataset in `load_dataset` in [data_config.py](./src/data_configs/data_config.py). Wether the specific model contains such a normalization layer depends on the implementation of the [RainbowModel.py](./src/models/RainbowModel.py) subclass.

## Execution
All relevant scripts can be executed via [runner.py](src/runner.py) by passing parameters (after setting up setting up the environment - see `Environment Setup` below). 
Note that `runner.py` needs to be executed from [this directory](.) to ensure all relative paths stay correct.
There are two parameters for executing `runner.py`:
- `--dataset [dataset]` (or `-d [dataset]`) for specifiying the data set you want to execute an action on. `[dataset]` can be one of `gait`, `nursing` or `opportunity`.
- `--action [dataset]` (or `-a [action]`) for specifying which action you want to execute on the chosen dataset. `[action]` can be one of 
  - `export`: For each subject `s` of the chosen data set: Train a model on the entire data set excluding data from `s`, then export as tflite model after freezing all non-dense layers. The models exported from this experiment will be stored under `saved_experiments`. For using tflite models within our [app](../xsens_android), copy the tflite model file to `Pixel 6\Internal shared storage\Android\data\sensors_in_paradise.xsens\files\useCases\[USE_CASE]\model.tflite`. `[USE_CASE]` is `default` by default.
  - `experiment`: run personalization experiment for chosen data set as described in thesis
An examplatory command for execution could be `python src/runner.py --dataset nursing --action export`.
    
## Environment setup

### Prerequisites
1. Install [conda](https://docs.conda.io/en/latest/)
2. Open a terminal
  - Create a new conda environment based on the conda environment specification we worked in [conda_thesis_env.txt](conda_thesis_env.txt): run `conda create --name [REPLACE_WITH_A_NAME] python=3.9.13 --file conda_thesis_env.txt`
  - Activate the newly created environment `conda activate [REPLACE_WITH_A_NAME]`
  - Install `tflite-support` via pip (it's not available using conda yet): `pip install tflite_support`
  - You have now installed all necessary dependencies. If this did not work, you can manually install them by following the instructions below. Otherwise continue with downloading the data sets in the data set section.

### Dependencies
- Required Packages. Already installed if you followed the instructions in Prerequisites (alternatively, run the commands below in the same order within a conda environment):
  - tensorflow (2.8.0) `conda install -c conda-forge tensorflow==2.8` (extra steps might be needed for different PCs, but this should work on the lab servers)
  - pandas (any) `conda install pandas`
  - sklearn (any) `conda install -c anaconda scikit-learn`
  - matplotlib (any) `conda install -c conda-forge matplotlib`
  - Linter: autopep8 `pip install autopep8`
  - TFlite-support: `pip install tflite-support`

### Data sets
The code in this directory requires different data sets to work. 
See [data README](./data/README.md) on how to acquire them.

### Development setup
We use a Formatter (Black) and a Linter (PyLint) for the code. The included vscode configuration lets them run on every save.

- Black: In VSCode, open this folder and save a file. If black is not installed, it will ask you what to do. Choose "Yes" for installing it.
- PyLint: A similar dialog should appear when PyLint is not installed. Install it.

## Guidelines

- ml coding is based on experiments
  - we explicitly allow to copy code (break the software development rule) in some cases
    - like the k-fold cross validation, there is no good modularity possible as it changes too often

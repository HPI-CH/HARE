import os
from time import sleep
import pandas as pd
import psutil
from dataclasses import asdict

from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.utils import shuffle

from evaluation.evaluator import *
from loading.cache_recording import load_cached_recordings
from models import JensModelBuilder, DeepConvBuilder
from models.ResNetModel import ResNetConfig
from preprocessing import apply_preprocessing, windowize_by_activity_wasteful, forward_fill, normalize
from preprocessing.filtering import keep_columns_all_sensors
from sonar_types import Recording, Window
from utils import log, Config, map_activities_to_int, filter_activities_negative, filter_empty
from utils.logger import error

#################################
########### CONSTANTS ###########
#################################
# As described in the paper, one sensor has 14 values
n_features_per_sensor = 14

# Dataset path (22 labels)
dataset_path_full_labels = os.path.join(Config.dataset_root_dir, 'sonar_filtered')

# Dataset path (15 labels)
dataset_path_clustered_labels = os.path.join(Config.dataset_root_dir, 'sonar_15')


def run():
    # TODO: @Orhan Hier hinschreiben, was ausgeführt werden soll
    for idx, model in enumerate(['resnet', 'deepconv', 'convlstm']):
        for jdx, val in enumerate(['kfold', 'lro', 'lwo']):
            if idx == 0 and jdx < 2:
                continue
            train(model_name=model, validation=val, sensors=['LF', 'LW', 'ST', 'RW', 'RF'], dataset='22labels')


def train(model_name='deepconv', validation='kfold', sensors=[], dataset='22labels'):
    '''
    Configuration values:
    - model: `'deepconv' | 'convlstm' | 'resnet'` the model which should be used for training
    - validation: `'kfold' | 'lro' | 'lso'` which validation method is used
    - sensors: any list of `LF, LW, ST, RW, RF` (the sensor names). The sensors' data which is passed to the model.
    - dataset: `'22labels' | '15labels'` which dataset to use for training

    Outputs:
    - the results of training in a `results.csv` file
    '''

    # Quick check to ensure enough memory is available
    stats = psutil.virtual_memory()  # returns a named tuple
    available = getattr(stats, 'available')
    if available / 1024**3 < 16:
        error("Probably not enough memory available! More than 16GB are recommended. Script continues in 5 seconds.")
        sleep(5)

    # All architectural and hyperparameters

    n_outputs = 22 if dataset == '22labels' else 15
    n_features = len(sensors) * n_features_per_sensor
    # 600 timesteps à 60Hz equals 10 seconds
    n_timesteps = 600
    n_epochs = 1

    # For validation method
    n_folds = 5

    #################################
    ###### MODEL CONFIGURATION ######
    #################################
    if model_name == 'deepconv':
        # Specific for DeepConvLSTM
        stride_size = 1
        kernel_size = 1
        n_lstm_layers = 128
        n_filters = [32, 32, 32, 32]
        model_config = DeepConvBuilder(n_features=n_features, n_outputs=n_outputs, n_timesteps=n_timesteps, stride_size=stride_size,
                                       kernel_size=kernel_size, n_epochs=n_epochs, n_filters=n_filters, n_lstm_layers=n_lstm_layers)
    elif model_name == 'convlstm':
        # Our Conv LSTM Configuration
        stride_size = 5
        kernel_size = 5
        n_lstm_layers = 128
        n_filters = [32, 64]
        model_config = DeepConvBuilder(n_features=n_features, n_outputs=n_outputs, n_timesteps=n_timesteps, stride_size=stride_size,
                                       kernel_size=kernel_size, n_epochs=n_epochs, n_filters=n_filters, n_lstm_layers=n_lstm_layers)
    elif model_name == 'resnet':
        model_config = ResNetConfig(n_outputs=n_outputs, n_features=n_features, n_timesteps=n_timesteps, n_epochs=n_epochs)
    else:
        raise Exception('Unknown model name "' + model_name + '"')

    #################################
    ########## DATA LOADING #########
    #################################
    log("Loading Data...")
    recordings = load_cached_recordings(dataset_path_full_labels if dataset == '22labels' else dataset_path_clustered_labels)

    #################################
    ####### DATA PREPROCESSING ######
    #################################
    # activities dict is filled dynamically in prepare_recordings
    activities_dict = {}
    recordings_prepared = prepare_recordings(model_config, recordings, activities_dict, sensors_to_keep=sensors)
    labels = list(activities_dict.keys())
    print('LABELS')
    print(labels)

    #################################
    ############ TRAINING ###########
    #################################
    log('Training...')
    evaluations = []
    recordings_prepared = np.array(recordings_prepared)
    kfold = LeaveOneGroupOut() if validation == 'lso' else KFold(n_splits=n_folds)
    entities_to_split = windowize_recordings(model_config, recordings_prepared) if validation == 'lwo' else recordings_prepared
    entities_to_split = np.array(entities_to_split)

    for fold_idx, (train_index, test_index) in enumerate(kfold.split(entities_to_split)):
        train_recs, test_recs = entities_to_split[train_index], entities_to_split[test_index]
        if validation == 'lwo':
            train_windows, test_windows = train_recs, test_recs
        else:
            train_windows, test_windows = windowize_recordings(model_config, train_recs), windowize_recordings(model_config,
                                                                                                               test_recs)

        eval, model = fit_kfold_iteration(model_config, train_windows, test_windows, labels=labels)

        eval['kfold_iteration'] = fold_idx
        eval['model_name'] = model_name
        eval['validation'] = validation
        eval['sensors'] = ','.join(sensors)
        eval['dataset'] = dataset
        log(eval)
        evaluations.append(eval)

    eval_df = pd.DataFrame(evaluations)
    output_path = 'results.csv'
    eval_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))


def fit_kfold_iteration(model_config: JensModelBuilder, train_windows: list[Window],
                        test_windows: list[Window], labels: list[str] = None):
    model = model_config.build()
    model.summary()

    print(asdict(model_config))

    X_train, y_train = model_config.convert(train_windows)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = model_config.convert(test_windows)

    # A model config can provide callbacks which are passed to fit for training
    callbacks = []
    if hasattr(model_config, 'create_callbacks'):
        callbacks = model_config.create_callbacks()

    model.fit(X_train, y_train, epochs=model_config.n_epochs,
              validation_data=(X_test, y_test), callbacks=callbacks)
    log("Done")

    # Generate all necessary evaluation metrics
    evaluation = evaluate(model, X_test, y_test, [
        accuracy(),
        f1_score(),
        recall_score(),
        activity_distribution(y_train),
    ])

    log(f"Accuracy: {evaluation['accuracy']}")
    log(f"F1-Score: {evaluation['f1_score_weighted']}")
    log(f"Recall: {evaluation['recall_weighted']}")

    return {**evaluation, **asdict(model_config)}, model


def prepare_recordings(model_config: JensModelBuilder, recs: list[Recording], activity_dict: dict[str, int],
                       sensors_to_keep=Config.sonar_sensor_suffix_order) -> list[
        Window]:
    if len(sensors_to_keep) == 0:
        raise Exception('Specify atleast one sensor to keep')

    return apply_preprocessing(recs, [
        keep_columns_all_sensors(Config.sonar_column_names, sensors_to_keep),
        forward_fill(),
        normalize(),
        filter_activities_negative(
            activities_to_remove=['null - activity', 'null']
        ),
        filter_empty(),
        map_activities_to_int(empty_activity_dict=activity_dict),
    ])


def windowize_recordings(model_config: JensModelBuilder, recs: list[Recording]):
    return apply_preprocessing(recs, [
        windowize_by_activity_wasteful(model_config.n_timesteps),
    ])

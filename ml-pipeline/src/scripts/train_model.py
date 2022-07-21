import os
from time import sleep
import pandas as pd
import psutil
from dataclasses import asdict

from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from evaluation.evaluator import *
from loading.cache_recording import load_cached_recordings
from models import JensModelBuilder, DeepConvBuilder
from preprocessing import apply_preprocessing, windowize_by_activity_wasteful, forward_fill, normalize, \
    linear_interpolation
from sonar_types import Recording, Window
from utils import log, Config, map_activities_to_int, filter_activities_negative, filter_empty
from utils.logger import error


def run():
    train(ffill=True)


def train(ffill=True):
    # Quick check to ensure enough memory is available
    stats = psutil.virtual_memory()  # returns a named tuple
    available = getattr(stats, 'available')
    if available / 1024**3 < 32:
        error("Probably not enough memory available! More than 32GB are recommended. Script continues in 5 seconds.")
        sleep(5)

    # All architectural and hyperparameters
    n_outputs = 18
    n_features = 70
    n_timesteps = 900
    n_epochs = 10
    stride_size = 5
    kernel_size = 5
    n_lstm_layers = 128
    n_filters = [32, 64]

    n_folds = 5

    model_config = DeepConvBuilder(n_features=n_features, n_outputs=n_outputs, n_timesteps=n_timesteps, stride_size=stride_size,
                                   kernel_size=kernel_size, n_epochs=n_epochs, n_filters=n_filters, n_lstm_layers=n_lstm_layers)

    log("Loading Data...")
    recordings = load_cached_recordings(os.path.join(Config.dataset_root_dir, 'lab_data_filtered'))

    log('Training...')
    evaluations = []

    # activities dict is filled dynamically in prepare_recordings
    activities_dict = {}
    recordings_prepared = prepare_recordings(model_config, recordings, activities_dict, ffill=ffill)
    recordings_prepared = np.array(recordings_prepared)
    labels = list(activities_dict.keys())

    kfold = KFold(n_splits=n_folds)
    for fold_idx, (train_index, test_index) in enumerate(kfold.split(recordings_prepared)):
        train_recs, test_recs = recordings_prepared[train_index], recordings_prepared[test_index]
        train_windows, test_windows = windowize_recordings(model_config, train_recs), windowize_recordings(model_config,
                                                                                                           test_recs)

        eval, model = fit_kfold_iteration(model_config, train_windows, test_windows, labels=labels)

        eval['kfold_iteration'] = fold_idx
        log(eval)
        evaluations.append(eval)

    eval_df = pd.DataFrame(evaluations)
    eval_df.to_csv('results.csv')


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
        confidence(),
        activity_distribution(y_train),
        # confusion_matrix(labels=labels),
    ])

    log(f"Accuracy: {evaluation['accuracy']}")
    log(f"F1-Score: {evaluation['f1_score_weighted']}")

    return {**evaluation, **asdict(model_config)}, model


def prepare_recordings(model_config: JensModelBuilder, recs: list[Recording], activity_dict: dict[str, int],
                       ffill=True) -> list[
        Window]:
    return apply_preprocessing(recs, [
        forward_fill() if ffill else linear_interpolation(),
        normalize(),
        filter_activities_negative(
            activities_to_remove=['null - activity', 'accessoires anlegen', 'haare waschen', 'aufwischen (staub)',
                                  'föhnen']
        ),
        filter_empty(),
        map_activities_to_int(empty_activity_dict=activity_dict),
    ])


def windowize_recordings(model_config: JensModelBuilder, recs: list[Recording]):
    return apply_preprocessing(recs, [
        windowize_by_activity_wasteful(model_config.n_timesteps),
    ])

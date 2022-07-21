from typing import Any

import numpy as np
import tensorflow
import wandb
from sklearn import metrics

from sonar_types.EvaluationFunction import EvaluationFunction
from utils.config import Config


def standard_evaluation_metrics():
    return [
        accuracy(),
        f1_score(),
        confidence(),
        activity_distribution(),
        confusion_matrix_wandb(),
    ]


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, evaluation_funcs: list[EvaluationFunction] = None) -> dict[
    str, Any]:
    if evaluation_funcs is None:
        evaluation_funcs = standard_evaluation_metrics()

    y_preds = model.predict(X_test)

    evaluation = {}

    for evaluation_func in evaluation_funcs:
        evaluation = {
            **evaluation,
            **evaluation_func(y_test, y_preds),
        }

    return evaluation


def evaluate_dataset(model, dataset: tensorflow.data.Dataset, evaluation_funcs: list[EvaluationFunction] = None) -> \
        dict[
            str, Any]:
    if evaluation_funcs is None:
        evaluation_funcs = standard_evaluation_metrics()
    X = []
    y = []
    for w, l in dataset:
        X.append(w)
        y.append(l)
    return evaluate(model, np.array(X), np.array(y), evaluation_funcs)


def accuracy() -> EvaluationFunction:
    def fn(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        predictions = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        return {
            'accuracy': (np.sum(predictions == y_true) / len(predictions))
        }

    return fn


def f1_score() -> EvaluationFunction:
    def fn(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        predictions = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        f1_score_val = metrics.f1_score(y_true, predictions, average="weighted")
        f1_score_macro = metrics.f1_score(y_true, predictions, average="macro")
        return {
            'f1_score_weighted': f1_score_val,
            'f1_score_macro': f1_score_macro,
        }

    return fn


def confidence() -> EvaluationFunction:
    def fn(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        predictions = np.argmax(y_pred, axis=1)
        y_true_idx = np.argmax(y_true, axis=1)
        correct_preds = np.max(y_pred, axis=1)[predictions == y_true_idx]
        incorrect_preds = np.max(y_pred, axis=1)[predictions != y_true_idx]

        return {
            'confidence_correct_mean': np.mean(correct_preds),
            'confidence_incorrect_mean': np.mean(incorrect_preds),
            'confidence_correct_std': np.std(correct_preds),
            'confidence_incorrect_std': np.std(incorrect_preds),
        }

    return fn


def activity_distribution(y_train: np.ndarray = None) -> EvaluationFunction:
    def fn(y_true: np.ndarray, _: np.ndarray) -> dict[str, Any]:
        return {
            'activities_train': np.sum(y_train, axis=0) if y_train is not None else 'unknown',
            'activities_test': np.sum(y_true, axis=0)
        }

    return fn


def confusion_matrix(*, labels=Config.sonar_labels) -> EvaluationFunction:
    def fn(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        # Convert raw categorical labels to idx-labels
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        return {
            'confusion_matrix': metrics.confusion_matrix(y_true, y_pred)
        }

    return fn


labels_18_dict = {'rollstuhl transfer': 0, 'geschirr einsammeln': 1, 'essen reichen': 2, 'waschen am waschbecken': 3,
                  'umkleiden': 4, 'haare kämmen': 5, 'bett machen': 6, 'bad vorbereiten': 7, 'mundpflege': 8,
                  'gesamtwaschen im bett': 9, 'rollstuhl schieben': 10, 'aufräumen': 11, 'küchenvorbereitung': 12,
                  'getränke ausschenken': 13, 'essen austragen': 14, 'essen auf teller geben': 15,
                  'medikamente stellen': 16, 'dokumentation': 17}


def confusion_matrix_wandb(*, labels=None) -> EvaluationFunction:
    if labels is None:
        labels = list(labels_18_dict.keys())

    def fn(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        y_true = np.argmax(y_true, axis=1)
        return {
            'confusion_matrix': wandb.plot.confusion_matrix(y_true=y_true, probs=y_pred, class_names=labels)
        }

    return fn

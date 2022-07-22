from tkinter import N
from models.RainbowModel import RainbowModel
from itertools import product
from sklearn.model_selection import KFold
import numpy as np


class BestPerformer:
    def __init__(self,
    params,
    score) -> None:
        self.params = params
        self.score = score
        

class GridSearchCV:
    """Runs grid search cross validation on a given model and set of parameters to optimize"""
    
    def __init__(
        self,
        model: RainbowModel,
        parameters: dict,
    ) -> None:
        self.model = model
        self.parameters = parameters
        self.cv_results_ = dict()
        self.best_performer = None
        self.best_model = None
    
    def _permutations(self) -> dict:
        """Returns all possible combinations of parameters"""
        permutations = []
        keys = list(self.parameters.keys())
        values = list(self.parameters.values())
        for index, combination in enumerate(product(*values)):
            entry = dict(zip(keys, combination))
            entry["name"] = str(index) + "_".join(map(str, combination))
            permutations.append(entry)
        return permutations

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Fits the model on the training data and returns the best parameters"""
        self.cv_results_["perm_scores"] = []
        self.cv_results_["mean_test_score"] = []
        self.cv_results_["std_test_score"] = []
        self.cv_results_["params"] = []
        best_performer = BestPerformer(None, 0)

        # iterate over all permutations and fit the model for each permutation
        for permutation in self._permutations():
            perm_scores = []
            print(f"Fitting model with parameters: {permutation}")
            # setup kfold cross validation
            kf = KFold(n_splits=5, shuffle=False)
            # iterate over all folds
            for train_index, test_index in kf.split(X, y):
                # split data into train and test
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # clone model and fit on train data
                klass = self.model.__class__
                new_params = self.model.get_params()
                new_params.update(permutation)
                curr_model = klass(**new_params)
                print(f"New model params are: {curr_model.get_params()}")
                curr_model.fit(X_train, y_train)
                # predict on test data
                score = curr_model.evaluate(X_test, y_test)[1]
                perm_scores.append(score)
                # update best performer if needed
                if score > best_performer.score:
                    best_performer = BestPerformer(permutation, score)

            # append mean and std of scores to cv_results_
            mean_score = np.mean(perm_scores)
            self.cv_results_["mean_test_score"].append(mean_score)
            self.cv_results_["std_test_score"].append(np.std(perm_scores))
            self.cv_results_["params"].append(permutation)
            self.cv_results_["perm_scores"].append(perm_scores)
            
        self.best_performer = best_performer
        klass = self.model.__class__
        new_params = self.model.get_params()
        new_params.update(best_performer.params)
        best_model = klass(**new_params)
        print(f"Training best model on all Data")
        best_model.fit(X, y)
        self.best_model = best_model

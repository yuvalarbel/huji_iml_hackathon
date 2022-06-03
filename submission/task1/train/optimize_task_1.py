import numpy as np
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

from consts import *
from preprocess_task_1 import preprocess_task_1
from model import LocationModel, TypeModel, SubtypeModels

TAG = '_optimized_draft'


CLASSIFIER_PARAM_GRID = [(RandomForestClassifier,
                          {'n_estimators': [100],
                           'max_depth': [None, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                           'max_features': [None, 'sqrt', 'log2', 0.2, 0.5]})]
REGRESSOR_PARAM_GRID = [(RandomForestRegressor,
                         {'n_estimators': [100],
                          'max_depth': [None, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                          'max_features': ['sqrt', 'log2', 0.5]})]


def train_task_1(training_data_file_path, training_labels_file_path,
                 validation_data_file_path, validation_labels_file_path,
                 test_data_file_path, test_labels_file_path,
                 save_path):
    X_train = preprocess_task_1(pd.read_csv(training_data_file_path))
    X_val = preprocess_task_1(pd.read_csv(validation_data_file_path))
    X_test = preprocess_task_1(pd.read_csv(test_data_file_path))
    y_train = pd.read_csv(training_labels_file_path)
    y_val = pd.read_csv(validation_labels_file_path)
    y_test = pd.read_csv(test_labels_file_path)

    print("Data preprocessing finished.")

    type_opt = Optimizer(TypeModel, param_grid=CLASSIFIER_PARAM_GRID)
    x_location_opt = Optimizer(LocationModel, param_grid=REGRESSOR_PARAM_GRID)
    y_location_opt = Optimizer(LocationModel, param_grid=REGRESSOR_PARAM_GRID)
    subtype_opt = Optimizer(SubtypeModels, param_grid=CLASSIFIER_PARAM_GRID, special=True)
    print("Optimizers initialized.")

    type_opt.fit(X_train, y_train[TYPE], X_val, y_val[TYPE])
    print("Val Score for optimized type:", type_opt.best_val_score)
    print("Parameters for optimized type:", type_opt.best_params)
    print("Test Score for optimized type:", type_opt.score(X_test, y_test[TYPE]))

    x_location_opt.fit(X_train, y_train.x, X_val, y_val.x)
    print("Val Score for optimized X location:", x_location_opt.best_val_score)
    print("Parameters for optimized X location:", x_location_opt.best_params)
    x_test_score = x_location_opt.score(X_test, y_test.x)
    print("Test Score for optimized X location:", x_test_score)

    y_location_opt.fit(X_train, y_train.y, X_val, y_val.y)
    print("Val Score for optimized Y location:", y_location_opt.best_val_score)
    print("Parameters for optimized Y location:", y_location_opt.best_params)
    y_test_score = y_location_opt.score(X_test, y_test.y)
    print("Test Score for optimized Y location:", y_test_score)

    print("Final location score on val:", x_location_opt.best_val_score + y_location_opt.best_val_score)
    print("Final location score on test:", x_test_score + y_test_score)

    type_predictions_val = type_opt.best_model.predict(X_val)
    type_predictions_test = type_opt.best_model.predict(X_test)
    subtype_opt.fit(X_train, y_train[[TYPE, SUBTYPE]], X_val, y_val[SUBTYPE], type_predictions_val)
    print("Val Score for optimized subtype:", subtype_opt.best_val_score)
    print("Parameters for optimized subtype:", subtype_opt.best_params)
    print("Test Score for optimized subtype:", subtype_opt.score(X_test, y_test[SUBTYPE], type_predictions_test))

    print("Optimizers fitting finished.")

    joblib.dump(type_opt, save_path + '_type_opt' + TAG)
    joblib.dump(x_location_opt, save_path + '_xlocation_opt' + TAG)
    joblib.dump(y_location_opt, save_path + '_ylocation_opt' + TAG)
    joblib.dump(subtype_opt, save_path + '_subtype_opt' + TAG)

    print("Optimized models saved.")


class Optimizer(object):
    ITERATIONS = 10

    def __init__(self, model, param_grid, special=False):
        self.model = model
        self.param_grid = param_grid

        self.best_val_score = -np.inf
        self.best_model = None
        self.best_params = None

        self.special = special

    def fit(self, X_train, y_train, X_val, y_val, previous_predictions=None):
        for model_type, params in self.param_grid:
            for j, specific_params in enumerate(dict_of_options_to_subdicts(params)):
                if j % 10 == 0:
                    print("{}/{}".format(j, len(dict_of_options_to_subdicts(params))))
                for i in range(self.ITERATIONS):
                    model = self.model(model_type, specific_params)
                    model.fit(X_train, y_train)
                    val_score = model.score(X_val, y_val) if not self.special \
                        else model.score(X_val, y_val, previous_predictions)
                    if val_score > self.best_val_score:
                        self.best_val_score = val_score
                        self.best_model = model
                        self.best_params = specific_params

    def score(self, X_test, y_test, previous_predictions=None):
        return self.best_model.score(X_test, y_test) if not self.special \
            else self.best_model.score(X_test, y_test, previous_predictions)


def dict_of_options_to_subdicts(dict_of_options):
    items = list(dict_of_options.items())
    keys = [key for key, values in items]
    options = [values for key, values in items]

    results = []
    for group in itertools.product(*options):
        results.append(dict(zip(keys, group)))
    return results


if __name__ == '__main__':
    args = [TASK_1_TRAINING_SET, TASK_1_TRAINING_LABELS,
            TASK_1_VALIDATION_SET, TASK_1_VALIDATION_LABELS,
            TASK_1_TEST_SET, TASK_1_TEST_LABELS,
            TASK_1_MODEL_PATH]
    train_task_1(*['../' + arg for arg in args])

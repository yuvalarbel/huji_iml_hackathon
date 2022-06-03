import numpy as np
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

from consts import *
from preprocess_task_1 import preprocess_task_1
from model import LocationModel, TypeModel, SubtypeModels

TAG = '_optimized_draft'


TYPE_PARAM_GRID_RFC = [(RandomForestClassifier,
                        {'n_estimators': [100],
                         'max_depth': [None, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                         'max_features': [None, 'sqrt', 'log2', 0.2, 0.5]})]
LOCATION_PARAM_GRID_RFR = [(RandomForestRegressor,
                            {'n_estimators': [100],
                             'max_depth': [None, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                             'max_features': ['sqrt', 'log2', 0.5]})]
# SUBTYPE_PARAM_GRID = {'model_type': [RandomForestClassifier],
#                       'n_estimators': [100, 200, 300, 400, 500],
#                       'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
#                       'max_features': [None, 'sqrt', 'log2', 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

TYPE_PARAM_GRID_BASE = [(RandomForestClassifier,
                         {'n_estimators': [100], 'max_depth': [None]})]
LOCATION_PARAM_GRID_BASE = [(RandomForestRegressor,
                             {'n_estimators': [100], 'max_depth': [None]})]
SUBTYPE_PARAM_GRID_BASE = [(RandomForestClassifier,
                            {'n_estimators': [100], 'max_depth': [None]})]


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

    type_opt = Optimizer(TypeModel, param_grid=TYPE_PARAM_GRID_RFC)
    print("Type optimizer initialized.")
    location_opt = Optimizer(LocationModel, param_grid=LOCATION_PARAM_GRID_RFR)
    print("Location optimizer initialized.")
    # subtype_opt = Optimizer(SubtypeModels, param_grid=SUBTYPE_PARAM_GRID_BASE)
    # print("Subtype optimizer initialized.")

    type_opt.fit(X_train, y_train[TYPE], X_val, y_val[TYPE])
    print("Val Score for optimized type:", type_opt.best_val_score)
    print("Parameters for optimized type:", type_opt.best_params)
    print("Test Score for optimized type:", type_opt.score(X_test, y_test[TYPE]))

    location_opt.fit(X_train, y_train[['x', 'y']], X_val, y_val[['x', 'y']])
    print("Val Score for optimized location:", location_opt.best_val_score)
    print("Parameters for optimized location:", location_opt.best_params)
    print("Test Score for optimized location:", location_opt.score(X_test, y_test[['x', 'y']]))

    # subtype_opt.fit(X_train, y_train[SUBTYPE], X_val, y_val[SUBTYPE])
    # print("Test Score for optimized subtype:", subtype_opt.score(X_test, y_test))
    print("Optimizers fitting finished.")

    joblib.dump(type_opt, save_path + '_type_opt' + TAG)
    joblib.dump(location_opt, save_path + '_location_opt' + TAG)
    # joblib.dump(subtype_opt, save_path + '_subtype_opt' + TAG)

    print("Optimized models saved.")


class Optimizer(object):
    ITERATIONS = 5

    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = param_grid

        self.best_val_score = -np.inf
        self.best_model = None
        self.best_params = None

    def fit(self, X_train, y_train, X_val, y_val):
        for model_type, params in self.param_grid:
            for j, specific_params in enumerate(dict_of_options_to_subdicts(params)):
                if j % 10 == 0:
                    print("{}/{}".format(j, len(dict_of_options_to_subdicts(params))))
                for i in range(self.ITERATIONS):
                    model = self.model(model_type, specific_params)
                    model.fit(X_train, y_train)
                    val_score = model.score(X_val, y_val)
                    if val_score > self.best_val_score:
                        self.best_val_score = val_score
                        self.best_model = model
                        self.best_params = specific_params

    def score(self, X_test, y_test):
        return self.best_model.score(X_test, y_test)


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

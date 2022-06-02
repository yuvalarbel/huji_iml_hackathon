import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

from consts import *
from preprocess_task_1 import preprocess_task_1
from model import LocationModel, TypeModel, SubtypeModels

TAG = '_optimized_draft'


TYPE_PARAM_GRID_RFC = {'n_estimators': [100, 200, 300, 400, 500],
                       'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                       'max_features': [None, 'sqrt', 'log2', 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
LOCATION_PARAM_GRID_RFR = {'n_estimators': [100, 200, 300, 400, 500],
                           'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                           'max_features': [None, 'sqrt', 'log2', 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
# SUBTYPE_PARAM_GRID = {'model_type': [RandomForestClassifier],
#                       'n_estimators': [100, 200, 300, 400, 500],
#                       'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
#                       'max_features': [None, 'sqrt', 'log2', 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

SUBTYPE_PARAM_GRID = [{'model_type': [RandomForestClassifier],
                       'n_estimators': [100],
                       'max_depth': [None]}]


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

    type_gs = GridSearch(RandomForestClassifier(n_estimators=100, max_depth=None),
                         param_grid=TYPE_PARAM_GRID_RFC)
    x_location_gs = GridSearch(RandomForestRegressor, param_grid=LOCATION_PARAM_GRID_RFR)
    y_location_gs = GridSearch(RandomForestRegressor, param_grid=LOCATION_PARAM_GRID_RFR)
    # subtype_gs = GridSearch(SubtypeModels, param_grid=SUBTYPE_PARAM_GRID)

    print("GridSearch initialized.")

    type_gs.fit(X_train, y_train[TYPE], X_val, y_val[TYPE], scoring='f1_macro')
    x_location_gs.fit(X_train, y_train.x, X_val, y_val.x, scoring='neg_mean_squared_error')
    y_location_gs.fit(X_train, y_train.y, X_val, y_val.y, scoring='neg_mean_squared_error')
    # subtype_gs.fit(X_train, y_train[SUBTYPE], X_val, y_val[SUBTYPE])

    print("GridSearch fitting finished.")

    print("Test Score for optimized type:", type_gs.score(X_test, y_test))
    print("Test Score for optimized location, X:", x_location_gs.score(X_test, y_test))
    print("Test Score for optimized location, Y:", y_location_gs.score(X_test, y_test))
    # print("Test Score for optimized subtype:", subtype_gs.score(X_test, y_test))

    joblib.dump(type_gs, save_path + '_type_gs' + TAG)
    joblib.dump(x_location_gs, save_path + '_x_location_gs' + TAG)
    joblib.dump(y_location_gs, save_path + '_y_location_gs' + TAG)
    # joblib.dump(subtype_gs, save_path + '_subtype_gs' + TAG)

    print("Optimized models saved.")


class Optimizer()


if __name__ == '__main__':
    args = [TASK_1_TRAINING_SET, TASK_1_TRAINING_LABELS,
            TASK_1_VALIDATION_SET, TASK_1_VALIDATION_LABELS,
            TASK_1_TEST_SET, TASK_1_TEST_LABELS,
            TASK_1_MODEL_PATH]
    train_task_1(*['../' + arg for arg in args])

import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
import joblib

from consts import TASK_1_TRAINING_SET, TASK_1_TRAINING_LABELS, TASK_1_MODEL_PATH, TYPES, BASE_LABELS, TYPE, SUBTYPE
from preprocess_task_1 import preprocess_task_1


CLASSIFIER_FINAL_PARAMS = {'n_estimators': 100,
                           'learning_rate': 0.6,
                           'max_depth': 4}
X_REGRESSOR_FINAL_PARAMS = {'alpha': 10}
Y_REGRESSOR_FINAL_PARAMS = {'n_estimators': 100,
                            'max_depth': 29,
                            'max_features': 'sqrt'}


def train_task_1(training_data_file_path, training_labels_file_path, save_path, tag=''):
    """
    Train the model
    """
    data = pd.read_csv(training_data_file_path)
    processed_data = preprocess_task_1(data)

    labels = pd.read_csv(training_labels_file_path)

    model = Task1Model(save_path, tag)
    model.train(processed_data, labels)
    model.save()


class Task1Model(object):
    def __init__(self, save_path, tag=''):
        self.save_path = save_path
        self.tag = tag
        self.models = {
            'linqmap_type': GradientBoostingClassifier(**CLASSIFIER_FINAL_PARAMS),
            'x': ElasticNet(**X_REGRESSOR_FINAL_PARAMS),
            'y': RandomForestRegressor(**Y_REGRESSOR_FINAL_PARAMS),

            'linqmap_subtype_jam': GradientBoostingClassifier(**CLASSIFIER_FINAL_PARAMS),
            'linqmap_subtype_road_closed': GradientBoostingClassifier(**CLASSIFIER_FINAL_PARAMS),
            'linqmap_subtype_weatherhazard': GradientBoostingClassifier(**CLASSIFIER_FINAL_PARAMS),
            'linqmap_subtype_accident': GradientBoostingClassifier(**CLASSIFIER_FINAL_PARAMS)
        }
        self.trained = {label: False for label in self.models}

    def train(self, data, labels):
        """
        Train the model
        """
        for label in BASE_LABELS:
            self.models[label].fit(data, labels[label])
            self.trained[label] = True

        for specific_type in TYPES:
            idxs = np.logical_and(labels[TYPE] == specific_type,
                                  labels[SUBTYPE].notna())
            type_labels = labels[SUBTYPE][idxs]
            if len(type_labels) == 0:
                continue
            type_data = data[idxs]
            model_label = SUBTYPE + "_" + specific_type.lower()
            self.models[model_label].fit(type_data, type_labels)
            self.trained[model_label] = True

    def save(self):
        """
        Save the model
        """
        for name, model in self.models.items():
            joblib.dump(model, self.save_path + '_' + name + self.tag)
        with open(self.save_path + '_trained_check' + self.tag, 'w') as f:
            json.dump(self.trained, f)


if __name__ == "__main__":
    train_task_1('../' + TASK_1_TRAINING_SET, '../' + TASK_1_TRAINING_LABELS, '../' + TASK_1_MODEL_PATH)

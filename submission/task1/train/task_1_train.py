import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

from consts import TASK_1_TRAINING_SET, TASK_1_TRAINING_LABELS, TASK_1_MODEL_PATH, TYPES, BASE_LABELS, TYPE, SUBTYPE
from preprocess_task_1 import preprocess_task_1


def train_model(training_data_file_path, training_labels_file_path, tag='', save_path='../' + TASK_1_MODEL_PATH):
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
    def __init__(self, save_path, tag='', n_estimators=100, max_depth=None):
        self.save_path = save_path
        self.tag = tag
        self.models = {
            'linqmap_type': RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth),
            'x': RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth),
            'y': RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth),

            'linqmap_subtype_jam': RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth),
            'linqmap_subtype_road_closed': RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth),
            'linqmap_subtype_weatherhazard': RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth),
            'linqmap_subtype_accident': RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
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
    train_model('../' + TASK_1_TRAINING_SET, '../' + TASK_1_TRAINING_LABELS)

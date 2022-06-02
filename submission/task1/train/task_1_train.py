import pandas as pd
# import random forest regression
from sklearn.ensemble import RandomForestRegressor
import joblib

from consts import TASK_1_TRAINING_SET, TASK_1_MODEL_PATH
from preprocess_task_1 import preprocess_task_1



def train_model(training_data_file_path):
    """
    Train the model
    """
    data = pd.read_csv(training_data_file_path)
    data = preprocess_task_1(data)
    model = Task1Model()
    model.train(data)
    model.save()


class Task1Model(object):
    """
    Task 1 Model
    """
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    def train(self, data):
        """
        Train the model
        """
        self.model.fit(data)

    def save(self):
        """
        Save the model
        """
        joblib.dump(self.model, TASK_1_MODEL_PATH)


if __name__ == "__main__":
    train_model(TASK_1_TRAINING_SET)

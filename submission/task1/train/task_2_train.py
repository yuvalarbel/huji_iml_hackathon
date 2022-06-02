from consts import TASK_2_TRAIN
import pandas as pd
import numpy as np
from preprocess_task_2 import preprocess_task_2


def train_model(training_data_file_path):
    """
    Train the model
    """
    dt = preprocess_task_2(pd.read_csv(training_data_file_path))
    dt.update_date_weekday = (dt.update_date_weekday + 2) % 7
    unique_days = pd.unique(dt.update_date_weekday)
    



if __name__ == "__main__":
    train_model('C:\\Users\\User\\School\\Hackathons\\IML Hackathon '
                '2022\\huji_iml_hackathon\\submission\\task1\\data\\task2\\waze_data_train.csv')

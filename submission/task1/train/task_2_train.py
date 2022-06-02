from consts import TASK_2_TRAIN
import pandas as pd
import numpy as np
from preprocess_task_2 import preprocess_task_2
from sklearn import linear_model


def train_model(training_data_file_path):
    """
    Train the model
    """
    dt = preprocess_task_2(pd.read_csv(training_data_file_path))
    dt.update_date_weekday = (dt.update_date_weekday + 2) % 7
    model_df = pd.DataFrame(
        columns=['group', 'weekday', 'hour', 'ACCIDENT', 'JAM',
                 'ROAD_CLOSED', 'WEATHERHAZARD'])
    dt.string_group = dt.update_date_weekday.astype(str) \
                      + ' ' + dt.update_date_hour.astype(str)
    grouped = dt.groupby(dt.string_group)

    model_df.ACCIDENT = grouped.linqmap_type_ACCIDENT.sum()
    model_df.JAM = grouped.linqmap_type_JAM.sum()
    model_df.ROAD_CLOSED = grouped.linqmap_type_ROAD_CLOSED.sum()
    model_df.WEATHERHAZARD = grouped.linqmap_type_WEATHERHAZARD.sum()
    model_df.group = model_df.index.astype(str)
    model_df.weekday = model_df.group.apply(lambda x: x.split()[0])
    model_df.hour = model_df.group.apply(lambda x: x.split()[1])

    train_x = model_df.drop(['group', 'weekday', 'hour'], 1)
    train_y = model_df.drop(['group', 'ACCIDENT', 'JAM',
                 'ROAD_CLOSED', 'WEATHERHAZARD'], 1)

    test_x=pd.DataFrame(columns=['weekday', 'hour'])
    test_x['weekday']=[1,1,1,1,1,1,3,3,3,3,3,3,5,5,5,5,5,5]
    test_x['hour']=[8,9,12,13,18,19,8,9,12,13,18,19,8,9,12,13,18,19]




if __name__ == "__main__":
    train_model('C:\\Users\\User\\School\\Hackathons\\IML Hackathon '
                '2022\\huji_iml_hackathon\\submission\\task1\\data\\task2\\waze_data_train.csv')

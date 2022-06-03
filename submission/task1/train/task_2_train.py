from datetime import datetime

from consts import TASK_2_TRAIN, TASK_2_TEST_DATES
import pandas as pd
import numpy as np
from preprocess_task_2 import preprocess_task_2
from sklearn import linear_model

DATETIME_FORMAT = '%Y-%m-%d'


def train_model(training_data_file_path, date_list, output_path):
    """
    Train the model
    """

    # Preprocess dataframe for poisson regression
    n = len(date_list)
    train_x, train_y = preprocess_task_2(pd.read_csv(training_data_file_path))

    model_accident = linear_model.PoissonRegressor()
    model_accident.fit(train_x, train_y['ACCIDENT'])
    model_jam = linear_model.PoissonRegressor()
    model_jam.fit(train_x, train_y['JAM'])
    model_road_closed = linear_model.PoissonRegressor()
    model_road_closed.fit(train_x, train_y['ROAD_CLOSED'])
    model_weather_hazard = linear_model.PoissonRegressor()
    model_weather_hazard.fit(train_x, train_y['WEATHERHAZARD'])

    # Create csv files for each date
    for date in date_list:
        test_x = pd.DataFrame(columns=['weekday', 'hour'])
        day = datetime.strptime(date, DATETIME_FORMAT)
        day = (day.weekday() + 2) % 4
        test_x['weekday'] = [day] * 6
        test_x['hour'] = [8, 9, 12, 13, 18, 19]

        test_y = pd.DataFrame(columns=['ACCIDENT', 'JAM',
                                       'ROAD_CLOSED', 'WEATHERHAZARD'])
        test_y.ACCIDENT = model_accident.predict(test_x).astype(int)
        test_y.JAM = model_jam.predict(test_x).astype(int)
        test_y.ROAD_CLOSED = model_road_closed.predict(test_x).astype(int)
        test_y.WEATHERHAZARD = model_weather_hazard.predict(test_x).astype(int)

        # Add together sets of hours
        inds = test_y.index
        curr_table = pd.DataFrame(
            np.array(test_y[inds % 2 == 0]) + np.array(test_y[inds % 2 == 1]),
            columns=['ACCIDENT', 'JAM', 'ROAD_CLOSED', 'WEATHERHAZARD'])
        curr_table.to_csv(output_path + " " + date + ".csv", index=False)



if __name__ == "__main__":
    train_model('C:\\Users\\User\\School\\Hackathons\\IML Hackathon '
                '2022\\huji_iml_hackathon\\submission\\task1\\data'
                '\\waze_data_train.csv', TASK_2_TEST_DATES,
                "C:\\Users\\User\\Desktop\\IMLHack\\Task2\\")

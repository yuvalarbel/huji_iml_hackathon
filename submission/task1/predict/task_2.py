import pandas as pd
from datetime import datetime
import numpy as np

DATETIME_FORMAT = '%Y-%m-%d'


def run_task_2(date_list, models):
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
    run_task_2()

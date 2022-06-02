#!%PYTHON_HOME%\python.exe
# coding: utf-8

# Standard Library Imports
import numpy as np
import pandas as pd


class Preprocess(object):
    DROP_COLUMNS = ['OBJECTID', 'pubDate', 'linqmap_reportDescription', 'linqmap_nearby',
                    'linqmap_reportMood', 'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'nComments']
    Y_COLUMNS = ['linqmap_type', 'linqmap_subtype', 'x', 'y']
    DATETIME_FORMAT = '%dd/%mm/%yyyy %H:%M:%S'
    DATE_COLS = ['month', 'day', 'weekday']
    TIME_COLS = ['hour', 'minute']
    DATETIME_COLUMNS = {'update_date': DATE_COLS + TIME_COLS}
    DATETIME_DIFFERENCES = [('pubDate', 'update_date')]
    NUMBER_COLS = ['linqmap_reportRating', 'linqmap_roadType', 'linqmap_reliability']

    def __init__(self, data):
        self.data = data
        self.filter_city()
        self.convert_update_date()
        self.drop_unused_columns()
        self.features = data[[]]
        self.groups_of_features = pd.DataFrame()
        self.labels = None

        self.feature_funcs = [
            self.datetimes,
            self.numbers
        ]

    def run(self):
        self.create_features()
        return self.features

    def group_records(self):
        copied_data = self.data.copy()
        copied_data = copied_data.reset_index()
        copied_data = copied_data.drop(["index"], axis=1)
        for name, value in copied_data.iteritems():
            for i in range(4):
                copied_value = value[i:value.shape[0] - 5 + i]
                copied_value = copied_value.reset_index()
                copied_value = copied_value.drop(["index"], axis=1)
                self.groups_of_features[name + "_" + str(i)] = copied_value

        labels_data = copied_data[4:]
        labels_data = labels_data.reset_index()
        self.groups_of_features["type_label"] = labels_data["linqmap_type"]
        self.groups_of_features["subtype_label"] = labels_data["linqmap_subtype"]
        self.groups_of_features["x_label"] = labels_data["x"]
        self.groups_of_features["y_label"] = labels_data["y"]

    def filter_city(self):
        self.data = self.data[self.data["linqmap_city"] == "תל אביב - יפו"]

    def create_features(self):
        for func in self.feature_funcs:
            func()

    def add_new_feature(self, name, feature):
        assert feature.size == self.data.shape[0]
        assert name not in self.features.columns
        self.features.insert(self.features.shape[1], name, feature)

    ###### Feature functions ######
    def col_to_datetime(self, col):
        return pd.to_datetime(self.data[col], format=self.DATETIME_FORMAT)

    def datetimes(self):
        for col_name, cols in self.DATETIME_COLUMNS.items():
            datetime_col = self.col_to_datetime(col_name)
            for col in cols:
                self.add_new_feature(col_name + '_' + col, getattr(datetime_col.dt, col))

    def datetime_differences(self):
        for earlier, later in self.DATETIME_DIFFERENCES:
            diff = self.get_datetime_diff_in_secs(earlier, later)
            self.add_new_feature('_'.join([earlier, later, 'diff']), diff)

    def get_datetime_diff_in_secs(self, earlier, later):
        later_dt = self.col_to_datetime(later)
        earlier_dt = self.col_to_datetime(earlier)
        diff = (later_dt - earlier_dt) / np.timedelta64(1, 's')
        return diff

    def numbers(self):
        for col in self.NUMBER_COLS:
            self.add_new_feature(col, self.data[col])

    def convert_update_date(self):
        self.data['update_date'] = pd.to_datetime(self.data['update_date'], unit='ms')

    def drop_unused_columns(self):
        self.data = self.data.drop(self.DROP_COLUMNS, axis=1)


def load_data(filename: str, has_tags: bool):
    full_data = pd.read_csv(filename).sort_values(by=['update_date'])
    preprocesser = Preprocess(full_data)
    preprocesser.run()
    preprocesser.group_records()

    return preprocesser.groups_of_features


if __name__ == '__main__':
    processed_data = load_data("../task1/data/waze_data.csv", False)
    x = 1

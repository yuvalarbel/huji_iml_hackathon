#!%PYTHON_HOME%\python.exe
# coding: utf-8

# Standard Library Imports
import re
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
import time
from datetime import datetime


class Preprocess(object):
    Y_COLUMNS = ['linqmap_type', 'linqmap_subtype', 'x', 'y']
    DATETIME_FORMAT = '%dd/%mm/%yyyy %H:%M:%S'
    DATE_COLS = ['month', 'day', 'weekday']
    TIME_COLS = ['hour', 'minute']
    DATETIME_COLUMNS = {'update_date': DATE_COLS + TIME_COLS}
    DATETIME_DIFFERENCES = [('pubDate', 'update_date')]
    NUMBER_COLS = ['linqmap_reportRating', 'linqmap_roadType', 'linqmap_reliability']

    def __init__(self, data):
        self.data = data
        self.convert_update_date()
        self.unnormalized_features = data[[]]
        self.features = None
        self.labels = None

        self.feature_funcs = [
            self.datetimes,
            self.numbers,
        ]

    def run(self):
        self.create_features()
        self.create_labels()
        self.create_multiclass_labels()
        return self.features, self.multiclass_labels

    def create_features(self):
        for func in self.feature_funcs:
            func()

    def add_new_feature(self, name, feature):
        assert feature.size == self.data.shape[0]
        assert name not in self.unnormalized_features.columns
        self.unnormalized_features.insert(self.unnormalized_features.shape[1], name, feature)

    def filter_features(self):
        self.index_filter = self.unnormalized_features[self.FILTER_COLUMN_NAME] >= self.TWO_WEEKS
        self.unnormalized_features = self.unnormalized_features[self.index_filter].copy()

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


def load_data(filename: str, has_tags: bool):
    full_data = pd.read_csv(filename)
    preprocesser = Preprocess(full_data)
    x=1


if __name__ == '__main__':
    load_data("../task1/data/waze_data.csv", False)
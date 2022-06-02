import pandas as pd


class Preprocess(object):
    DUMMY_LIST = ['linqmap_type', 'linqmap_subtype']
    DROP_COLUMNS = ['OBJECTID', 'pubDate', 'linqmap_reportDescription', 'linqmap_nearby',
                    'linqmap_reportMood', 'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'nComments',
                    'linqmap_city', 'linqmap_street', 'test_set', 'update_date']
    DATETIME_FORMAT = '%dd/%mm/%yyyy %H:%M:%S'
    DATE_COLS = ['weekday']
    TIME_COLS = ['hour']
    DATETIME_COLUMNS = {'update_date': DATE_COLS + TIME_COLS}
    NUMBER_COLS = ['linqmap_reportRating', 'linqmap_roadType', 'linqmap_reliability']

    def __init__(self, data):
        self.data = data
        self.convert_update_date()
        self.datetimes()
        self.magvar()
        self.dummy_df()
        self.drop_unused_columns()

    def group_records(self):
        groups_of_features = pd.DataFrame()
        copied_data = self.data.copy()
        copied_data = copied_data.reset_index()

        for i in range(4):
            records = copied_data[copied_data["index"] % 4 == i]
            records = records.reset_index()
            records = records.drop(["index"], axis=1)
            records = records.drop(["level_0"], axis=1)
            for name, value in records.iteritems():
                groups_of_features[name + "_" + str(i)] = value

        return groups_of_features

    def add_new_feature(self, name, feature):
        self.data[name] = feature

    ###### Feature functions ######
    def col_to_datetime(self, col):
        return pd.to_datetime(self.data[col], format=self.DATETIME_FORMAT)

    def datetimes(self):
        for col_name, cols in self.DATETIME_COLUMNS.items():
            datetime_col = self.col_to_datetime(col_name)
            for col in cols:
                self.add_new_feature(col_name + '_' + col, getattr(datetime_col.dt, col))

    def convert_update_date(self):
        self.data['update_date'] = pd.to_datetime(self.data['update_date'], unit='ms')

    def drop_unused_columns(self):
        self.data = self.data.drop(self.DROP_COLUMNS, axis=1)

    def dummy_df(self):
        for x in self.DUMMY_LIST:
            dummies = pd.get_dummies(self.data[x], prefix=x, dummy_na=False)
            self.data = self.data.drop(x, 1)
            self.data = pd.concat([self.data, dummies], axis=1)

    def magvar(self):
        import numpy as np
        magvar = self.data["linqmap_magvar"]
        sin_magvar = np.sin(magvar)
        cos_magvar = np.cos(magvar)

        self.add_new_feature("sin_magvar", sin_magvar)
        self.add_new_feature("cos_magvar", cos_magvar)


def preprocess_task_1(data):
    # data = pd.read_csv("data/waze_take_features.csv")
    preprocesser = Preprocess(data)
    return preprocesser.group_records()


# if __name__ == '__main__':
#     preprocess_task_1(None)
#     x = 1

import pandas
import pandas as pd


class Preprocess(object):
    DROP_COLUMNS = ['OBJECTID', 'pubDate', 'linqmap_reportDescription', 'linqmap_nearby',
                    'linqmap_reportMood', 'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'nComments',
                    'linqmap_city', 'linqmap_street', 'update_date',
                    'linqmap_magvar', 'Unnamed: 0', 'linqmap_type', 'linqmap_subtype']
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

    def run(self):
        return self.data

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
        import consts
        for type in consts.TYPES:
            type_value = (self.data["linqmap_type"] == type).astype(int)
            self.add_new_feature("linqmap_type_" + type, type_value)

        for subtype in consts.SUBTYPES:
            type_value = (self.data["linqmap_subtype"] == subtype).astype(int)
            self.add_new_feature("linqmap_subtype_" + subtype, type_value)

    def magvar(self):
        import numpy as np
        magvar = self.data["linqmap_magvar"]
        sin_magvar = np.sin(magvar)
        cos_magvar = np.cos(magvar)

        self.add_new_feature("sin_magvar", sin_magvar)
        self.add_new_feature("cos_magvar", cos_magvar)

def preprocess_task_2(data):
    preprocesser = Preprocess(data)
    return preprocesser.run()

if __name__ == '__main__':
    pr=preprocess_task_2(pandas.read_csv('data/task2/waze_data_train.csv'))
    pr.to_csv("C:\\Users\\User\\Desktop\\IMLHack\\preprocessed_train_2.csv")

#     x = 1
import pandas as pd


class Preprocess(object):
    DUMMY_LIST = ['linqmap_type_0', 'linqmap_type_1', 'linqmap_type_2', 'linqmap_type_3',
                  'linqmap_subtype_0', 'linqmap_subtype_1', 'linqmap_subtype_2', 'linqmap_subtype_3',
                  'type_label', 'subtype_label']
    DROP_COLUMNS = ['OBJECTID', 'pubDate', 'linqmap_reportDescription', 'linqmap_nearby',
                    'linqmap_reportMood', 'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'nComments',
                    'linqmap_city', 'linqmap_street']
    Y_COLUMNS = ['linqmap_type', 'linqmap_subtype', 'x', 'y']
    DATETIME_FORMAT = '%dd/%mm/%yyyy %H:%M:%S'
    DATE_COLS = ['month', 'day', 'weekday']
    TIME_COLS = ['hour', 'minute']
    DATETIME_COLUMNS = {'update_date': DATE_COLS + TIME_COLS}
    NUMBER_COLS = ['linqmap_reportRating', 'linqmap_roadType', 'linqmap_reliability']

    def __init__(self, data):
        self.data = data

        self.drop_unused_columns()
        self.convert_update_date()
        self.features = data[[]]
        self.groups_of_features = pd.DataFrame()
        self.labels = None


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

    def numbers(self):
        for col in self.NUMBER_COLS:
            self.add_new_feature(col, self.data[col])

    def convert_update_date(self):
        self.data['update_date'] = pd.to_datetime(self.data['update_date'], unit='ms')

    def drop_unused_columns(self):
        self.data = self.data.drop(self.DROP_COLUMNS, axis=1)

    def dummy_df(self):
        for x in self.DUMMY_LIST:
            dummies = pd.get_dummies(self.groups_of_features[x], prefix=x, dummy_na=False)
            self.groups_of_features = self.groups_of_features.drop(x, 1)
            self.groups_of_features = pd.concat([self.groups_of_features, dummies], axis=1)


def preprocess_task_1(data):
    preprocesser = Preprocess(data)
    preprocesser.group_records()
    preprocesser.dummy_df()
    return preprocesser.groups_of_features
import pandas as pd
import numpy as np

from consts import TRAINING_DATA_PATH, TEL_AVIV, LABEL_COLUMNS

TRAIN_PERCENT, VALIDATION_PERCENT, TEST_PERCENT = 0.6, 0.2, 0.2


def create_sets():
    data = pd.read_csv(TRAINING_DATA_PATH)
    data = data[data.linqmap_city == TEL_AVIV].sort_values('update_date').copy()

    data['new_update_date'] = pd.to_datetime(data['update_date'], unit='ms')
    data['update_date_day'] = data.new_update_date.dt.strftime("%Y-%m-%d")
    data.index = np.arange(data.shape[0])

    data['test_set'] = 0
    data['linenum'] = 0

    counter = -1
    five = 1
    last_day = None
    for i, row in data.iterrows():
        if row.update_date_day != last_day or five == 5:
            counter += 1
            five = 1
            last_day = row.update_date_day
        else:
            five += 1
        data.loc[i, 'test_set'] = counter
        data.loc[i, 'linenum'] = five

    # drop test sets that don't have 5 records
    group_is_valid = data.groupby('test_set').count().OBJECTID == 5
    groups = group_is_valid[group_is_valid].index.values
    np.random.shuffle(groups)

    train_size = int(groups.shape[0] * TRAIN_PERCENT)
    validation_size = int(groups.shape[0] * (TRAIN_PERCENT + VALIDATION_PERCENT))

    train_groups = groups[:train_size]
    validation_groups = groups[train_size:validation_size]
    test_groups = groups[validation_size:]

    train_set = data[np.logical_and(data.test_set.isin(train_groups), data.linenum < 5)].copy()
    validation_set = data[np.logical_and(data.test_set.isin(validation_groups), data.linenum < 5)].copy()
    test_set = data[np.logical_and(data.test_set.isin(test_groups), data.linenum < 5)].copy()

    train_set = train_set.drop(['new_update_date', 'update_date_day', 'linenum'], axis=1)
    validation_set = validation_set.drop(['new_update_date', 'update_date_day', 'linenum'], axis=1)
    test_set = test_set.drop(['new_update_date', 'update_date_day', 'linenum'], axis=1)

    train_labels = data[np.logical_and(data.test_set.isin(train_groups), data.linenum == 5)].copy()[LABEL_COLUMNS]
    validation_labels = data[np.logical_and(data.test_set.isin(validation_groups), data.linenum == 5)].copy()[LABEL_COLUMNS]
    test_labels = data[np.logical_and(data.test_set.isin(test_groups), data.linenum == 5)].copy()[LABEL_COLUMNS]

    train_set.to_csv(r'data\task1\train_set.csv', index=False)
    validation_set.to_csv(r'data\task1\validation_set.csv', index=False)
    test_set.to_csv(r'data\task1\test_set.csv', index=False)

    train_labels.to_csv(r'data\task1\train_labels.csv', index=False)
    validation_labels.to_csv(r'data\task1\validation_labels.csv', index=False)
    test_labels.to_csv(r'data\task1\test_labels.csv', index=False)


if __name__ == "__main__":
    create_sets()

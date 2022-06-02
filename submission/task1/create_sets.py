import pandas as pd
import numpy as np

from consts import TRAINING_DATA_PATH, TEL_AVIV

TRAIN_PERCENT, VALIDATION_PERCENT, TEST_PERCENT = 0.6, 0.2, 0.2


def create_sets():
    data = pd.read_csv(TRAINING_DATA_PATH)
    data = data[data.linqmap_city == TEL_AVIV].sort_values('update_date').copy()

    data['new_update_date'] = pd.to_datetime(data['update_date'], unit='ms')
    data['update_date_day'] = data.new_update_date.dt.strftime("%Y-%m-%d")
    data.index = np.arange(data.shape[0])

    data['test_set'] = 0

    counter = -1
    five = 0
    last_day = None
    for i, row in data.iterrows():
        if row.update_date_day != last_day or five == 4:
            counter += 1
            five = 0
            last_day = row.update_date_day
        else:
            five += 1
        data.loc[i, 'test_set'] = counter

    data = data.drop(['new_update_date', 'update_date_day'], axis=1)

    # drop test sets that don't have 5 records
    group_is_valid = data.groupby('test_set').count().OBJECTID == 5
    groups = group_is_valid[group_is_valid].index.values

    # shuffle groups
    np.random.shuffle(groups)

    train_size = int(groups.shape[0] * TRAIN_PERCENT)
    validation_size = int(groups.shape[0] * (TRAIN_PERCENT + VALIDATION_PERCENT))

    train_groups = groups[:train_size]
    validation_groups = groups[train_size:validation_size]
    test_groups = groups[validation_size:]

    train_set = data[data.test_set.isin(train_groups)].copy()
    validation_set = data[data.test_set.isin(validation_groups)].copy()
    test_set = data[data.test_set.isin(test_groups)].copy()

    train_set.to_csv(r'data\task1\train_set.csv', index=False)
    validation_set.to_csv(r'data\task1\validation_set.csv', index=False)
    test_set.to_csv(r'data\task1\test_set.csv', index=False)


if __name__ == "__main__":
    create_sets()

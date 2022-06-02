import pandas as pd

from consts import TRAINING_DATA_PATH


def preprocess(training_set_file_path):
    return pd.read_csv(training_set_file_path)


if __name__ == '__main__':
    preprocess(TRAINING_DATA_PATH)

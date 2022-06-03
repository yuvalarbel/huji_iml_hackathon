
from consts import *

from train.train_task_1 import train_task_1
from predict.predict_task_1 import predict_task_1
from test.test_task_1 import test_task_1


def run_training():
    tag = '_full_data_final_model'
    train_task_1(FULL_SET, FULL_LABELS, TASK_1_MODEL_PATH, tag)
    # predictions = predict_task_1(TASK_1_TEST_SET, TASK_1_MODEL_PATH, tag)
    # test_task_1(predictions, TASK_1_TEST_LABELS)


if __name__ == '__main__':
    run_training()


from consts import TASK_1_TRAINING_SET, TASK_1_TRAINING_LABELS, TASK_1_VALIDATION_SET, TASK_1_VALIDATION_LABELS, TASK_1_MODEL_PATH

from train.train_task_1 import train_task_1
from predict.predict_task_1 import predict_task_1
from test.test_task_1 import test_task_1


def run_training():
    # TASK_1_VALIDATION_SET
    # TASK_1_VALIDATION_LABELS
    train_task_1(TASK_1_TRAINING_SET, TASK_1_TRAINING_LABELS, TASK_1_MODEL_PATH)
    predictions = predict_task_1(TASK_1_VALIDATION_SET, TASK_1_MODEL_PATH)
    test_task_1(predictions, TASK_1_VALIDATION_LABELS)
    print(predictions.shape)


if __name__ == '__main__':
    run_training()

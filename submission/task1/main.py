import argparse

from consts import TRAINING_DATA_PATH, TEST_DATA_PATH
from preprocess import preprocess
from task_1 import run_task_1
from task_2 import run_task_2


def main(training_set, task_1_test_set, task_2_list_of_dates):
    """
    Main function
    :param training_set: relative path to training set
    :type training_set: str
    :param task_1_test_set: relative path to task 1 test set
    :type task_1_test_set: str
    :param task_2_list_of_dates: list of dates to predict for task 2
    :type task_2_list_of_dates: list of pd.date
    """
    clean_data = preprocess(training_set)

    try:
        run_task_1(clean_data)
        print(f"Finished task 1!")
    except Exception as e:
        print(f"Exception ({type(e)}) in task 1:", str(e))

    try:
        run_task_2(clean_data)
        print(f"Finished task 2!")
    except Exception as e:
        print(f"Exception ({type(e)}) in task 2:", str(e))


def parse_args():
    """
    Parse command line arguments
    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(usage="%(prog)s [training_set] [task_1_test_set] [task_2_list_of_dates]")
    parser.add_argument("training_set", help="relative path to training set", type=str, default=TRAINING_DATA_PATH)
    parser.add_argument("task_1_test_set", help="relative path to task 1 test set", type=str, default=TEST_DATA_PATH)
    parser.add_argument("task_2_list_of_dates", help="list of dates to predict for task 2", type=str, nargs="*")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.training_set, args.task_1_test_set, args.task_2_list_of_dates)

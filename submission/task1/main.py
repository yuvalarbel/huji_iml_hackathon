import argparse

import consts
import train.train_task_2
from consts import TEST_DATA_PATH, TASK_2_TEST_DATES
from predict.predict_task_1 import predict_task_1
from predict.predict_task_2 import predict_task_2


USAGE = "\n%(prog)s\n" \
        "%(prog)s [-t task_1_test_set] [-d task_2_list_of_dates]\n" \
        "%(prog)s [--testset task_1_test_set] [--dates task_2_list_of_dates]"


def main(task_1_test_set, task_2_list_of_dates):
    """
    Main function

    :param task_1_test_set: relative path to task 1 test set
    :type task_1_test_set: str
    :param task_2_list_of_dates: list of dates to predict for task 2
    :type task_2_list_of_dates: list of pd.date
    """
    print("Task 1 test set path:", task_1_test_set)
    print("Task 2 test dates:", task_2_list_of_dates)

    clean_data = None

    try:
        predict_task_1(clean_data)
        print(f"Finished task 1!")
    except Exception as e:
        print(f"Exception ({type(e)}) in task 1:", str(e))

    try:
        ma, mj, mrc, mwh=train.train_task_2.train_model(consts.TASK_2_TRAIN,
                                                        task_2_list_of_dates)
        predict_task_2(task_2_list_of_dates,'./',ma, mj, mrc, mwh)
        print(f"Finished task 2!")
    except Exception as e:
        print(f"Exception ({type(e)}) in task 2:", str(e))


def parse_args():
    """
    Parse command line arguments
    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(usage=USAGE)
    parser.add_argument('-t', '--testset', dest="task_1_test_set", help="relative path to task 1 test set",
                        default=TEST_DATA_PATH, type=str, required=False)
    parser.add_argument('-d', '--dates', dest="task_2_list_of_dates", help="list of dates to predict for task 2",
                        default=TASK_2_TEST_DATES, nargs="*", type=str, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.task_1_test_set, args.task_2_list_of_dates)

TRAINING_DATA_PATH = 'data/waze_data.csv'
TEST_DATA_PATH = 'data/waze_take_features.csv'
TASK_2_TEST_DATES = ['2022/06/05', '2022/06/07', '2022/06/09']

TASK_1_TRAINING_SET = 'data/task1/train_set.csv'
TASK_1_TRAINING_LABELS = 'data/task1/train_labels.csv'

TEL_AVIV = "תל אביב - יפו"

TASK_1_MODEL_PATH = 'saved_models/task1_model'
LABEL_COLUMNS = ['linqmap_type', 'linqmap_subtype', 'x', 'y']

TYPES = ['JAM', 'ROAD_CLOSED', 'WEATHERHAZARD', 'ACCIDENT']

MOST_COMMON_SUBTYPES = {
    'ACCIDENT': 'ACCIDENT_MINOR',
    'JAM': 'JAM_STAND_STILL_TRAFFIC',
    'ROAD_CLOSED': 'ROAD_CLOSED_EVENT',
    'WEATHERHAZARD': 'HAZARD_ON_ROAD_CAR_STOPPED',
}

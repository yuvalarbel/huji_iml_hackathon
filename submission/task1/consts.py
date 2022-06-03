TRAINING_DATA_PATH = 'data/waze_data.csv'
TEST_DATA_PATH = 'data/waze_take_features.csv'
TASK_2_TEST_DATES = ['2022-06-05', '2022-06-07', '2022-06-09']
TASK_2_TRAIN = 'data/task2/waze_data_train.csv'

TASK_1_TRAINING_SET = 'data/task1/train_set.csv'
TASK_1_TRAINING_LABELS = 'data/task1/train_labels.csv'
TASK_1_VALIDATION_SET = 'data/task1/validation_set.csv'
TASK_1_VALIDATION_LABELS = 'data/task1/validation_labels.csv'
TASK_1_TEST_SET = 'data/task1/test_set.csv'
TASK_1_TEST_LABELS = 'data/task1/test_labels.csv'


TEL_AVIV = "תל אביב - יפו"

TASK_1_MODEL_PATH = 'saved_models/task1_model'

TYPES = ['JAM', 'ROAD_CLOSED', 'WEATHERHAZARD', 'ACCIDENT']
SUBTYPES = ['JAM_STAND_STILL_TRAFFIC', 'HAZARD_ON_SHOULDER_CAR_STOPPED',
            'ROAD_CLOSED_EVENT', 'JAM_HEAVY_TRAFFIC', 'JAM_MODERATE_TRAFFIC',
            'HAZARD_ON_ROAD_CONSTRUCTION', 'HAZARD_ON_ROAD_OBJECT',
            'HAZARD_ON_ROAD_CAR_STOPPED', 'HAZARD_ON_ROAD_TRAFFIC_LIGHT_FAULT',
            'ROAD_CLOSED_CONSTRUCTION', 'ACCIDENT_MINOR', 'ACCIDENT_MAJOR',
            'HAZARD_ON_ROAD_POT_HOLE', 'HAZARD_ON_SHOULDER',
            'HAZARD_WEATHER_HEAVY_SNOW', 'HAZARD_ON_SHOULDER_MISSING_SIGN',
            'HAZARD_ON_ROAD_ROAD_KILL', 'HAZARD_ON_ROAD',
            'HAZARD_WEATHER_HAIL', 'HAZARD_WEATHER',
            'HAZARD_ON_SHOULDER_ANIMALS', 'HAZARD_ON_ROAD_ICE',
            'HAZARD_WEATHER_FOG', 'HAZARD_WEATHER_FLOOD']

MOST_COMMON_SUBTYPE = {
    'ACCIDENT': 'ACCIDENT_MINOR',
    'JAM': 'JAM_STAND_STILL_TRAFFIC',
    'ROAD_CLOSED': 'ROAD_CLOSED_EVENT',
    'WEATHERHAZARD': 'HAZARD_ON_ROAD_CAR_STOPPED',
}

TASK_1_MODELS_NAMES = ['linqmap_type', 'x', 'y',
                       'linqmap_subtype_jam',
                       'linqmap_subtype_road_closed',
                       'linqmap_subtype_weatherhazard',
                       'linqmap_subtype_accident']

BASE_LABELS = ['linqmap_type', 'x', 'y']
TYPE = 'linqmap_type'
SUBTYPE = 'linqmap_subtype'

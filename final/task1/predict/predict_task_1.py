import pandas as pd
import json
import joblib

from consts import TASK_1_VALIDATION_SET, TASK_1_MODEL_PATH, TASK_1_MODELS_NAMES, MOST_COMMON_SUBTYPE, BASE_LABELS, TYPE, SUBTYPE
from preprocess_task_1 import preprocess_task_1


def predict_task_1(prediction_set_file_path, model_path, tag=''):
    data = pd.read_csv(prediction_set_file_path)
    processed_data = preprocess_task_1(data)

    models, trained = get_models(model_path, tag)

    results = processed_data[[]].copy()
    for label in BASE_LABELS:
        if trained[label]:
            results[label] = models[label].predict(processed_data)

    results[SUBTYPE] = ""
    for type_, mc_subtypes in MOST_COMMON_SUBTYPE.items():
        model_label = SUBTYPE + "_" + type_.lower()
        type_data = processed_data[results[TYPE] == type_]
        if trained[model_label] and type_data.shape[0]:
            results.loc[type_data.index, SUBTYPE] = models[model_label].predict(type_data)
        else:
            results.loc[results[TYPE] == type_, SUBTYPE] = mc_subtypes

    return results


def get_models(model_path, tag):
    models = {}
    for name in TASK_1_MODELS_NAMES:
        models[name] = joblib.load(model_path + '_' + name + tag)
    trained = json.load(open(model_path + '_trained_check' + tag, 'r'))
    return models, trained


if __name__ == "__main__":
    predict_task_1('../' + TASK_1_VALIDATION_SET, '../' + TASK_1_MODEL_PATH)

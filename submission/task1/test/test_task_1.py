import pandas as pd
from sklearn.metrics import f1_score


def test_task_1(predicted_results, true_results_file_path):
    true_results = pd.read_csv(true_results_file_path)

    type_score = f1_score(true_results.linqmap_type, predicted_results.linqmap_type, average='macro')

    true_subtype_lines = true_results[true_results.linqmap_subtype.notna()]
    subtype_score = f1_score(true_subtype_lines.linqmap_subtype,
                             predicted_results.loc[true_subtype_lines.index].linqmap_subtype,
                             average='macro')

    euclidean_distance = (predicted_results['x'] - true_results['x']) ** 2 + \
                         (predicted_results['y'] - true_results['y']) ** 2
    location_score = euclidean_distance.mean()

    # print all scores
    print('Type F1-Macro Score:', type_score)
    print('Subtype F1-Macro Score:', subtype_score)
    print('Location Euclid Distance Sum:', location_score)

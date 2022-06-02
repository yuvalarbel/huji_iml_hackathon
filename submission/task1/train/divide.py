import pandas as pd


def divide(data):
    groups_of_features = pd.DataFrame()
    copied_data = data.copy()
    copied_data = copied_data.reset_index()

    for i in range(5):
        records = copied_data[copied_data["index"] % 5 == i]
        records = records.reset_index()
        records = records.drop(["index"], axis=1)
        if i != 4:
            for name, value in records.iteritems():
                groups_of_features[name + "_" + str(i)] = value
        else:
            groups_of_features["type_label"] = records["linqmap_type"]
            groups_of_features["subtype_label"] = records["linqmap_subtype"]
            groups_of_features["x_label"] = records["x"]
            groups_of_features["y_label"] = records["y"]

    return groups_of_features


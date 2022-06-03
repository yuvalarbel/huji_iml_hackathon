import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score

from consts import TYPE, SUBTYPE, MOST_COMMON_SUBTYPE, TYPES


class TypeModel(BaseEstimator):
    def __init__(self, model_type, params):
        self.model = model_type(**params)

    def fit(self, X, y, sample_weight = None):
        self.model.fit(X, y, sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y, sample_weight = None):
        predictions = self.predict(X)
        return f1_score(y, predictions, average='macro')


class LocationModel(BaseEstimator):
    def __init__(self, model_type, params):
        self.model = model_type(**params)

    def fit(self, X, y, sample_weight = None):
        self.model.fit(X, y, sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y, sample_weight = None):
        predictions = self.predict(X)
        return -np.mean((predictions - y) ** 2)


class SubtypeModels(BaseEstimator):
    def __init__(self, model_type, params):
        self.models = {
            'jam': model_type(**params),
            'road_closed': model_type(**params),
            'weatherhazard': model_type(**params),
            'accident': model_type(**params)
        }
        self.trained = {
            'jam': False,
            'road_closed': False,
            'weatherhazard': False,
            'accident': False
        }

    def fit(self, X, y, sample_weight=None):
        for specific_type in TYPES:
            idxs = np.logical_and(y[TYPE] == specific_type,
                                  y[SUBTYPE].notna())
            if sum(idxs) == 0:
                self.trained[specific_type] = False
                continue
            type_sample_weight = sample_weight[idxs] if isinstance(sample_weight, np.ndarray) else sample_weight
            self.models[specific_type.lower()].fit(X[idxs], y[SUBTYPE][idxs], sample_weight=type_sample_weight)
            self.trained[specific_type.lower()] = True

    def predict(self, X, type_predictions):
        results = X[[]].copy()
        results[TYPE] = type_predictions
        results[SUBTYPE] = ""
        for type_, mc_subtypes in MOST_COMMON_SUBTYPE.items():
            type_data = X[results[TYPE] == type_]
            if self.trained[type_.lower()] and type_data.shape[0]:
                results.loc[type_data.index, SUBTYPE] = self.models[type_.lower()].predict(type_data)
            else:
                results.loc[results[TYPE] == type_, SUBTYPE] = mc_subtypes
        return results[SUBTYPE]

    def score(self, X, y, type_predictions):
        predictions = self.predict(X, type_predictions)
        true_subtype_lines = y[y.notna()]
        return f1_score(true_subtype_lines,
                        predictions.loc[true_subtype_lines.index],
                        average='macro')

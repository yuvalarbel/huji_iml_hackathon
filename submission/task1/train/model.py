import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
        self.model_x = model_type(**params)
        self.model_y = model_type(**params)

    def fit(self, X, y, sample_weight = None):
        self.model_x.fit(X, y.x, sample_weight)
        self.model_y.fit(X, y.y, sample_weight)

    def predict(self, X):
        results = X[[]].copy()
        results['x'] = self.model_x.predict(X)
        results['y'] = self.model_y.predict(X)
        return results

    def score(self, X, y, sample_weight = None):
        predictions = self.predict(X)
        euclidean_distance = (predictions.x - y.x) ** 2 + \
                             (predictions.y - y.y) ** 2
        return euclidean_distance.sum()

    def set_params(self, **params):
        small_params = {k: v for k, v in params.items() if k != 'model_type'}
        if self.model_type != params['model_type']:
            self.model_type = params['model_type']
            self.model_x = self.model_type()
            self.model_y = self.model_type()
        self.model_x.set_params(**small_params)
        self.model_y.set_params(**small_params)
        return super().set_params(**params)


class SubtypeModels(BaseEstimator):
    def __init__(self):
        self.model_type = RandomForestClassifier
        self.models = {
            'jam': RandomForestClassifier(n_estimators=100, max_depth=None),
            'road_closed': RandomForestClassifier(n_estimators=100, max_depth=None),
            'weatherhazard': RandomForestClassifier(n_estimators=100, max_depth=None),
            'accident': RandomForestClassifier(n_estimators=100, max_depth=None)
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

    def predict(self, X):
        results = X[[]].copy()
        results[SUBTYPE] = ""
        for type_, mc_subtypes in MOST_COMMON_SUBTYPE.items():
            type_data = X[results[TYPE] == type_]
            if self.trained[type_.lower()] and type_data.shape[0]:
                results.loc[type_data.index, SUBTYPE] = self.models[type_.lower()].predict(type_data)
            else:
                results.loc[results[TYPE] == type_, SUBTYPE] = mc_subtypes
        return results

    def score(self, X, y, sample_weight=None):
        predictions = self.predict(X)
        true_subtype_lines = y[y.notna()]
        return f1_score(true_subtype_lines,
                        predictions.loc[true_subtype_lines.index],
                        average='macro')

    def set_params(self, **params):
        small_params = {k: v for k, v in params.items() if k != 'model_type'}
        if self.model_type != params['model_type']:
            self.model_type = params['model_type']
            for type_, model in self.models.items():
                self.models[type_] = self.model_type()
        for type_, model in self.models.items():
            self.models[type_].set_params(**small_params)
        return super().set_params(**params)

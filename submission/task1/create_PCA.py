import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

def train_PCA(data):
    pca_model=PCA()
    pca_model.fit(data)
    pca_model.
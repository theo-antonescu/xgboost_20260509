import polars as pl
import os
import sklearn
from sklearn.datasets import make_regression


def generate_regression_data():
    # Generate synthetic regression data
    X, y = make_regression(n_samples=10000, n_features=10, noise=0.2, random_state=123)
    return X, y


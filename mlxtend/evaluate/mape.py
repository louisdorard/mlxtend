# TODO: add comments here and docstrings in functions below

import numpy as np

def mape(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)/y_true)
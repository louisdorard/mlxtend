# Author: Louis Dorard <louisdorard.com>
# TODO: add comments here and docstrings in functions below

from sklearn.metrics import confusion_matrix
from pandas import DataFrame

def cost(df, target_col, pred_col, cost_matrix=None):

    if (type(target_col) is int):
        target_col = df.columns[target_col]
        print("Using " + target_col + " as true label column")
    if (type(pred_col) is int):
        pred_col = df.columns[pred_col]
        print("Using " + pred_col + " as prediction column")

    y_true = df[target_col].values
    y_pred = df[pred_col].values

    return (DataFrame(confusion_matrix(y_true, y_pred)) * DataFrame(cost_matrix)).sum().sum()
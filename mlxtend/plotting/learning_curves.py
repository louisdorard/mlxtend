# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# A function for plotting learning curves of classifiers.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.utils.data import df2Xydf
import numpy as np
import matplotlib.pyplot as plt

def compute_learning_curves(model,
                        train,
                        val,
                        scoring='misclassification error'):
    """Plots learning curves of a classifier.

    Parameters
    ----------
    X_train : array-like, shape = [n_samples, n_features]
        Feature matrix of the training dataset.
    y_train : array-like, shape = [n_samples]
        True class labels of the training dataset.
    X_val : array-like, shape = [n_samples, n_features]
        Feature matrix of the val dataset.
    y_val : array-like, shape = [n_samples]
        True class labels of the val dataset.
    model : Classifier object. Must have a .predict .fit method.
    train_marker : str (default: 'o')
        Marker for the training set line plot.
    val_marker : str (default: '^')
        Marker for the val set line plot.
    scoring : str (default: 'misclassification error')
        If not 'misclassification error', accepts the following metrics
        (from scikit-learn):
        {'accuracy', 'average_precision', 'f1_micro', 'f1_macro',
        'f1_weighted', 'f1_samples', 'log_loss',
        'precision', 'recall', 'roc_auc',
        'adjusted_rand_score', 'mean_absolute_error', 'mean_squared_error',
        'median_absolute_error', 'r2'}
    print_model : bool (default: True)
        Print model parameters in plot title if True.
    style : str (default: 'fivethirtyeight')
        Matplotlib style
    legend_loc : str (default: 'best')
        Where to place the plot legend:
        {'best', 'upper left', 'upper right', 'lower left', 'lower right'}

    Returns
    ---------
    errors : (training_error, val_error): tuple of lists

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/plot_learning_curves/

    """
    X_train, y_train = df2Xydf(train)
    X_val, y_val = df2Xydf(val)

    if scoring != 'misclassification error':
        from sklearn import metrics
        from mlxtend.evaluate.mape import mape

        scoring_func = {
            'accuracy': metrics.accuracy_score,
            'average_precision': metrics.average_precision_score,
            'f1': metrics.f1_score,
            'f1_micro': metrics.f1_score,
            'f1_macro': metrics.f1_score,
            'f1_weighted': metrics.f1_score,
            'f1_samples': metrics.f1_score,
            'log_loss': metrics.log_loss,
            'precision': metrics.precision_score,
            'recall': metrics.recall_score,
            'roc_auc': metrics.roc_auc_score,
            'adjusted_rand_score': metrics.adjusted_rand_score,
            'mean_absolute_error': metrics.mean_absolute_error,
            'mean_squared_error': metrics.mean_squared_error,
            'median_absolute_error': metrics.median_absolute_error,
            'mape': mape,
            'r2': metrics.r2_score}

        if scoring not in scoring_func.keys():
            raise AttributeError('scoring must be in', scoring_func.keys())

    else:
        def misclf_err(y_predict, y):
            return (y_predict != y).sum() / float(len(y))

        scoring_func = {
            'misclassification error': misclf_err}

    train_errors = []
    val_errors = []

    rng = [int(i) for i in np.linspace(0, X_train.shape[0], 11)][1:]
    for r in rng:
        model.fit(X_train[:r], y_train[:r])

        y_train_predict = model.predict(X_train[:r])
        y_val_predict = model.predict(X_val)

        train_misclf = scoring_func[scoring](y_train[:r], y_train_predict)
        train_errors.append(train_misclf)

        val_misclf = scoring_func[scoring](y_val, y_val_predict)
        val_errors.append(val_misclf)
    
    from pandas import DataFrame
    performance = DataFrame()
    performance["train"] = train_errors
    performance["val"] = val_errors

    return performance


def plot_learning_curves(performance, scoring='misclassification error',
                         train_marker='o',
                         val_marker='^',
                         legend_loc='best'):

    plt.plot(np.arange(10, 101, 10), performance["train"].values,
                label='training set', marker=train_marker)
    plt.plot(np.arange(10, 101, 10), performance["val"].values,
                label='val set', marker=val_marker)
    plt.xlabel('Training set size in percent')

    plt.ylabel('Performance ({})'.format(scoring))
    plt.legend(loc=legend_loc, numpoints=1)
    plt.xlim([0, 110])
    max_y = max(max(performance["val"].values), max(performance["train"].values))
    min_y = min(min(performance["val"].values), min(performance["train"].values))
    plt.ylim([min_y - min_y * 0.15, max_y + max_y * 0.15])
    
    plt.show()
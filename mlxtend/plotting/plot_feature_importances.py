# Author: Louis Dorard <louisdorard.com>
# TODO: add comments here and docstrings in functions below

import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances(model, n_features=20):
    feature_names = model.steps[0][1].get_feature_names()
    feature_importances = model.steps[-1][1].feature_importances_
    n_features = min(n_features, len(feature_names))

    sorted_idx = np.argsort(feature_importances)
    pos = np.arange(n_features) + .5
    plt.barh(pos, feature_importances[sorted_idx][-n_features:], align='center')
    plt.yticks(pos, np.array(feature_names)[sorted_idx][-n_features:])
    plt.title('Feature Importances')

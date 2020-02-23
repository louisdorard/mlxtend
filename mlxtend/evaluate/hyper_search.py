# TODO: add comments here

import numpy as np

def hyper_search_report(search, n_top=5):
    """
    Show best models from a GridSearchCV or RandomizedSearchCV

    Taken from sklearn docs
    """
    results = search.cv_results_
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def hyperopt2sklearn(h_params, hp_dist, s_params_default):
    """
    Convert hyper-parameters in use by hyperopt to ones that can be used in sklearn

    This consists in transforming those that should be integers, from float to int.
    For this, we need to look at the configuration of hyper-parameters' distributions, to see which ones should be integers.

    Inputs:
    * h_params - hyper-parameters used in hyperopt
    * hp_dist - configuration of hyper-parameters' distributions
    * s_params_default - default hyper-parameters for sklearn

    Output: s_params - hyper-parameters for sklearn
    """
    s_params = h_params
    for v in hp_dist.keys():
        if hp_dist[v]['type']=="int":
            s_params[v] = int(h_params[v])
    s_params.update(s_params_default)
    return s_params
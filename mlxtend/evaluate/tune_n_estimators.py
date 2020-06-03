# Author: Louis Dorard <louisdorard.com>
# TODO: add comments here and docstrings in functions below

from mlxtend.utils.data import df2Xydf
from sklearn.metrics import accuracy_score, roc_auc_score
from mlxtend.evaluate.cost import cost
from mlxtend.evaluate.mape import mape
from sklearn.base import clone

def tune_n_estimators(base_model,
                      train,
                      val,
                      metric, cost_matrix=[],
                      values=[100, 300, 500, 700, 900]):

    #base_params = base_model.get_params()
    #params = base_params.copy()
    
    X_train, y_train = df2Xydf(train)
    X_val, y_val = df2Xydf(val)

    models = []

    for i in range(len(values)):

        models.append(clone(base_model))
        models[i].steps[-1][1].n_estimators = values[i]
        models[i].fit(X_train, y_train)

        #params['n_estimators'] = values[i]
        #model.set_params(**params)
        #s = scorer(model, X_val, y_val)

        y_pred = models[i].predict(X_val)
        if (metric=="accuracy"):
            s = accuracy_score(y_val, y_pred)
        elif (metric=="roc_auc"):
            s = roc_auc_score(y_val, y_pred)
        elif (metric=="mape"):
            s = mape(y_val, y_pred)
        elif (metric=="cost"):
            s = cost(y_val, y_pred, cost_matrix)

        if (i==0):
            best_score = s
            best_model = models[i]
        else:
            decrease = (s < best_score)
            improvement = ((metric=="mape" or metric=="cost") and decrease) \
                       or ((metric=="accuracy" or metric=="roc_auc") and not decrease)
            if (improvement):
                best_score = s
                best_model = models[i]

        print("n_estimators=" + str(values[i]) + ": " + str(s))

    return models
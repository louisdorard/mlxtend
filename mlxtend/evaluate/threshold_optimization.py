# Author: Louis Dorard <louisdorard.com>
# TODO: add comments here and docstrings in functions below

from pandas import DataFrame, concat

def threshold_optimization(df, proba_col, target_col, cost_matrix=None):

    """
    Example cost_matrix: [[0, 1], # TN, FP
                          [3, 0]] # FN, TP"""

    if (type(target_col) is int):
        target_col = df.columns[target_col]
        print("Using " + target_col + " as true label column")
    if (type(proba_col) is int):
        proba_col = df.columns[proba_col]
        print("Using " + proba_col + " as probability column")

    df_costs = df.copy()

    is_positive = (df_costs[target_col] == 1)
    nb_positive = len(df_costs.loc[is_positive])
    nb_negative = len(df_costs.loc[~is_positive])

    df_costs.sort_values(by=proba_col, ascending=False, inplace=True)

    # proba_col is now listing all threshold values to consider, in descending order
    # for each, we can compute the number of TP/FP/TN/FN it gives, and the cost

    df_costs['nb_TP'] = df_costs[target_col].cumsum()
    df_costs['nb_FP'] = (1-df_costs[target_col]).cumsum()
    df_costs['nb_TN'] = nb_negative - df_costs['nb_FP']
    df_costs['nb_FN'] = nb_positive - df_costs['nb_TP']

    df_costs['cost'] = df_costs['nb_TP'] * cost_matrix[1][1] \
                     + df_costs['nb_TN'] * cost_matrix[0][0] \
                     + df_costs['nb_FP'] * cost_matrix[0][1] \
                     + df_costs['nb_FN'] * cost_matrix[1][0]
    
    # all we need is to find which threshold value minimizes the cost

    index_opt = df_costs['cost'].idxmin()
    total_cost = df_costs.loc[index_opt]['cost']
    threshold = df_costs.loc[index_opt][proba_col]

    return threshold, total_cost
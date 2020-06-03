# Author: Louis Dorard <louisdorard.com>
# TODO: add comments here and docstrings in functions below

def add_regression_errors(df, target_col, pred_col):
    df["error"] = df[pred_col] - df[target_col]
    df["error_%"] = 100 * df["error"] / df[target_col]

def top_k_regression(df, error_col, k=5):
    df_abs_err = abs(df[error_col])
    smallest_errors_idx = list(df_abs_err.nsmallest(k).index)
    largest_errors_idx = list(df_abs_err.nlargest(k).index)
    smallest_errors = df.loc[smallest_errors_idx]
    largest_errors = df.loc[largest_errors_idx]
    return smallest_errors, largest_errors

# Author: Louis Dorard <louisdorard.com>
# TODO: add comments here and docstrings in functions below
# Inspired from https://github.com/hundredblocks/ml-powered-applications/blob/master/ml_editor/model_evaluation.py

def get_confusion_masks(df, target_col, pred_col):
    correct_mask = (df[pred_col] == df[target_col])
    incorrect_mask = (df[pred_col] != df[target_col])
    TP_mask = correct_mask * df[pred_col]
    TN_mask = correct_mask * ~df[pred_col]
    FP_mask = incorrect_mask * df[pred_col]
    FN_mask = incorrect_mask * ~df[pred_col]
    return TP_mask, TN_mask, FP_mask, FN_mask

def add_classification_errors(df, target_col, pred_col):
    TP_mask, TN_mask, FP_mask, FN_mask = get_confusion_masks(df, target_col, pred_col)
    # save error information, to make top-k analysis easier in a spreadsheet program
    df.loc[TP_mask, "error"] = False
    df.loc[TN_mask, "error"] = False
    df.loc[FP_mask, "error"] = True
    df.loc[FN_mask, "error"] = True
    df.loc[TP_mask, "error_type"] = "TP"
    df.loc[TN_mask, "error_type"] = "TN"
    df.loc[FP_mask, "error_type"] = "FP"
    df.loc[FN_mask, "error_type"] = "FN"

def top_k_classification(df, proba_col, target_col, pred_col, k=5):

    TP_mask, TN_mask, FP_mask, FN_mask = get_confusion_masks(df, target_col, pred_col)
    TP = df[TP_mask]
    TN = df[TN_mask]
    FP = df[FP_mask]
    FN = df[FN_mask]
    top_TP = TP.nlargest(k, proba_col)
    top_TN = TN.nsmallest(k, proba_col)
    top_FP = FP.nsmallest(k, proba_col)
    top_FN = FN.nlargest(k, proba_col)

    # Infer decision threshold: it's in between the smallest probability for an input detected as Positive and the largest probability for an input detected as Negative
    P_mask = (df[pred_col]) # this identifies rows where prediction is True
    N_mask = ~(df[pred_col])
    decision_threshold = df[P_mask][proba_col].nsmallest(1).values[0]

    most_uncertain = df.iloc[(df[proba_col] - decision_threshold).abs().argsort()[:k]]

    return (
        top_TP,
        top_TN,
        top_FP,
        top_FN,
        most_uncertain
    )
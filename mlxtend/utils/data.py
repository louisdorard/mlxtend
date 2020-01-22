# TODO: add comments here and docstring in functions below

from sklearn.model_selection import train_test_split

def df2Xy(df, target_column=''):
    if (target_column==''):
        X = df.values[:,0:-1]
        y = df.values[:,-1]
    else:
        outputs = df[target_column]
        y = outputs.values
        features = df.drop(target_column, axis=1)
        X = features.values
    return X, y

def df2Xy_labelencoded(df, target_column=''):
    # Encode string class values as integers, so they can be used in Xgboost
    X, y = df2Xy(df, target_column)
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y)
    return X, y

def random_extract(X, y, seed, ratio=0.1):
    _, X_extract_2, _, y_extract_2 = train_test_split(X, y, test_size=ratio, random_state=seed)
    return X_extract_2, y_extract_2
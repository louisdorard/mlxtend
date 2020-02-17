# TODO: add comments here

from mlxtend.utils.data import df2Xy, filename2path
from pandas import read_csv
from sklearn.model_selection import train_test_split

def kaggle_gmsc_data_nomissing():
    data = read_csv(filename2path("give-me-some-credit"), index_col=0)
    data.dropna(inplace=True)
    return df2Xy(data, 'SeriousDlqin2yrs')

def split_kaggle_gmsc_data_nomissing(test_size=0.1, seed=42):
    X, y = kaggle_gmsc_data_nomissing()
    return train_test_split(X, y, test_size=test_size, random_state=seed)

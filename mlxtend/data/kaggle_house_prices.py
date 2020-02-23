# TODO: add comments here

from mlxtend.utils.data import df2Xy, filename2path
from pandas import read_csv
from sklearn.model_selection import train_test_split

def kaggle_house_prices():
    data = read_csv(filename2path("house-prices"), index_col=0)
    data.dropna(inplace=True)
    return df2Xy(data)

def split_kaggle_gmsc_data_nomissing(test_size=0.2, seed=42):
    X, y = kaggle_house_prices()
    return train_test_split(X, y, test_size=test_size, random_state=seed)

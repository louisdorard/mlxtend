# TODO: add comments here and docstrings in functions below

from mlxtend.utils.data import df2Xy, filename2path
from pandas import read_csv
from sklearn.model_selection import train_test_split

def hotel_reviews_data():
    data = read_csv(filename2path("hotel-reviews.csv"))
    X = data.text.values
    y = data.label.values
    return X, y

def split_hotel_reviews_data(test_size=0.2, seed=42):
    X, y = hotel_reviews_data()
    return train_test_split(X, y, test_size=test_size, random_state=seed)
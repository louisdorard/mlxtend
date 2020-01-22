# TODO: add comments here and docstrings in functions below

from dotenv import load_dotenv
from mlxtend.utils.data import df2Xy
import numpy as np
from os import getenv
from pandas import read_csv
from sklearn.model_selection import train_test_split

load_dotenv()
DATA_PATH = getenv("DATA_PATH")
print("Using data in " + DATA_PATH)

def split_hotel_reviews_data(test_size=0.2, seed=42):
    X, y = hotel_reviews_data()
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def hotel_reviews_data():
    data = read_csv(DATA_PATH + "hotel-reviews.csv")
    X = data.text.values
    y = data.label.values
    return X, y

# TODO: add comments here

from dotenv import load_dotenv
from mlxtend.utils.data import df2Xy
import numpy as np
from os import getenv
from pandas import read_csv
from sklearn.model_selection import train_test_split

load_dotenv()
DATA_PATH = getenv("DATA_PATH")
print("Using data in " + DATA_PATH)

def split_kaggle_gmsc_data_nomissing(test_size=0.1, seed=42):
    X, y = kaggle_gmsc_data_nomissing()
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def kaggle_gmsc_data_nomissing():
    data = read_csv(DATA_PATH + "kaggle-give-me-credit.csv", index_col=0)
    data.dropna(inplace=True)
    return df2Xy(data, 'SeriousDlqin2yrs')

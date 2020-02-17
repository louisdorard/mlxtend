# TODO: add comments here and docstring in functions below

from sklearn.model_selection import train_test_split
from os import getenv
from pandas import read_csv
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

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


# The functions below allow to look for DATA_PATH in the environment variables, or in a .env file.
#
# The former is useful in the context of a docker container running one single project, where we would provide DATA_PATH as an environment variable.
# The latter is useful when developing several projects on the same machine, where it's more flexible to store DATA_PATH in the projects' repos' .env files.

def _path_from_envvar(filename):
    trainfull_path = ""
    print("Looking for DATA_PATH in environment variables")
    DATA_PATH_ENVVAR = getenv("DATA_PATH")
    if DATA_PATH_ENVVAR is None:
        print("  Not found")
    else:
        print(" Found it!")
        TRAINFULL_PATH_ENVVAR = DATA_PATH_ENVVAR + "/" + filename
        print("Looking for file at " + TRAINFULL_PATH_ENVVAR + " ...")
        if Path(TRAINFULL_PATH_ENVVAR).expanduser().exists():
            print("  Found it!")
            trainfull_path = TRAINFULL_PATH_ENVVAR
    return trainfull_path
    
def _path_from_envfile(filename):
    trainfull_path = ""
    print("Looking for .env file")
    if find_dotenv()=='':
        print("  Not found")
    else:
        print("  Found it!")
        load_dotenv()
        DATA_PATH_ENVFILE = getenv("DATA_PATH")
        TRAINFULL_PATH_ENVFILE = DATA_PATH_ENVFILE + "/" + filename
        print("Looking for file at " + TRAINFULL_PATH_ENVFILE + " ...")
        if Path(TRAINFULL_PATH_ENVFILE).expanduser().exists():
            print("  Found it!")
            trainfull_path = TRAINFULL_PATH_ENVFILE
    return trainfull_path

def filename2path(projectname="", filename="train_full.csv"):
    # try to find filename in subdirectory of DATA_PATH whose name is projectname, via the env var or an env file
    # if it can't be found there, try to find filename in DATA_PATH directly
    # another implementation could be to have _path_from_envvar and _path_from_envfile look in these two possible locations, instead of filename2path
    if (projectname==""): print("Project name is empty")
    path = _path_from_envvar(projectname + "/" + filename)
    if path=="":
        path = _path_from_envfile(projectname + "/" + filename)
        if path=="":
            print("Couldn't find '" + filename + "' in a subdirectory of DATA_PATH dedicated to project '" + projectname + "'... Now looking in DATA_PATH directly...")
            path = _path_from_envvar(filename)
            if path=="":
                path = _path_from_envfile(filename)
                if path=="":
                    print("Couldn't find " + filename)
    return path

def load(projectname="", filename="train_full.csv", target_column=""):
    data = read_csv(filename2path(projectname, filename), index_col=0)
    return df2Xy(data, target_column)

def load_split(projectname="", filename="train_full.csv", target_column="", test_size=0.2, seed=42):
    X, y = load(projectname, filename, target_column)
    return train_test_split(X, y, test_size=test_size, random_state=seed)
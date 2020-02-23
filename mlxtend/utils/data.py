# TODO: add comments here and docstring in functions below

from sklearn.model_selection import train_test_split
from os import getenv
from pandas import read_csv
from dotenv import find_dotenv, load_dotenv, dotenv_values, get_key
from pathlib import Path


###
# Data transformation utils
###

def df2Xydf(df, target_column=""):
    """
    Transform dataframe into 2 dataframes: one for inputs (X) and one for outputs (y)
    """
    if (target_column==""): target_column = df.columns[-1]
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    return X, y

def df2Xy(df, target_column=""):
    """
    Transform dataframe into 2 arrays: one for inputs (X) and one for outputs (y)
    """
    X, y = df2Xydf(df, target_column)
    return X.values, y.values

def df2Xy_labelencoded(df, target_column=""):
    """
    Transform dataframe into X and y arrays, and LabelEncode y.
    This encodes string class values as integers, so they can be used in libraries such as XGBoost.
    """
    X, y = df2Xy(df, target_column)
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y)
    return X, y

def random_extract(X, y, seed, ratio=0.1):
    _, X_extract_2, _, y_extract_2 = train_test_split(X, y, test_size=ratio, random_state=seed)
    return X_extract_2, y_extract_2


###
# Data finding utils
###

# The functions below allow to look for DATA_PATH in the environment variables, or in a .env file.
#
# The former is useful in the context of a docker container running one single project, where we would provide DATA_PATH as an environment variable.
# The latter is useful when developing several projects on the same machine, where it's more flexible to store DATA_PATH in the projects' repos' .env files.

def _path_from_envvar(filename):
    """
    Look for file with given name, in DATA_PATH given by environment variable
    """
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
    """
    Look for file with given name, in DATA_PATH given by .env file
    """
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

def filename2path(projectname, filename="train_full_raw.csv"):
    """
    Find path to a file given a project name and a file name

    Look for DATA_PATH environment variable, or if it doesn't exist, DATA_PATH variable in .env file. Look for file in subdirectory of DATA_PATH whose name is projectname, then in DATA_PATH directly.
    """

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


###
# Data loading utils
###

def load_data(projectname, filename):
    return read_csv(filename2path(projectname, filename), index_col=0)

from dotenv import get_key, dotenv_values
from mlxtend.utils.data import load_data
from joblib import load
import kaggle
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from glob import glob
from papermill import execute_notebook
from pathlib import Path

def _identity(X):
    """The identity function."""
    return X

class Orchestrator():

    # IDEA: Implement grid search mechanism for (hyper-)parameters in featurize and remove_train functions?
    # * Keep in mind that each step of the search will include creating new train, val, test sets/files from raw.
    # * Do selection with same val set?

    def __init__(self, featurize=None):
        """
        * featurize_stateless: _stateless_ featurization function that applies to a "raw" dataframe and returns a dataframe with new features (note: statefull featurization functions should be implemented in an sklearn pipeline)
        """
        env_file = "ml.env"
        print("Using ml.env values: " + str(dotenv_values("ml.env")))
        self.project_name = get_key(env_file, "PROJECT_NAME")
        self.competition_name = get_key(env_file, "COMPETITION_NAME")
        self.target_name = get_key(env_file, "TARGET_NAME")
        self.model_name = get_key(env_file, "MODEL_NAME")
        if (self.model_name is None): self.model_name = "model"
        self.val_size = get_key("ml.env", "VAL_SIZE")
        self.val_size = 0.2 if (self.val_size is None or self.val_size=="") else float(self.val_size)
        self.seed = get_key(env_file, "SEED")
        self.seed = 42 if (self.seed is None or self.seed=="") else int(self.seed)
        self.featurize = _identity if (featurize is None) else featurize

    def _clean_train(self, train):
        """
        Basic cleaning steps for the train set

        Just one step for now:
        * Remove rows where output is missing - if any
        """
        return train.dropna(subset=[self.target_name])

    def predict(self, data, model_name="model"):
        """
        Predict on featurized input(s) against model

        Model is specified via a name and loaded from a file
        See sklearn doc on model persistence: https://scikit-learn.org/stable/modules/model_persistence.html
        """
        # IDEA: instead of the below, send a request to mlflow server? (which will do the model loading and predicting)
        model = load("models/" + model_name + ".joblib")
        return model.predict(data)

    def create_train_val(self, remove_train=None):
        """
        Four steps:
        1. split train_full_raw intro train_raw and val_raw
        2. featurize train and val
        3. prepare train
        4. save to data/train.csv and data/val.csv

        Arguments:
        * remove_train: function that removes rows from a dataframe and that will only be applied to the featurized train set
        """

        # 1. Split 
        remove_train = _identity if (remove_train is None) else remove_train
        train_full_raw = load_data(self.project_name, "train_full_raw.csv")
        # IDEA: create symbolic links, to have train_full_raw.csv and test_raw.csv in data/, and load from there
        train_raw, val_raw = train_test_split(train_full_raw, test_size=self.val_size, random_state=self.seed)
        train_raw.to_csv("data/train_raw.csv")
        val_raw.to_csv("data/val_raw.csv")

        # 2. Featurize
        train = self.featurize(train_raw)
        val = self.featurize(val_raw)

        # 3. Prepare train
        train = remove_train(self._clean_train(train))

        # 4. Save
        train.to_csv("data/train.csv")
        val.to_csv("data/val.csv")

    def fit(self, model_name="", full=False):
        model_name = "XGB"
        n = glob("*" + model_name + "*.ipynb")[0]
        execute_notebook(n, "/output/" + n, parameters=dict(full=full))
        # TODO: have these notebooks package models with mlflow?
        # check that models/model_name.joblib exists (or model_name-full.joblib, if full=True)
        model_path = "models/" + model_name
        if full: model_path = model_path + "-full"
        model_path = model_path + ".joblib"
        if not Path(model_path).exists():
            print("Model wasn't saved at " + model_path + " as expected!")

    def validate(self, model_name=""):
        from mlxtend.utils.data import df2Xydf
        if (model_name==""):
            model_name = self.model_name
        val_raw = read_csv("data/val_raw.csv", index_col=0)
        val = self.featurize(val_raw)
        X_val, y_val = df2Xydf(val, self.target_name)
        val[self.target_name] = self.predict(X_val, model_name)
        val.to_csv("data/val_pred.csv")

    def submit_test_pred(self, model_name=""):
        """
        Predict on raw test set against model and submit to Kaggle
        """
        if (model_name==""):
            model_name = self.model_name
        submission_file = "data/" + model_name + "-submission.csv"
        test_raw = load_data(self.project_name, "test_raw.csv")
        test = self.featurize(test_raw)
        submission = test.copy()
        submission[self.target_name] = self.predict(test, model_name)
        submission[[self.target_name]].to_csv(submission_file)
        kaggle.api.competition_submit(submission_file, model_name, self.competition_name)
        # TODO: store score as model metadata

    def ship(self, model_name="model"):
        # TODO: check eval score (?) and ship model file to mlflow server
        return NotImplemented
    
    def run_workflow(self):
        self.create_train_val()
        self.fit()
        self.validate()
        self.submit_test_pred()
        self.ship() # move to after .fit(), in case we want to use mlflow server for predictions

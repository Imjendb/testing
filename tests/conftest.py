import pytest
import pandas as pd
import pathlib
import os
from starter.starter.ml.data import process_data
from sklearn.model_selection import train_test_split
from joblib import load

@pytest.fixture(scope="session")
def data(request):
    # another test
    local_path = pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"data" #branch6
    data = pd.read_csv(local_path/"census.csv") #branch6
    # this is test2

    return data
@pytest.fixture(scope="session")
def model(request):

    model_path=pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"model"/"trained_model.pickle"
    model = load(model_path) #branch6
    # this is a test
    return model

@pytest.fixture(scope="session")
def preprocessed_data(request):

    local_path = pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"data"
    data = pd.read_csv(local_path/"census.csv")
    encoder = load(pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"model"/"encoder.pickle")
    lb = load(pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"model"/"lb.pickle")
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country", #branch6
]
    #this is test 3
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=False,encoder=encoder,lb=lb
    ) #branch6
    #testing here
    return X_train,y_train

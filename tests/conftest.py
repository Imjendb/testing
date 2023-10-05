import pytest
import pandas as pd
import pathlib
import os
from starter.starter.ml.data import process_data
from sklearn.model_selection import train_test_split
from joblib import load

@pytest.fixture(scope="session")
def data(request):

    local_path = pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"data"
    data = pd.read_csv(local_path/"census.csv")
    # testing here also

    return data
@pytest.fixture(scope="session")
def model(request):

    model_path=pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"model"/"trained_model.pickle"
    model = load(model_path)
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
    "native-country",
]
    
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=False,encoder=encoder,lb=lb
    )
    #testing here
    return X_train,y_train

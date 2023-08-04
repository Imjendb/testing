import json
import numpy as np
from fastapi.testclient import TestClient

from starter.main import app

client = TestClient(app)

def test_say_hello():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_data_pred1():
    
    data = {"age": [39,10],
    "workclass": ['State-gov','State-gov'],
    "fnlgt": [77516,77516],
    "education": ['Bachelors','Bachelors'],
    "education-num": [13,13],
    "marital-status": ['Never-married','Never-married'],
    "occupation": ['Adm-clerical','Adm-clerical'],
    "relationship": ['Not-in-family','Not-in-family'],
    "race": ['White','White'],
    "sex": ['Male','Male'],
    "capital-gain": [2174,2174],
    "capital-loss": [0,0],
    "hours-per-week": [40,40],
    "native-country": ['United-States','United-States']}

    r = client.post("/predict/", data=json.dumps(data))
    predictions=np.asarray(json.loads(r.json()["predicted_value_binary"]))
    assert r.status_code == 200
    assert np.isin(predictions,[0,1]).all()


def test_data_pred2():
    
    data = {"age": [39,10,20],
    "workclass": ['State-gov','State-gov','Self-emp-not-inc'],
    "fnlgt": [77516,83311,215646],
    "education": ['Bachelors','Bachelors','HS-grad'],
    "education-num": [13,13,9],
    "marital-status": ['Never-married','Never-married','Divorced'],
    "occupation": ['Adm-clerical','Adm-clerical','Handlers-cleaners'],
    "relationship": ['Not-in-family','Not-in-family','Husband'],
    "race": ['White','White','Black'],
    "sex": ['Male','Male','Female'],
    "capital-gain": [2174,2174,0],
    "capital-loss": [0,0,0],
    "hours-per-week": [40,40,40],
    "native-country": ['United-States','United-States','Cuba']}


    r = client.post("/predict/", data=json.dumps(data))
    binary_predictions=np.asarray(json.loads(r.json()["predicted_value_binary"]))
    str_predictions=np.asarray(json.loads(r.json()["predicted_value_str"]))
    assert r.status_code == 200
    assert len(binary_predictions)==3
    assert len(str_predictions)==3
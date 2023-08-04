import json
import numpy as np
from fastapi.testclient import TestClient

from starter.main import app

client = TestClient(app)

def test_say_hello():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_data_pred_case1():
    
    data = {"age": [39],
    "workclass": ['State-gov'],
    "fnlgt": [77516],
    "education": ['Bachelors'],
    "education-num": [13],
    "marital-status": ['Never-married'],
    "occupation": ['Adm-clerical'],
    "relationship": ['Not-in-family'],
    "race": ['White'],
    "sex": ['Male'],
    "capital-gain": [2174],
    "capital-loss": [0],
    "hours-per-week": [40],
    "native-country": ['United-States']}

    r = client.post("/predict/", data=json.dumps(data))
    prediction_bin=np.asarray(json.loads(r.json()["predicted_value_binary"]))
    prediction_str=np.asarray(json.loads(r.json()["predicted_value_str"]))
    assert r.status_code == 200
    assert prediction_bin==np.array(0)
    assert prediction_str[0]=='<=50K'


def test_data_pred_case2():
    
    data = {"age": [10],
    "workclass": ['State-gov'],
    "fnlgt": [77516],
    "education": ['Bachelors'],
    "education-num": [13],
    "marital-status": ['Never-married'],
    "occupation": ['Adm-clerical'],
    "relationship": ['Not-in-family'],
    "race": ['White'],
    "sex": ['Male'],
    "capital-gain": [2174],
    "capital-loss": [0],
    "hours-per-week": [40],
    "native-country": ['United-States']}

    r = client.post("/predict/", data=json.dumps(data))
    prediction_bin=np.asarray(json.loads(r.json()["predicted_value_binary"]))
    prediction_str=np.asarray(json.loads(r.json()["predicted_value_str"]))
    assert r.status_code == 200
    assert prediction_bin==np.array(1)
    assert prediction_str[0]=='>50K'
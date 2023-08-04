# Put the code for your API here.
from typing import Union, List
import json
from fastapi import FastAPI, HTTPException
#from starter.starter.ml.model import inference
from joblib import load
from pydantic import BaseModel, conlist, Field
import pathlib
import os
import numpy as np
from starter.starter.ml.data import process_inference_data
import pandas as pd

app = FastAPI(
    title="Exercise API",
    description="API for census dataset ml model",
    version="1.0.0",
)
class Census(BaseModel):
    age: List[int]=Field(example=[39,40])
    workclass: List[str]=Field(example=['State-gov','State-gov'])
    fnlgt: List[int]=Field(example=[77516,77516])
    education: List[str]=Field(example=['Bachelors','Bachelors'])
    education_num: List[int]=Field(alias="education-num",example=[13,13])
    marital_status: List[str]=Field(alias="marital-status",example=['Never-married','Never-married'])
    occupation: List[str]=Field(example=['Adm-clerical','Adm-clerical'])
    relationship: List[str]=Field(example=['Not-in-family','Not-in-family'])
    race: List[str]=Field(example=['White','White'])
    sex: List[str]=Field(example=['Male','Male'])
    capital_gain: List[int]=Field(alias="capital-gain",example=[2174,2174])
    capital_loss: List[int]=Field(alias="capital-loss",example=[0,0])
    hours_per_week: List[int]=Field(alias="hours-per-week",example=[40,40])
    native_country: List[str]=Field(alias="native-country",example=['United-States','United-States'])

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/predict/")
async def model_inference(data: Census):
    model_path=pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"model"/"trained_model.pickle"
    model = load(model_path)
    encoder = load(pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"model"/"encoder.pickle")
    lb = load(pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"model"/"lb.pickle")

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    data_ret=dict(data)
    data_ret=pd.DataFrame.from_dict(data_ret)
    X_val= process_inference_data(data_ret, categorical_features=cat_features,encoder=encoder)
    prediction = model.predict(X_val)
    final_prediction=lb.inverse_transform(prediction)
    bin=json.dumps(prediction.tolist())
    val=json.dumps(final_prediction.tolist())
    return {"predicted_value_binary": bin,"predicted_value_str": val}
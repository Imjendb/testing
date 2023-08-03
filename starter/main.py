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

class Census(BaseModel):
    age: List[int]
    workclass: List[str]
    fnlgt: List[int]
    education: List[str]
    education_num: List[int]=Field(alias="education-num")
    marital_status: List[str]=Field(alias="marital-status")
    occupation: List[str]
    relationship: List[str]
    race: List[str]
    sex: List[str]
    capital_gain: List[int]=Field(alias="capital-gain")
    capital_loss: List[int]=Field(alias="capital-loss")
    hours_per_week: List[int]=Field(alias="hours-per-week")
    native_country: List[str]=Field(alias="native-country")


app = FastAPI(
    title="Exercise API",
    description="API for census dataset ml model",
    version="1.0.0",
)

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/predict/")
async def model_inference(data: Census):
    model_path=pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"model"/"trained_model.pickle"
    model = load(model_path)
    encoder = load(pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"model"/"encoder.pickle")

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
    val=json.dumps(prediction.tolist())
    return {"predicted_value": val}
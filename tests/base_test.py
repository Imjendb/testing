from project_package import base
import pandas as pd
from sklearn.linear_model import LogisticRegression
from starter.starter.ml.model import train_model
from starter.starter.ml.model import compute_model_metrics
from joblib import load
import pathlib
from numpy import ndarray
from starter.starter.ml.model import inference
import numpy


def test_train_model(preprocessed_data):
    X_train,y_train=preprocessed_data
    model=train_model(X_train, y_train,max_iter=100, random_state=23)
    assert type(model)==LogisticRegression

def test_inference(preprocessed_data,model):
    X_train,y_train=preprocessed_data
    assert type(model.predict(X_train))==ndarray

def test_compute_model_metrics(preprocessed_data, model):
    X_train,y_train=preprocessed_data
    metrics=compute_model_metrics(y_train,inference(model,X_train))
    assert type(metrics)==tuple
    assert type(metrics[0])==numpy.float64
    assert type(metrics[1])==numpy.float64
    assert type(metrics[2])==numpy.float64


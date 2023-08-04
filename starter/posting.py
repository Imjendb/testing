import requests
import json

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

r = requests.post("https://render-deployment-project.onrender.com/predict/", data=json.dumps(data))

print("Predictions are: ", r.json()["predicted_value_str"])
print("Status code: ", r.status_code)

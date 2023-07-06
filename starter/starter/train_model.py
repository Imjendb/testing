from sklearn.model_selection import train_test_split
import os
import pathlib
import pandas as pd
from ml.data import process_data
from ml.model import train_model
import pickle
import pandas as pd

#import data
data_directory=pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"data"
data = pd.read_csv(data_directory/"census.csv")

# split data into training and testing sets
train, test = train_test_split(data, test_size=0.20)


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
#preprocess data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# train model
model=train_model(X_train, y_train,max_iter=1000, random_state=23)
model_path=data_directory=pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"model"/"trained_model.pickle"

# save model
#pickle.dump(model, open(model_path, "wb"))
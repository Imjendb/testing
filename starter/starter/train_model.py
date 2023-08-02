from sklearn.model_selection import train_test_split
import os
import pathlib
import pandas as pd
from ml.data import process_data
from ml.model import train_model
from ml.model import compute_model_metrics
from ml.model import inference
import pickle
import pandas as pd
import numpy as np
import os
from ml.model import performance_on_slide

data_directory=pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"data"
slices_path=pathlib.Path(os.path.abspath('__file__')).parent/"starter"
model_path=pathlib.Path(os.path.abspath('__file__')).parent/"starter"/"model"/"trained_model.pickle"

data = pd.read_csv(data_directory/"census.csv")

# replace "?" with nan
data.replace("?", np.nan, inplace=True)

# replace the null values in the columns with missing values 
# In particular, the nan values are replaces with the value that has the highest occurence
high_occ_workclass = data['workclass'].value_counts().idxmax()
high_occ_occupation = data['occupation'].value_counts().idxmax()
high_occ_country = data['native-country'].value_counts().idxmax()

nan_columns = ['workclass', 'occupation', 'native-country']
high_occ = [high_occ_workclass, high_occ_occupation, high_occ_country]
for i in range(len(nan_columns)):
    data[nan_columns[i]].fillna(high_occ[i], inplace=True)

print('Dataset columns with null values:\n', data.isnull().sum())


# split data into training and testing sets
train, test = train_test_split(data, test_size=0.20, random_state=23)


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
model=train_model(X_train, y_train,max_iter=10000, random_state=23)

# save model
pickle.dump(model, open(model_path, "wb"))
print('training ends')
print('start evaluation--overall performance--')
X_val, y_val, encoder_, lb_ = process_data(test, categorical_features=cat_features, label="salary", training=False,encoder=encoder,lb=lb)
fbeta,precision,recall=compute_model_metrics(y_val,inference(model,X_val))
print("fbeta:",fbeta)
print("precision:",precision)
print("recall:",recall)
print('------------------------------------------------')

with open(slices_path/"slice_output.txt",'w') as f:    
    print('Evaluation--data slicing--',file=f)
    for feature in cat_features:
        print(f"###############################################",file=f)
        print(f"############### Feature: {feature}  ###############",file=f)
        print(f"###############################################",file=f)
        for cls in test[feature].unique():
            fbeta,precision,recall=performance_on_slide(feature,cls,test, model,cat_features,encoder,lb)
            print(f"fbeta on {feature} = {cls} slices:",file=f)
            print(fbeta,file=f)
            print(f"precision on {feature} = {cls} slices:",file=f)
            print(precision,file=f)
            print(f"recall on {feature} = {cls} slices:",file=f)
            print(recall,file=f)
            print('------------------------------------------------',file=f)
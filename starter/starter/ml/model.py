from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train,max_iter=1000, random_state=23):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    lr = LogisticRegression(max_iter=max_iter, random_state=random_state)
    lr.fit(X_train, y_train.ravel())
    return lr


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def performance_on_slide(feature,cls,data, model,cat_features,encoder,lb):
    """ computes model performance on slice of data.

    Inputs
    ------
    feature: the exact feature among categorical feature to consider
    cls : the fixed value of a the given feature 
    data: data on which perform data slicing
    model
        Trained machine learning model.
    cat_features: list[str]
        List containing the names of the categorical features (default=[])
    encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer
    Returns
    -------
    fbeta : float
    precision : float
    recall : float
    """
    from ml.data import process_data
    slice1=data[data[feature] == cls]
    X_val, y_val, encoder_, lb_ = process_data(slice1, categorical_features=cat_features, label="salary", training=False,encoder=encoder,lb=lb)
    fbeta,precision,recall=compute_model_metrics(y_val,inference(model,X_val))
    return fbeta,precision,recall

def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


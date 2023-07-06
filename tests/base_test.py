from project_package import base
import pandas as pd

def test_base():
    return 0


def test_run():
    assert base.runme() == 1

def test_column_presence_and_type(data):
    required_columns = {
        "age": pd.api.types.is_integer_dtype,
        "workclass": pd.api.types.is_object_dtype,
        "fnlgt": pd.api.types.is_integer_dtype,
        "education": pd.api.types.is_object_dtype,
        "education-num": pd.api.types.is_integer_dtype,
        "marital-status": pd.api.types.is_object_dtype,
        "occupation": pd.api.types.is_object_dtype,
        "relationship": pd.api.types.is_object_dtype,
        "race": pd.api.types.is_object_dtype,
        "sex": pd.api.types.is_object_dtype,
        "capital-gain": pd.api.types.is_integer_dtype,
        "capital-loss": pd.api.types.is_integer_dtype,  
        "hours-per-week": pd.api.types.is_integer_dtype,
        "native-country": pd.api.types.is_object_dtype,
        "salary": pd.api.types.is_object_dtype
    }
    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"


def test_column_ranges(data):

    ranges = {
        "age": (17, 90),
        "fnlgt": (12285, 1484705),
        "education-num": (1, 16),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
        "hours-per-week": (1, 99),
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={data[col_name].min()} and max={data[col_name].max()}"
        )

def test_split_data_range(preprocessed_data):
    X_train,y_train=preprocessed_data
    assert len(X_train.shape)==2
    assert len(y_train.shape)==1
    assert X_train.shape[1]==108
    assert X_train.shape[0]>=1
    assert y_train.shape[0]>=1


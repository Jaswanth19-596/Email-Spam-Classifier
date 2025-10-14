from dotenv import load_dotenv
import yaml
import os
import mlflow
from utils.text_cleaner import TextCleaner
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pytest
import yaml


load_dotenv()

host = os.getenv("DATABRICKS_HOST")
token = os.getenv("DATABRICKS_TOKEN")



os.environ["DATABRICKS_HOST"] = host
os.environ["DATABRICKS_TOKEN"] = token

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/madhajaswanth@gmail.com/TempExperiment")


def load_model():
    model = mlflow.pyfunc.load_model(
        "models:/development.models.spam_classifier@staging"
    )
    return model


@pytest.fixture(scope="class")
def model():
    return load_model()


@pytest.fixture(scope="class")
def vectorizer():
    return joblib.load("models/text_vectorizer.pkl")


@pytest.fixture(scope="class")
def text_cleaner():
    return TextCleaner()


class TestBasicFunctionality:

    def test_model_loading(self, model):
        assert model is not None

    def test_model_prediction(self, model, vectorizer, text_cleaner):
        text = ["Hello how are you"]
        cleaned_text = text_cleaner.fit_transform(pd.Series(text))

        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        columns = [
            str(i)
            for i in range(
                params["feature_engineering"]["TfIdfVectorizer"]["max_features"]
            )
        ]
        input_data = pd.DataFrame(
            vectorizer.transform(cleaned_text).toarray(), columns=columns
        )

        prediction = model.predict(input_data)

        assert prediction[0] in [0, 1]

    def test_model_performance(self, model):

        X_test = pd.read_csv("data/processed/X_test.csv")
        y_test = pd.read_csv("data/processed/y_test.csv")

        y_pred = model.predict(X_test)

        assert accuracy_score(y_test, y_pred) > 0.5, f"Accuracy too low"
        assert precision_score(y_test, y_pred) > 0.2, f"Precision too low"
        assert recall_score(y_test, y_pred) > 0.5, f"Recall too low"

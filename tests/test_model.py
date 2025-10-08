

from dotenv import load_dotenv
import yaml
import os
import mlflow
from utils.text_cleaner import TextCleaner
import joblib
import pandas as pd


load_dotenv()


class TestBasicFunctionality:
    
    def load_model(self):
        os.environ['DATABRICKS_HOST'] = 'https://dbc-61387035-3f92.cloud.databricks.com'
        os.environ['DATABRICKS_TOKEN'] = os.getenv('DATABRICKS_ACCESS_TOKEN')

        mlflow.set_tracking_uri('databricks')
        mlflow.set_experiment('/Users/madhajaswanth@gmail.com/TempExperiment')

        model = mlflow.pyfunc.load_model('models:/development.models.spam_classifier@staging')
        return model
    




    def test_model_loading(self):
        model = self.load_model()
        assert model != None


    def test_model_prediction(self):
        model = self.load_model()
        text_cleaner = TextCleaner()
        vectorizer = joblib.load('models/text_vectorizer.pkl')

        text = ["Hello how are you","Click the link below to access 10000"]

        cleaned_text = text_cleaner.fit_transform(pd.Series(text))

        print(cleaned_text.shape)

        columns = [i for i in range(10000)]
        input_data = pd.DataFrame(vectorizer.transform(cleaned_text).toarray(), columns = columns)

        print(input_data.shape)
        print(input_data.head())

        prediction = model.predict(input_data)

        print(prediction)
        assert prediction == 0

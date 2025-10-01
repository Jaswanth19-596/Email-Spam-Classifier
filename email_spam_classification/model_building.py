from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os
import pandas as pd
import pickle
import yaml

def load_data(input_path):
    df = pd.read_csv(input_path)
    return df

def save_model(model, file_path):
    pickle.dump(model, open(file_path, 'wb'))

def load_params(params_path = 'params.yaml'):
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params['model_building']

def get_model(model_name, hyperparameters):
    models = {
        'GaussianNB': GaussianNB,
        'RandomForest': RandomForestClassifier,
        'LogisticRegression': LogisticRegression,
        'SVM': SVC
    }
    if model_name not in models:
        raise ValueError(f"Model is not supported. Choose from {list(models.keys())}")
    
    return models[model_name](**hyperparameters)


def main():

    # Define input directory
    input_dir = os.path.join("data", "processed")

    # Load the data
    X_train = load_data(os.path.join(input_dir, 'X_train.csv'))
    y_train = load_data(os.path.join(input_dir, 'y_train.csv'))
    y_train = y_train.values.ravel()

    # Load the params of this module
    params = load_params()

    # Get the model type
    model_name = params['model_type']

    # Get the hyper parameters of this particular model
    hyperparameters = params[model_name]

    # Build the model for using model name and hyperparameters
    model = get_model(model_name, hyperparameters)

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    save_model(model, 'models/model.pkl')

main()

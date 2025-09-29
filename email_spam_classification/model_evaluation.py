import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score, recall_score, roc_auc_score
import os

def load_model(file_path):
    model = pickle.load(open(file_path, 'rb'))
    return model
    
def load_data(path):
    df = pd.read_csv(path)
    return df

def save(metrics_dict, path):
    with open(path, 'w') as file:
        json.dump(metrics_dict, file, indent = 4)


def main():

    # Load the model
    model = load_model('models/model.pkl')

    # Define input directory
    input_dir = os.path.join("data", "processed")

    # Load the datasets
    X_test = load_data(os.path.join(input_dir, 'X_test.csv'))
    y_test = load_data(os.path.join(input_dir, 'y_test.csv'))

    # Find predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Create the dictionary
    metrics_dict={
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
    }

    # Save the metrics
    save(metrics_dict, 'metrics.json')


main()
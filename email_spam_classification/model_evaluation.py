import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score, recall_score, roc_auc_score
import os
import logging

logger = logging.getLogger(__name__)

def load_model(file_path):
    logger.info(f"Starting Model Loading from {file_path}")

    try:
        model = pickle.load(open(file_path, 'rb'))
        logger.info(f"Loaded model successfully")
        return model
    except FileNotFoundError:
        logger.critical(f'Model not found at {file_path}')
        raise
    except Exception:
        logger.critical(f'Error while loading the model')
        raise
    
def load_data(file_path):
    logger.info(f"Starting Data Loading from {file_path}")
     
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data successfully")
        return df
    except FileNotFoundError:
        logger.critical(f'Data not found at {file_path}')
        raise
    except Exception:
        logger.critical(f'Error while loading the Data')
        raise
    

def save(metrics_dict, file_path):
    logger.info(f"Starting Saving the metrics at {file_path}")

    try:
        with open(file_path, 'w') as file:
            json.dump(metrics_dict, file, indent = 4)
        logger.info(f"Successfully saved the metrics")
    except Exception:
        logger.critical(f'Error while saving the metrics at {file_path}')
        raise


def main():
    logger.info("=" * 50)
    logger.info("Model Evaluation Stage Start")
    logger.info("=" * 50)
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
    logger.info("Model Evaluation Stage End")



if __name__ == '__main__':

    logging.basicConfig(
        level = logging.DEBUG,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.FileHandler('logs.log', mode = 'a')]
    )

    main()
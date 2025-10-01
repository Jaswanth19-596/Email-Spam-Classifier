from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os
import pandas as pd
import pickle
import yaml
import logging

logger = logging.getLogger(__name__)

def load_data(input_path):
    logger.info(f"Starting Loading from {input_path}")

    try:
        df = pd.read_csv(input_path)
        logger.info(f"File loaded successfully")
        logger.debug(f"Shape of data : {df.shape}")
        return df
    except FileNotFoundError:
        logger.critical(f'File not found at {input_path}')
        raise

def save_model(model, file_path):
    logger.info(f"Started saving the file at {file_path}")
    try:
        pickle.dump(model, open(file_path, 'wb'))
        logger.info(f"Successfully saved the model at {file_path}")
    except Exception:
        logger.info(f"Error while saving the model at {file_path}")
        raise

def load_params(params_path = 'params.yaml'):
    logger.info(f"Started loading the params at {params_path}")

    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"Successfully loaded the params")
        return params['model_building']
    except Exception:
        logger.info(f"Error while loading the params at {params_path}")
        raise


def get_model(params):
    logger.info("Creating the model using the params")

     # Get the model type
    model_name = params['type']

      # Get the hyper parameters of this particular model
    hyperparameters = params[model_name]

    models = {
        'GaussianNB': GaussianNB,
        'RandomForest': RandomForestClassifier,
        'LogisticRegression': LogisticRegression,
        'SVM': SVC
    }
    if model_name not in models:
        logger.info(f"Model is not supported. Choose from {list(models.keys())}")
        raise ValueError(f"Model is not supported. Choose from {list(models.keys())}")
    
    logger.info("Model created successfully")
    return models[model_name](**hyperparameters)


def main():
    logger.info('=' * 50)
    logger.info("Model Building Start")
    logger.info('=' * 50)
    # Define input directory
    input_dir = os.path.join("data", "processed")

    # Load the data
    X_train = load_data(os.path.join(input_dir, 'X_train.csv'))
    y_train = load_data(os.path.join(input_dir, 'y_train.csv'))
    y_train = y_train.values.ravel()

    # Load the params of this module
    params = load_params()

    # Build the model for using parameters
    model = get_model(params)

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    save_model(model, 'models/model.pkl')
    logger.info("Model Building End")


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.DEBUG,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.FileHandler('logs.log', mode = 'a')]
    )
    main()

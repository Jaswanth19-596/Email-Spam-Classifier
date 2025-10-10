import logging
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import yaml

logger = logging.getLogger(__name__)


def load_data(input_path):
    logger.info("Loading Data Start")

    try:
        df = pd.read_csv(input_path)
        logger.info("Loaded Data Successfully")
        return df
    except Exception:
        logger.critical(f"Error while loading data at {input_path}")
        raise


def get_params(file_path="params.yaml"):
    logger.info("Loading Params")
    try:
        with open(file_path, "r") as f:
            params = yaml.safe_load(f)

        logger.info("Loaded params successfully")
        return params["feature_engineering"]
    except Exception:
        logger.critical(f"Error while loading params at {file_path}")


def get_vectorizer(params):
    logger.info("Loading Vectorizer")

    text_vectorizer_name = params["type"]
    hyperparamters = params[text_vectorizer_name]

    # Convert list to tuple for sklearn
    if "ngram_range" in hyperparamters:
        hyperparamters["ngram_range"] = tuple(hyperparamters["ngram_range"])

    textVectorizers = {
        "TfIdfVectorizer": TfidfVectorizer,
        "CountVectorizer": CountVectorizer,
    }

    textVectorizer = textVectorizers[text_vectorizer_name]

    res = textVectorizer(**hyperparamters)

    logger.info("Returning Text Vectorizer")
    return res


def save(df, output_path):
    logger.info("Started saving data")
    try:
        df.to_csv(output_path, index=False)
        logger.info("Successfully Saved Data")

    except Exception:
        logger.critical(f"Error while saving the data at {output_path}")


def main():
    logger.info("=" * 50)
    logger.info("Feature Engineering Stage")
    logger.info("=" * 50)

    # Define the paths of input and output directories
    input_dir = os.path.join("data", "interim")
    output_dir = os.path.join("data", "processed")

    # Load the datasets
    X_train = load_data(os.path.join(input_dir, "X_train.csv"))
    X_test = load_data(os.path.join(input_dir, "X_test.csv"))
    y_train = load_data(os.path.join(input_dir, "y_train.csv"))
    y_test = load_data(os.path.join(input_dir, "y_test.csv"))

    # Get the parameters
    params = get_params("params.yaml")

    # Create the vectorizer
    textVectorizer = get_vectorizer(params)

    X_train = pd.DataFrame(textVectorizer.fit_transform(X_train["text"]).toarray())
    X_test = pd.DataFrame(textVectorizer.transform(X_test["text"]).toarray())

    # Save the vectorizer
    joblib.dump(textVectorizer, "models/text_vectorizer.pkl")

    os.makedirs(output_dir, exist_ok=True)
    # Save the data to the output directory
    save(X_train, os.path.join(output_dir, "X_train.csv"))
    save(X_test, os.path.join(output_dir, "X_test.csv"))
    save(y_train, os.path.join(output_dir, "y_train.csv"))
    save(y_test, os.path.join(output_dir, "y_test.csv"))
    logger.info("Feature Engineering Stage End")


if __name__ == "__main__":
    logger.info("Started stage ")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs.log", mode="a"), logging.StreamHandler()],
    )

    main()

import json
import logging
import os
import joblib

from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import yaml

load_dotenv()

os.environ["DATABRICKS_HOST"] = "https://dbc-61387035-3f92.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_ACCESS_TOKEN")

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/madhajaswanth@gmail.com/TempExperiment")


logger = logging.getLogger(__name__)


def load_params(params_path="params.yaml"):
    logger.info(f"Started loading the params at {params_path}")

    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.info("Successfully loaded the params")
        return params["model_building"]
    except Exception:
        logger.info(f"Error while loading the params at {params_path}")
        raise


def load_model(file_path):
    logger.info(f"Starting Model Loading from {file_path}")

    try:
        model = joblib.load(file_path)
        logger.info("Loaded model successfully")
        return model
    except FileNotFoundError:
        logger.critical(f"Model not found at {file_path}")
        raise
    except Exception:
        logger.critical("Error while loading the model")
        raise


def load_data(file_path):
    logger.info(f"Starting Data Loading from {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info("Loaded data successfully")
        return df
    except FileNotFoundError:
        logger.critical(f"Data not found at {file_path}")
        raise
    except Exception:
        logger.critical("Error while loading the Data")
        raise


def save(metrics_dict, file_path):
    logger.info(f"Starting Saving the metrics at {file_path}")

    try:
        with open(file_path, "w") as file:
            json.dump(metrics_dict, file, indent=4)
        logger.info("Successfully saved the metrics")
    except Exception:
        logger.critical(f"Error while saving the metrics at {file_path}")
        raise


def register_and_stage_model(accuracy, threshold, model_uri):

    if accuracy <= threshold:
        logger.info(
            f"[VALIDATION] Model's accuracy {accuracy} is less than threshold {threshold}"
        )
        return

    logger.info(
        "[VALIDATION] Model has passed the accuracy threshold : Registering the Model"
    )

    try:
        registered_model = mlflow.register_model(
            model_uri, "development.models.spam_classifier"
        )
        logger.info(
            f"[REGISTRATION] Model version {registered_model.version} Successfully Registered !!"
        )

        client = MlflowClient()
        client.set_registered_model_alias(
            "development.models.spam_classifier", "Staging", registered_model.version
        )

        logger.info("[PROMOTION] Model moved to stage: Staging")
    except Exception as e:
        logger.error("Exception while registering and staging the model")
        raise e


def main():
    logger.info("=" * 50)
    logger.info("Model Evaluation Stage Start")
    logger.info("=" * 50)

    # Load the model
    model = load_model("models/model.pkl")

    # Define input directory
    input_dir = os.path.join("data", "processed")

    # Load the datasets
    X_test = load_data(os.path.join(input_dir, "X_test.csv"))
    y_test = load_data(os.path.join(input_dir, "y_test.csv"))

    # Find predictions
    logger.info("Predicting the Results using the Model")
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Create the dictionary
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

    logger.info("Starting the Run in experiment")
    try:
        with mlflow.start_run():
            logger.info("Logging the Metrics and Params")

            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)

            if hasattr(model, "get_params"):
                params = model.get_params()
                mlflow.log_params(params)

            logger.info("Logging the Model")
            logged_model = mlflow.sklearn.log_model(
                model, input_example=X_test.iloc[[1]], name="model"
            )

            register_and_stage_model(accuracy, 0.75, logged_model.model_uri)

    except Exception as e:
        logger.error("Exception while performing the Experiment")
        raise e

    # Save the metrics
    save(metrics_dict, "metrics.json")
    logger.info("Model Evaluation Stage End")


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs.log", mode="a")],
    )

    # Suppress verbose logs from third-party libraries
    logging.getLogger("databricks").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    main()

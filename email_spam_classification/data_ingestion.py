import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)



def load_data(input_path):
    logger.info(f"Starting Loading from  {input_path}")

    try:
        df = pd.read_csv(input_path, encoding = 'latin_1')
        logger.info(f"File loaded successfully")
        logger.debug(f"Shape of data : {df.shape}")
        return df
    except FileNotFoundError:
        logger.critical(f"File Not Found at {input_path}")
        raise FileNotFoundError
    except Exception:
        logger.error('Error encountered while loading the data')
        raise
    

def basic_cleaning(df):
    logger.info(f"Started Cleaning the data")

    try:
        # Remove last three features
        df = df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
        logger.info("Removed the Unnecessary columns")

        # Rename the columns
        df = df.rename({'v1': 'class', 'v2': 'text'}, axis = 1)
        logger.info("Renamed the columns")

        # Encoding the classes
        le = LabelEncoder()
        df['class'] = le.fit_transform(df['class'])
        logger.info("Encoded the classes")

        # Drop duplicated rows
        df = df.drop_duplicates()
        logger.info("Dropped the duplicates")

        logger.info("Completed Data Cleaning sucessfully")
        return df
    except Exception:
        logger.critical("Error while Cleaning the data")
        raise

def save(df, output_path):
    logger.info(f"Started saving the file at {output_path}")

    try:
        df.to_csv(output_path, index = False)
        logger.info(f"File saved successfully")
    except Exception:
        logger.critical(f"Exception while saving the file at path {output_path}")
        raise
    

def main():
    logger.info("=" * 50)
    logger.info("Starting the Data Ingestion  Pipeline")
    logger.info("=" * 50)

    input_path = os.path.join( "data", "raw", "emails.csv")
    output_dir = os.path.join("data", "interim")
    output_path = os.path.join(output_dir, 'data.csv')

    df = load_data(input_path)
    df = basic_cleaning(df)
    save(df, output_path)

    logger.info("Completed Data Ingestion Pipeline Successfully")


if __name__ == "__main__":

    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [
            logging.FileHandler('logs.log', mode = 'a')
        ]
    )
    main()
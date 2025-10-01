import contractions
import re
import spacy
import nltk
from nltk.corpus import stopwords
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging

logger = logging.getLogger(__name__)

nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
nlp_model = spacy.load('en_core_web_sm')


def load_data(input_path):
  logger.info("Started Loading Data")

  try:

    df = pd.read_csv(input_path)
    logger.info("File read successfully")
    return df
  except FileNotFoundError:
    logger.critical(f"File Not Found at {input_path}")
    raise

# Expanding Contractions
def expand_contractions(text):
  return contractions.fix(text)

# Tokenization
def spacy_tokenize(text):
  doc = nlp(text)
  tokens = [token.text.strip() for token in doc]
  return tokens


# Keeps only letters, numbers, and spaces
def remove_special_chars_regex(text):
    temp = []
    for word in text:
        cleaned_word = re.sub(r'[^a-zA-Z0-9\s]', '', word)
        if cleaned_word != '':
            temp.append(cleaned_word)

    return temp


def remove_stopwords(words):

  temp = []
  for word in words:
    if word not in stop_words:
      temp.append(word)

  return temp


def lemmatize(words):
  temp = []
  for word in words:
    doc = nlp_model(word)
    temp.append(doc[0].lemma_)
  return temp


def clean_data(df):

  logger.info("Cleaning Data Start")

   # Transforming text to lower case.
  df['text'] = df['text'].str.lower()

  # Expand the Contractions
  df['text'] = df['text'].apply(expand_contractions)

  # Tokenization
  df['text'] = df['text'].apply(spacy_tokenize)

  # Remove the special Characters
  df['text'] = df['text'].apply(remove_special_chars_regex)

  # Remove Stopwords
  df['text'] = df['text'].apply(remove_stopwords)

  # Apply lemmatization
  df['text'] = df['text'].apply(lemmatize)

  # Rejoining the words to form a sentence
  df['text'] =  df['text'].str.join(' ').str.strip()

  # Removing null values and duplicates
  df = df.drop_duplicates()

  df = df.dropna(subset=['text'])

  df = df[df['text'] != '']

  logger.info("Cleaning Data End")

  return df

# Get the params from the YAML file.
def get_params_yaml(file_path):
  logger.info("Getting Params Start")
  try:
    with open(file_path, 'r') as f:
      params = yaml.safe_load(f)

    test_size = params['data_cleaning']['test_size']

    logger.info("Getting params End")
    return test_size
  except FileNotFoundError:
    logger.critical(f"YAML File not found at {file_path}")
    raise


def save(df, file_path):
  logger.info("Saving Data Start")
  try:
    df.to_csv(file_path, index=False, header=True)
    logger.info("Saving Data End")
  except Exception:
    logger.critical(f"Exception while saving data at {file_path}")
    raise


def main():
  logger.info(f'=' * 50)
  logger.info(f'Stage Data Cleaning')
  logger.info(f'=' * 50)


  # Create the input path and output directory
  input_path = os.path.join("data", "interim", "data.csv")
  output_dir = os.path.join("data", "interim")

  # Load the data
  df = load_data(input_path)

  # clean the data
  df = clean_data(df)

  # Split the data into training and testing data
  X = df.drop(columns = 'class')
  y = df['class']

  test_size = get_params_yaml('params.yaml')

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=32)

  # Save the data
  save(X_train, os.path.join(output_dir, 'X_train.csv'))
  save(X_test, os.path.join(output_dir, 'X_test.csv'))
  save(y_train, os.path.join(output_dir, 'y_train.csv'))
  save(y_test, os.path.join(output_dir, 'y_test.csv'))

  logger.info(f'Stage Data Cleaning END')


if __name__ == '__main__':
  logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [
      logging.FileHandler('logs.log', mode = 'a'),
      logging.StreamHandler()
    ]
  )
  
  main()
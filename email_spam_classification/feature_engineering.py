from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
import yaml

def load_data(input_path):
    df = pd.read_csv(input_path)
    return df

def get_params_yaml(file_path):
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    ngram_range = eval(params['feature_engineering']['ngram_range'])
    max_features = int(params['feature_engineering']['max_features'])

    return ngram_range, max_features

def save(df, output_path):
    df.to_csv(output_path, index=False)



def main():

    # Define the paths of input and output directories
    input_dir = os.path.join("data", "interim")
    output_dir = os.path.join("data", "processed")

    # Load the datasets
    X_train = load_data(os.path.join(input_dir, 'X_train.csv'))
    X_test = load_data(os.path.join(input_dir, 'X_test.csv'))
    y_train = load_data(os.path.join(input_dir, 'y_train.csv'))
    y_test = load_data(os.path.join(input_dir, 'y_test.csv'))

    # Get the parameters
    ngram_range, max_features = get_params_yaml('params.yaml')

    # Perform Text Vectorization
    cv = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)

    X_train = pd.DataFrame(cv.fit_transform(X_train['text']).toarray())
    X_test = pd.DataFrame(cv.transform(X_test['text']).toarray())


    # Save the data to the output directory
    save(X_train, os.path.join(output_dir, 'X_train.csv'))
    save(X_test, os.path.join(output_dir, 'X_test.csv'))
    save(y_train, os.path.join(output_dir, 'y_train.csv'))
    save(y_test, os.path.join(output_dir, 'y_test.csv'))

main()
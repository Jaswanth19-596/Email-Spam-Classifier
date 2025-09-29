from sklearn.naive_bayes import GaussianNB
import os
import pandas as pd
import pickle

def load_data(input_path):
    df = pd.read_csv(input_path)
    return df

def save_model(model, file_path):
    pickle.dump(model, open(file_path, 'wb'))


def main():

    # Define input directory
    input_dir = os.path.join("data", "processed")

    # Load the data
    X_train = load_data(os.path.join(input_dir, 'X_train.csv'))
    y_train = load_data(os.path.join(input_dir, 'y_train.csv'))
    y_train = y_train.values.ravel()

    # Train the model
    gnb = GaussianNB()
    gnb.fit(X_train,y_train)

    # Save the model
    save_model(gnb, 'models/model.pkl')

main()

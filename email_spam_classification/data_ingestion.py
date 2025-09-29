import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


def load_data(input_path):
    df = pd.read_csv(input_path, encoding = 'latin_1')
    return df
    
def basic_cleaning(df):
    # Remove last three features
    df = df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

    # Rename the columns
    df = df.rename({'v1': 'class', 'v2': 'text'}, axis = 1)

    # Encoding the classes
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])

    # Drop duplicated rows
    df = df.drop_duplicates()
    return df

def save(df, output_path):
    df.to_csv(output_path, index = False)
    

def main():
    input_path = os.path.join( "data", "raw", "emails.csv")
    output_dir = os.path.join("data", "interim")
    output_path = os.path.join(output_dir, 'data.csv')
    df = load_data(input_path)
    df = basic_cleaning(df)
    save(df, output_path)


main()
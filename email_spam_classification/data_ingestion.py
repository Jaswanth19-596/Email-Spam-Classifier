import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


input_path = os.path.join( "data", "raw", "emails.csv")


df = pd.read_csv(input_path, encoding = 'latin_1')


# Remove last three features
df = df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

# Rename the columns
df = df.rename({'v1': 'class', 'v2': 'text'}, axis = 1)

# Encoding the classes
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

# Drop duplicated rows
df = df.drop_duplicates()


output_dir = os.path.join("data", "interim")

df.to_csv(os.path.join(output_dir, 'data.csv'), index = False)


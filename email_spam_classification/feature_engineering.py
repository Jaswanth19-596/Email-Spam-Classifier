from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd


input_dir = os.path.join("data", "interim")

X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))
y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv"))



cv = CountVectorizer(ngram_range=(1,2), max_features=1000)

X_train = pd.DataFrame(cv.fit_transform(X_train['text']).toarray())
X_test = pd.DataFrame(cv.transform(X_test['text']).toarray())

output_dir = os.path.join("data", "processed")
X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score, recall_score, roc_auc_score
import os

model = pickle.load(open('models/model.pkl', 'rb'))

input_dir = os.path.join("data", "processed")

X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv'))

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
}

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent = 4)
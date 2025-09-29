from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score, recall_score
import os
import pandas as pd
import pickle

input_dir = os.path.join("data", "processed")

X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

y_train = y_train.values.ravel()


gnb = GaussianNB()
gnb.fit(X_train,y_train)

pickle.dump(gnb, open('model.pkl', 'wb'))


import contractions
import re
import spacy
import nltk
from nltk.corpus import stopwords
import os
import pandas as pd
from sklearn.model_selection import train_test_split


nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')


input_path = os.path.join("data", "interim", "data.csv")



df = pd.read_csv(input_path)


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


stop_words = set(stopwords.words('english'))

def remove_stopwords(words):
  temp = []
  for word in words:
    if word not in stop_words:
      temp.append(word)

  return temp


#### Using Spacy's Lemmatization to lemmatize words.
nlp_model = spacy.load('en_core_web_sm')

def lemmatize(words):
  temp = []
  for word in words:
    doc = nlp_model(word)
    temp.append(doc[0].lemma_)
  return temp


# After good observation, Spacy done a good job. Removing NLTK

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


X = df.drop(columns = 'class')
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=32)

output_dir = os.path.join("data", "interim")
X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False, header=True)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False, header=True)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False, header=True)
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=True)

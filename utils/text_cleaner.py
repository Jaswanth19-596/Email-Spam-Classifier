import contractions
import re
import spacy
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
import pandas as pd

nltk.download("stopwords")

class TextCleaner(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.nlp_model = spacy.load("en_core_web_sm")

    # Expanding Contractions
    def expand_contractions(self, text):
        return contractions.fix(text)

    # Tokenization
    def spacy_tokenize(self, text):
        doc = self.nlp_model(text)
        tokens = [token.text.strip() for token in doc]
        return tokens

    # Keeps only letters, numbers, and spaces
    def remove_special_chars_regex(self, text):
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return cleaned_text

    def remove_stopwords(self, words):
        temp = []
        for word in words:
            if word not in self.stop_words:
                temp.append(word)
        return " ".join(temp)

    def lemmatize(self, text):
        temp = []
        doc = self.nlp_model(text)

        for token in doc:
            temp.append(token.lemma_)
        return " ".join(temp)

    def clean_data(self, text):
        text = text.lower()
        text = self.expand_contractions(text)
        text = self.remove_special_chars_regex(text)
        tokens = self.spacy_tokenize(text)
        text = self.remove_stopwords(tokens)
        if len(text) == 0:
            return "Empty"
        text = self.lemmatize(text)

        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        return X.apply(self.clean_data)

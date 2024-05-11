# data_preprocessing.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras import Tokenizer
from keras import pad_sequences


nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, max_words=5000, max_len=200):
        self.tokenizer = Tokenizer(num_words=max_words)
        self.max_len = max_len
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.strip()
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)

    def preprocess_texts(self, texts):
        texts = [self.clean_text(text) for text in texts]
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len)

from data_preprocessing import clean_text
from feature_extraction import create_tfidf_features
from train_model import train_model, save_model
from utils import save_object
import pandas as pd


# Load the CSV file, assuming no header row in the file
datas = pd.read_csv('../dataset/Suicide_Detection.csv')
data = datas.head(2500)
# Preprocess and extract features
# data['clean_text'] = data['text'].apply(clean_text)
data.loc[:, 'clean_text'] = data['text'].apply(clean_text)
print(data.columns)
X, vectorizer = create_tfidf_features(data['clean_text'])

# Train and save the model
model, X_test, y_test = train_model(X, data['class'])
save_model(model, 'model.pkl')
save_object(vectorizer, 'vectorizer.pkl')

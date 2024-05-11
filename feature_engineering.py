import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Ensure text is a string
    if not isinstance(text, str):
        return ""  # Return empty string if text is not a string
    
    text = text.lower()  # Lowercase text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # Remove non-letters
    text = text.strip()  # Remove whitespace
    tokens = text.split()  # Tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

# Load your CSV file
data = pd.read_csv('./dataset/Suicide_Detection.csv')
print(data.columns)


# Apply the cleaning function to the 'Tweet' column
data['Clean_Tweet'] = data['text'].apply(clean_text)

# Check the first few entries of the cleaned data
print(data[['text', 'Clean_Tweet']].head())
# Assuming 'data' is your DataFrame and it has the 'Clean_Tweet' column with cleaned data

# Save the DataFrame to a new CSV file
data.to_csv('cleaned_tweets.csv', index=False)

# If you want to include only specific columns, you can specify them like this:
data[['Clean_Tweet']].to_csv('cleaned_tweets.csv', index=False)


from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming 'data' is your DataFrame and 'Clean_Tweet' contains the preprocessed text
vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of max_features
tfidf_features = vectorizer.fit_transform(data['Clean_Tweet'])

# This creates a matrix of TF-IDF features


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define the model and parameters
model = LogisticRegression()
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'solver': ['liblinear', 'lbfgs'],  # Optimizer to use
    'class_weight': [None, 'balanced']  # Option to handle imbalanced data
}

# Setup the grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(tfidf_features,data['class'].apply(lambda x: "Potential Suicide post" if x == "suicide" else "Not Suicide post"))

# Best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best score: {:.2f}".format(grid_search.best_score_))

from sklearn.metrics import classification_report, confusion_matrix

# Predict using the best estimator
best_model = grid_search.best_estimator_
predictions = best_model.predict(tfidf_features)

# Evaluation
print(classification_report(data['class'].apply(lambda x: "Potential Suicide post" if x == "suicide" else "Not Suicide post"), predictions))
print("Confusion Matrix:\n", confusion_matrix(data['class'].apply(lambda x: "Potential Suicide post" if x == "suicide" else "Not Suicide post"), predictions))


import joblib

# Save the model
joblib.dump(best_model, 'suicidal_tendency_detector.pkl')

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text is re.sub(r'@\w+', '', text)
    text is re.sub(r'#\w+', '', text)
    text is re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Load and preprocess data
data = pd.read_csv('/content/drive/MyDrive/dataset/Suicide_Detection.csv')
data['Clean_Tweet'] = data['text'].apply(clean_text)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['Clean_Tweet'])
sequences = tokenizer.texts_to_sequences(data['Clean_Tweet'])
padded_sequences = pad_sequences(sequences, maxlen=200)

# Labels
labels = data['class'].apply(lambda x: 1 if x == "suicide" else 0).values
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Model with metrics
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=5000+1, output_dim=100, input_length=200),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

model = build_model()
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

# Evaluate the model
loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')
print(f'Test Precision: {precision:.2f}')
print(f'Test Recall: {recall:.2f}')
f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
print(f'Test F1 Score: {f1_score:.2f}')

# Confusion Matrix
predictions = (model.predict(X_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

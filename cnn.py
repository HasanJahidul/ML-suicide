import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential



# from text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.Model import Model
# from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention
# from sequence import pad_sequences
# Download necessary NLTK resources
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
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Load and preprocess data
data = pd.read_csv('/content/drive/MyDrive/dataset/Suicide_Detection.csv')
data['clean_text'] = data['text'].apply(clean_text)

# Filter out invalid labels
valid_labels = ['suicide', 'non-suicide']
data = data[data['class'].isin(valid_labels)]

# Map 'non-suicide' to 0 and 'suicide' to 1
label_mapping = {'non-suicide': 0, 'suicide': 1}
data['class'] = data['class'].map(label_mapping)

# Verify the mapping
print(data['class'].unique())

# Tokenize text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['clean_text'])
sequences = tokenizer.texts_to_sequences(data['clean_text'])
X = pad_sequences(sequences, maxlen=200)

# Labels
y = np.array(data['class'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def create_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=5000, output_dim=50),
        tf.keras.layers.Conv1D(32, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_cnn_model()
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

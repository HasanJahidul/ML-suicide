import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.Model import Model
# from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention
from sequence import pad_sequences

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""  # Ensure text is a string
    text = text.lower()  # Lowercase text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-letters
    text = text.strip()  # Remove whitespace
    tokens = text.split()  # Tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

# Load and preprocess data
data = pd.read_csv('./dataset/Suicide_Detection.csv')
data['Clean_Tweet'] = data['text'].apply(clean_text)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['Clean_Tweet'])
sequences = tokenizer.texts_to_sequences(data['Clean_Tweet'])
padded_sequences = pad_sequences(sequences, maxlen=200)

# Labels
labels = data['class'].apply(lambda x: 1 if x == "suicide" else 0).values
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)





# Define the attention mechanism
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
    
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W))
        a = tf.nn.softmax(e, axis=1)
        output = tf.reduce_sum(x * a, axis=1)
        return output

# Build model
def build_attention_model(input_dim, max_length):
    inputs = tf.keras.layers.Input(shape=(max_length,))
    x = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=100)(inputs)
    lstm_out = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    attention_out = AttentionLayer()(lstm_out)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(attention_out)
    model =  tf.keras.Model(inputs,outputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_attention_model(input_dim=5000+1, max_length=200)
model.summary()

from kerastuner import HyperModel, RandomSearch

class TextHyperModel(HyperModel):
    def __init__(self, input_dim, max_length):
        self.input_dim = input_dim
        self.max_length = max_length
    
    def build(self, hp):
        model = build_attention_model(
            input_dim=self.input_dim,
            max_length=self.max_length)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

hypermodel = TextHyperModel(input_dim=5000+1, max_length=200)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='model_tuning',
    project_name='suicide_detection'
)

# Perform hyperparameter tuning
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# Get the best model and evaluate on the test set
best_model = tuner.get_best_models()[0]
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Save the best model
best_model.save('suicidal_tendency_detector_attention.h5')

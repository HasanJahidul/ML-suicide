# model_building.py

import tensorflow as tf
from keras import Model
from keras import Input, Embedding, LSTM, Dense

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
    
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
    
    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W))
        a = tf.nn.softmax(e, axis=1)
        output = tf.reduce_sum(x * a, axis=1)
        return output

def build_attention_model(input_dim, max_length):
    inputs = Input(shape=(max_length,))
    x = Embedding(input_dim=input_dim, output_dim=100)(inputs)
    lstm_out = LSTM(64, return_sequences=True)(x)
    attention_out = AttentionLayer()(lstm_out)
    outputs = Dense(1, activation='sigmoid')(attention_out)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# train_evaluate.py

import tensorflow as tf
from keras import Model
from keras import Input, Embedding, LSTM, Dense
from kerastuner import HyperModel, RandomSearch
from model_building import build_attention_model

class MyHyperModel(HyperModel):
    def __init__(self, input_dim, max_length):
        self.input_dim = input_dim
        self.max_length = max_length
    
    def build(self, hp):
        model = build_attention_model(
            input_dim=self.input_dim,
            max_length=self.max_length
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

def tune_model(X_train, y_train, input_dim, max_length):
    hypermodel = MyHyperModel(input_dim=input_dim, max_length=max_length)

    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=3,
        directory='model_tuning',
        project_name='suicide_detection'
    )

    tuner.search(X_train, y_train, epochs=10, validation_split=0.2)
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model

def train_final_model(X_train, y_train, X_test, y_test, best_model):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', save_best_only=True, monitor='val_loss', mode='min'
    )
    history = best_model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
    )
    return history

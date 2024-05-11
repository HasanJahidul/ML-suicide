from kerastuner import HyperModel, RandomSearch
import tensorflow as tf
from Attention_Mechanism.model_building import build_attention_model

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

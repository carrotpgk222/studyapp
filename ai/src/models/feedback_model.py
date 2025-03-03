import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

def build_feedback_model(num_features):
    inputs = tf.keras.Input(shape=(num_features,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(16, activation='relu')(x)
    # Regression output for study consistency score (e.g., scale 1–5)
    outputs = layers.Dense(1, activation='linear', name='consistency_score')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

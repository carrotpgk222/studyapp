import tensorflow as tf
from tensorflow.keras import layers, Model

def build_time_model(num_features):
    inputs = tf.keras.Input(shape=(num_features,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(16, activation='relu')(x)
    
    # Two outputs: study duration and break time
    study_duration = layers.Dense(1, activation='linear', name='study_duration')(x)
    break_time = layers.Dense(1, activation='linear', name='break_time')(x)
    
    model = Model(inputs, [study_duration, break_time])
    model.compile(optimizer='adam',
                  loss={'study_duration': 'mse', 'break_time': 'mse'},
                  metrics={'study_duration': 'mae', 'break_time': 'mae'})
    return model

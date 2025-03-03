import tensorflow as tf
from tensorflow.keras import layers, Model

def build_subject_model(num_features, num_subjects):
    inputs = tf.keras.Input(shape=(num_features,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(16, activation='relu')(x)
    # Softmax output for subject recommendation
    outputs = layers.Dense(num_subjects, activation='softmax', name='subject_output')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
``
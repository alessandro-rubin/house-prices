import tensorflow as tf
from keras import layers
import numpy as np


def create_preprocessing_layer(numeric_features,categorical_features,tain_df):

    inputs = {}
    for name in numeric_features+categorical_features:
        if name in numeric_features:
            dtype = tf.float32
        if (name in categorical_features):
            dtype = tf.string
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)    

    numeric_inputs = {name:input for name,input in inputs.items()
                    if input.dtype==tf.float32}

    x = layers.Concatenate()(list(numeric_inputs.values()))
    norm = layers.Normalization()
    norm.adapt(np.array(tain_df[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)
    preprocessed_inputs = [all_numeric_inputs]

    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue
    
        lookup = layers.StringLookup(vocabulary=np.unique(tain_df[name]))
        one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)

    preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
    house_prices_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
    return house_prices_preprocessing


def create_model(preprocessing_head, inputs:dict):
    preprocessed_inputs = preprocessing_head(inputs)
    x=layers.Dense(100,activation='relu')(preprocessed_inputs)
    x=layers.Dropout(0.3)(x)
    x=layers.Dense(170,activation='relu')(x)
    x=layers.Dropout(0.3)(x)
    x=layers.Dense(150,activation='relu')(x)
    result = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, result)
    model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanSquaredLogarithmicError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanSquaredLogarithmicError()])
    return model
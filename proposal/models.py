import keras

def cnnlstm_model():
    input_layer = keras.layers.Input(shape=(16,299,299,3))

    feature_extractor = keras.layers.TimeDistributed(keras.applications.Xception(
        include_top=False, weights='imagenet'), trainable=False)(input_layer)

    recurrent_layer = keras.layers.ConvLSTM2D(128, kernel_size=3)(feature_extractor)

    batch_norm = keras.layers.BatchNormalization()(recurrent_layer)
    dropout = keras.layers.Dropout(0.6)(batch_norm)
    flatten = keras.layers.Flatten()(dropout)
    linear = keras.layers.Dense(128)(flatten)
    linear = keras.layers.Dropout(0.6)(linear)
    relu = keras.layers.Activation('relu')(linear)
    predictions = keras.layers.Dense(101, activation='softmax')(relu)

    model = keras.models.Model(inputs=input_layer, outputs=predictions)
    return model


def recurrent_feats_model():

    xception = keras.applications.Xception(include_top=True, weights='imagenet')

    extractor = keras.models.Model(inputs=xception.layers[0].input,
                                   outputs=xception.layers[-2].output)
    for layer in extractor.layers:
        layer.trainable=False

    input_layer = keras.layers.Input((None,299,299,3))
    td_layer = keras.layers.TimeDistributed(extractor)(input_layer)
    
    recurrent_layer = keras.layers.LSTM(
        512,
        return_sequences=False,
        dropout=0.5
    )(td_layer)
    linear = keras.layers.Dense(256, activation='relu')(recurrent_layer)
    linear = keras.layers.Dropout(0.5)(linear)
    predictions = keras.layers.Dense(101, activation='softmax')(linear)

    model = keras.models.Model(inputs=input_layer, outputs=predictions)
    return model

def recurrent_backbone():
    input_layer = keras.layers.Input((25,2048))
    recurrent_layer = keras.layers.LSTM(
        512,
        return_sequences=False,
        dropout=0.5
    )(input_layer)
    linear = keras.layers.Dense(256, activation='relu')(recurrent_layer)
    linear = keras.layers.Dropout(0.5)(linear)
    predictions = keras.layers.Dense(101, activation='softmax')(linear)

    model = keras.models.Model(inputs=input_layer, outputs=predictions)
    return model

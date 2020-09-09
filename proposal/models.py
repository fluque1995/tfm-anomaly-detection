import keras

def recurrent_feats_model():

    xception = keras.applications.Xception(include_top=True, weights='imagenet')

    extractor = keras.models.Model(inputs=xception.layers[0].input,
                                   outputs=xception.layers[-2].output)
    for layer in extractor.layers:
        layer.trainable=False

    input_layer = keras.layers.Input((None,299,299,3))
    td_layer = keras.layers.TimeDistributed(extractor)(input_layer)

    recurrent_layer = keras.layers.LSTM(
        1024,
        return_sequences=False,
        dropout=0.6
    )(td_layer)
    linear = keras.layers.Dense(512, activation='relu')(recurrent_layer)
    linear = keras.layers.Dropout(0.5)(linear)
    linear = keras.layers.Dense(128, activation='relu')(linear)
    linear = keras.layers.Dropout(0.5)(linear)
    predictions = keras.layers.Dense(101, activation='softmax')(linear)

    model = keras.models.Model(inputs=input_layer, outputs=predictions)
    return model

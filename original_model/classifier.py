import keras
import scipy.io as sio
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2

import configuration as cfg

def classifier_model():
    """Build the classifier

    :returns: Classifier model
    :rtype: keras.Model

    """
    model = Sequential()
    model.add(Dense(512, input_dim=4096, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(0.001), activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(1, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(0.001), activation='sigmoid'))
    return model


def build_classifier_model():
    """Build the classifier and load the pretrained weights

    :returns:
    :rtype:

    """
    model = classifier_model()
    model = load_weights(model, cfg.classifier_model_weigts)
    return model


def conv_dict(dict2):
    """Prepare the dictionary of weights to be loaded by the network

    :param dict2: Dictionary to format
    :returns: The dictionary properly formatted
    :rtype: dict

    """
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict


def load_weights(model, weights_file):
    """Loads the pretrained weights into the network architecture

    :param model: keras model of the network
    :param weights_file: Path to the weights file
    :returns: The input model with the weights properly loaded
    :rtype: keras.model

    """
    dict2 = sio.loadmat(weights_file)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

if __name__ == '__main__':
    model = build_classifier_model()
    model.summary()

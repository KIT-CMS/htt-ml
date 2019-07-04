from keras.layers import *
from keras.models import Sequential

def model2sequential(model1):
    inputs = model1.layers[0].input_shape[1]
    sequential = Sequential()
    first_layer_nodes = model1.layers[1].output_shape[1]
    sequential.add(Dense(first_layer_nodes,kernel_regularizer=None, input_dim=inputs, name='input'))
    for i, layer in enumerate(model1.layers[2:]):
        sequential.add(layer)
        assert layer.__class__.__name__ == sequential.layers[i+1].__class__.__name__
        sequential.layers[i+1].set_weights(layer.get_weights())
        assert len(sequential.layers[i+1].get_weights()) == len(layer.get_weights())

    if len(model1.layers)-1 != len(sequential.layers):
        raise Exception('Non matching number of layers: Model has {} layers while sequential has {} layers'.format(len(model1.layers), len(sequential.layers)))

    return sequential
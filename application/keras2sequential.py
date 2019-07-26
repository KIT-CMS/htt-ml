from keras.layers import *
from keras.models import Sequential

def model2sequential(model1):
    inputs = model1.layers[0].input_shape[1]
    sequential = Sequential()
    first_layer_nodes = model1.layers[1].output_shape[1]
    sequential.add(Dense(first_layer_nodes,kernel_regularizer=None, input_dim=inputs, name='input'))
    weights = model1.layers[1].get_weights()
    sequential.layers[0].set_weights(weights)
    new_weights = sequential.layers[0].get_weights()
    for list1,list2 in zip(weights,new_weights):
        assert np.array_equal(list1, list2)

    for i, layer in enumerate(model1.layers[2:]):
        sequential.add(layer)
        #sequential.summary()
        assert layer.__class__.__name__ == sequential.layers[i+1].__class__.__name__
        weights_original = layer.get_weights()
        sequential.layers[i+1].set_weights(layer.get_weights())
        weights_new = sequential.layers[i+1].get_weights()
        for list1, list2 in zip(weights_original, weights_new):
            assert np.array_equal(list1, list2)
        assert len(sequential.layers[i+1].get_weights()) == len(layer.get_weights())

    if len(model1.layers)-1 != len(sequential.layers):
        raise Exception('Non matching number of layers: Model has {} layers while sequential has {} layers'.format(len(model1.layers), len(sequential.layers)))
    test = [[np.random.uniform(low=-1, high=1, size=inputs)]]
    response1 = sequential.predict(test)
    response2 = model1.predict(test)
    assert np.array_equal(response1, response2)

    return sequential
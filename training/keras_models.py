import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import l2
import numpy as np
from tensorflow import keras
def functional(num_inputs, num_outputs):
    inputs1 =  keras.Input(shape=(num_inputs,))
    inputs2 =  keras.Input(shape=(num_inputs,))
    dense1 = keras.layers.Dense(200,activation='tanh',kernel_regularizer=l2(1e-5))
    dense2 = keras.layers.Dense(200,activation='tanh',kernel_regularizer=l2(1e-5))
    dense3 = keras.layers.Dense(200,activation='tanh',kernel_regularizer=l2(1e-5))
    dense4 = keras.layers.Dense(200,activation='tanh',kernel_regularizer=l2(1e-5))
    dense5 = keras.layers.Dense(num_outputs,activation='softmax',kernel_regularizer=l2(1e-5))
    dense6 = keras.layers.Dense(num_outputs,activation='softmax',kernel_regularizer=l2(1e-5))
    x1 = dense1(inputs1)
    x2 = dense2(inputs2)
    x1 = Dropout(0.3)(x1)
    x2 = Dropout(0.3)(x2)
    x1 = dense3(x1)
    x2 = dense4(x2)
    x1 = Dropout(0.3)(x1)
    x2 = Dropout(0.3)(x2)
    #x3 = keras.layers.concatenate([x1,x2])
    output1 = dense5(x1)
    output2 = dense6(x2)

    model = keras.Model(inputs=[inputs1,inputs2], outputs=[output1,output2])

    #keras.utils.plot_model(model, 'func_api2.png', show_shapes=True)
    model.compile(loss=["categorical_crossentropy","categorical_crossentropy"],
                    optimizer=Adam(learning_rate=1e-4),
                    weighted_metrics=["mean_squared_error","mean_squared_error"])
                

    return model

def example(num_inputs, num_outputs):
    """
    Example Keras model
    """
    model = Sequential()
    model.add(
        Dense(10,
              init="glorot_normal",
              activation="relu",
              input_dim=num_inputs))
    model.add(Dense(num_outputs, init="glorot_uniform", activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(),
                  metrics=[
                      "categorical_accuracy",
                  ])
    return model


def smhtt_simple(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(100,
              init="glorot_normal",
              activation="tanh",
              input_dim=num_inputs))
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Nadam(), metrics=[])
    return model


def smhtt_mt(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(300,
              init="glorot_normal",
              activation="tanh",
              W_regularizer=l2(1e-4),
              input_dim=num_inputs))
    model.add(
        Dense(300,
              init="glorot_normal",
              activation="tanh",
              W_regularizer=l2(1e-4)))
    model.add(
        Dense(300,
              init="glorot_normal",
              activation="tanh",
              W_regularizer=l2(1e-4)))
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Nadam(), metrics=[])
    return model


def smhtt_et(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(1000,
              init="glorot_normal",
              activation="tanh",
              W_regularizer=l2(1e-4),
              input_dim=num_inputs))
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Nadam(), metrics=[])
    return model


def smhtt_tt(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(200,
              init="glorot_normal",
              activation="tanh",
              W_regularizer=l2(1e-4),
              input_dim=num_inputs))
    model.add(
        Dense(200,
              init="glorot_normal",
              activation="tanh",
              W_regularizer=l2(1e-4)))
    model.add(
        Dense(200,
              init="glorot_normal",
              activation="tanh",
              W_regularizer=l2(1e-4)))
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Nadam(), metrics=[])
    return model


def smhtt_legacy(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(300,
              init="glorot_normal",
              activation="relu",
              W_regularizer=l2(1e-4),
              input_dim=num_inputs))
    model.add(
        Dense(300,
              init="glorot_normal",
              activation="relu",
              W_regularizer=l2(1e-4)))
    model.add(
        Dense(300,
              init="glorot_normal",
              activation="relu",
              W_regularizer=l2(1e-4)))
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Adam(), metrics=[])
    return model


def smhtt_dropout(num_inputs, num_outputs):
    model = Sequential()

    for i, nodes in enumerate([200] * 2):
        if i == 0:
            model.add(Dense(nodes, input_dim=num_inputs))
        else:
            model.add(Dense(nodes))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

    model.add(Dense(num_outputs))
    model.add(Activation("softmax"))

    model.compile(loss="mean_squared_error", optimizer=Nadam())
    return model


def smhtt_dropout_tanh(num_inputs, num_outputs):
    model = Sequential()

    for i, nodes in enumerate([200] * 2):
        if i == 0:
            model.add(
                Dense(nodes, kernel_regularizer=l2(1e-5),
                      input_dim=num_inputs))
        else:
            model.add(Dense(nodes, kernel_regularizer=l2(1e-5)))
        model.add(Activation("tanh"))
        model.add(Dropout(0.3))

    model.add(Dense(num_outputs, kernel_regularizer=l2(1e-5)))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=1e-4),
                  weighted_metrics=["mean_squared_error"])
    return model


def smhtt_dropout_tanh_large(num_inputs, num_outputs):
    model = Sequential()

    for i, nodes in enumerate([512] * 4):
        if i == 0:
            model.add(
                Dense(nodes, kernel_regularizer=l2(1e-5),
                      input_dim=num_inputs))
        else:
            model.add(Dense(nodes, kernel_regularizer=l2(1e-5)))
        model.add(Activation("tanh"))
        model.add(Dropout(0.3))

    model.add(Dense(num_outputs, kernel_regularizer=l2(1e-5)))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=1e-4),
                  weighted_metrics=["mean_squared_error"])
    return model


def smhtt_dropout_tanh_tensorflow(input_placeholder, keras_model):
    weights = {}
    for layer in keras_model.layers:
        print("Layer: {}".format(layer.name))
        for weight, array in zip(layer.weights, layer.get_weights()):
            print("    weight, shape: {}, {}".format(weight.name,
                                                     np.array(array).shape))
            weights[weight.name] = np.array(array)
    w1 = tf.compat.v1.get_variable('w1', initializer=weights['dense/kernel:0'])
    b1 = tf.compat.v1.get_variable('b1', initializer=weights['dense/bias:0'])
    w2 = tf.compat.v1.get_variable('w2',
                                   initializer=weights['dense_1/kernel:0'])
    b2 = tf.compat.v1.get_variable('b2', initializer=weights['dense_1/bias:0'])
    w3 = tf.compat.v1.get_variable('w3',
                                   initializer=weights['dense_2/kernel:0'])
    b3 = tf.compat.v1.get_variable('b3', initializer=weights['dense_2/bias:0'])

    l1 = tf.tanh(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.tanh(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.softmax(tf.add(b3, tf.matmul(l2, w3)))
    return l3


def smhtt_dropout_tensorflow(input_placeholder, keras_model):
    weights = {}
    for layer in keras_model.layers:
        print("Layer: {}".format(layer.name))
        for weight, array in zip(layer.weights, layer.get_weights()):
            print("    weight, shape: {}, {}".format(weight.name,
                                                     np.array(array).shape))
            weights[weight.name] = np.array(array)

    w1 = tf.get_variable('w1', initializer=weights['dense_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['dense_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['dense_2/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['dense_2/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['dense_3/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['dense_3/bias:0'])

    l1 = tf.nn.relu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.relu(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.softmax(tf.add(b3, tf.matmul(l2, w3)))
    return l3

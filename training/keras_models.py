from keras.models import Sequential, load_model, Model
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2
from keras_custom_metrics import *
from functools import partial, update_wrapper

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def example(num_inputs, num_outputs):
    """
    Example Keras model
    """
    model = Sequential()
    model.add(
        Dense(
            10, init="glorot_normal", activation="relu", input_dim=num_inputs))
    model.add(Dense(num_outputs, init="glorot_uniform", activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=[
            "categorical_accuracy",
        ])
    return model



def smhtt_simple(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(
            100, init="glorot_normal", activation="tanh",
            input_dim=num_inputs))
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Nadam(), metrics=[])
    return model


def smhtt_mt(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(
            300,
            init="glorot_normal",
            activation="tanh",
            W_regularizer=l2(1e-4),
            input_dim=num_inputs))
    model.add(
        Dense(
            300,
            init="glorot_normal",
            activation="tanh",
            W_regularizer=l2(1e-4)))
    model.add(
        Dense(
            300,
            init="glorot_normal",
            activation="tanh",
            W_regularizer=l2(1e-4)))
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Nadam(), metrics=[])
    return model


def smhtt_et(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(
            1000,
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
        Dense(
            200,
            init="glorot_normal",
            activation="tanh",
            W_regularizer=l2(1e-4),
            input_dim=num_inputs))
    model.add(
        Dense(
            200,
            init="glorot_normal",
            activation="tanh",
            W_regularizer=l2(1e-4)))
    model.add(
        Dense(
            200,
            init="glorot_normal",
            activation="tanh",
            W_regularizer=l2(1e-4)))
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Nadam(), metrics=[])
    return model


def smhtt_legacy(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(
            300,
            init="glorot_normal",
            activation="relu",
            W_regularizer=l2(1e-4),
            input_dim=num_inputs))
    model.add(
        Dense(
            300,
            init="glorot_normal",
            activation="relu",
            W_regularizer=l2(1e-4)))
    model.add(
        Dense(
            300,
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

def smhtt_dropout_relu(num_inputs, num_outputs):
    model = Sequential()

    for i, nodes in enumerate([200] * 2):
        if i == 0:
            model.add(Dense(nodes, input_dim=num_inputs))
        else:
            model.add(Dense(nodes))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        #model.add(Dropout(0.5))

    model.add(Dense(num_outputs))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=2e-2, decay=0.01))
    return model

def smhtt_significance(num_inputs, num_outputs):
    inputs = Input(shape=(num_inputs,))
    weights = Input(shape=(1,))

    layer_1 = Dense(200, activation=None, kernel_regularizer=None)(inputs)
    layer_1 = BatchNormalization()(layer_1)
    layer_1 = Activation('relu')(layer_1)
    #layer_1 = Dropout(rate=0.3)(layer_1)
    layer_2 = Dense(200, activation=None, kernel_regularizer=None)(layer_1)
    layer_2 = BatchNormalization()(layer_2)
    layer_2 = Activation('relu')(layer_2)
    #layer_2 = Dropout(rate=0.3)(layer_2)
    output_ggh = Dense(num_outputs, activation=None, kernel_regularizer=None, name=None)(layer_2)
    output_ggh = BatchNormalization(name=None)(output_ggh)
    output_ggh = Activation('softmax', name='ggh')(output_ggh)
    output_qqh = Lambda(lambda x: x, name='qqh')(output_ggh)
    output_ztt = Lambda(lambda x: x, name='ztt')(output_ggh)
    output_noniso = Lambda(lambda x: x, name='noniso')(output_ggh)
    output_misc = Lambda(lambda x: x, name='misc')(output_ggh)

    model = Model(inputs=[inputs, weights], outputs=[output_ggh, output_qqh, output_ztt, output_noniso, output_misc])

    loss_dict = dict()
    loss_dict['ggh'] = wrapped_partial(curry_loss_2(class_label=0), weights=weights)
    loss_dict['qqh'] = wrapped_partial(curry_loss_2(class_label=1), weights=weights)
    loss_dict['ztt'] = wrapped_partial(curry_loss_2(class_label=2), weights=weights)
    loss_dict['noniso'] = wrapped_partial(curry_loss_2(class_label=3), weights=weights)
    loss_dict['misc'] = wrapped_partial(curry_loss_2(class_label=4), weights=weights)

    model.compile(loss=loss_dict, optimizer=Adam(lr=1e-3), loss_weights=[1.,1.,1.,1.,1.])
    return model

def smhtt_ams(num_inputs, num_outputs):
    inputs = Input(shape=(num_inputs,))
    weights = Input(shape=(1,))

    layer_1 = Dense(200, activation=None, kernel_regularizer=None)(inputs)
    layer_1 = BatchNormalization()(layer_1)
    layer_1 = Activation('relu')(layer_1)
    #layer_1 = Dropout(rate=0.3)(layer_1)
    layer_2 = Dense(200, activation=None, kernel_regularizer=None)(layer_1)
    layer_2 = BatchNormalization()(layer_2)
    layer_2 = Activation('relu')(layer_2)
    #layer_2 = Dropout(rate=0.3)(layer_2)
    output_ggh = Dense(num_outputs, activation=None, kernel_regularizer=None, name=None)(layer_2)
    output_ggh = BatchNormalization()(output_ggh)
    output_ggh = Activation('softmax', name='ggh')(output_ggh)
    output_qqh = Lambda(lambda x: x, name='qqh')(output_ggh)
    output_ztt = Lambda(lambda x: x, name='ztt')(output_ggh)
    output_noniso = Lambda(lambda x: x, name='noniso')(output_ggh)
    output_misc = Lambda(lambda x: x, name='misc')(output_ggh)

    model = Model(inputs=[inputs, weights], outputs=[output_ggh, output_qqh, output_ztt, output_noniso, output_misc])

    loss_dict = dict()
    loss_dict['ggh'] = wrapped_partial(ams_curry_loss(class_label=0), weights=weights)
    loss_dict['qqh'] = wrapped_partial(ams_curry_loss(class_label=1), weights=weights)
    loss_dict['ztt'] = wrapped_partial(ams_curry_loss(class_label=2), weights=weights)
    loss_dict['noniso'] = wrapped_partial(ams_curry_loss(class_label=3), weights=weights)
    loss_dict['misc'] = wrapped_partial(ams_curry_loss(class_label=4), weights=weights)

    model.compile(loss=loss_dict, optimizer=Adam(lr=1e-3), loss_weights=[1.,1.,1.,1.,1.])
    return model


def smhtt_dropout_tanh(num_inputs, num_outputs):
    model = Sequential()

    for i, nodes in enumerate([200] * 2):
        if i == 0:
            model.add(Dense(nodes, kernel_regularizer=l2(1e-5), input_dim=num_inputs))
        else:
            model.add(Dense(nodes, kernel_regularizer=l2(1e-5)))
        model.add(Activation("tanh"))
        model.add(Dropout(0.3))

    model.add(Dense(num_outputs, kernel_regularizer=l2(1e-5)))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), weighted_metrics=["mean_squared_error"], metrics=['accuracy'])
    return model


def smhtt_dropout_tanh_tensorflow(input_placeholder, keras_model):
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
